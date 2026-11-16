"""Entropic FGW with free recruitment rows and a free loss column."""

import copy
from functools import partial
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp, xlogy
from ott.geometry import geometry

from moscot._types import ArrayLike, Device_t, DTypeLike
from moscot.backends.ott.solver import OTTJaxSolver
from moscot.base.output import MatrixSolverOutput
from moscot.utils.tagged_array import TaggedArray


def _quadratic(p: jax.Array, x: geometry.Geometry, y: geometry.Geometry) -> tuple[jax.Array, jax.Array]:
    """Return Q and its gradient, including derivatives of the variable resident marginals."""
    a, b = p.sum(1)[:, None], p.sum(0)[:, None]
    left = x.apply_square_cost(a, axis=1) + x.apply_square_cost(a, axis=0)
    right = y.apply_square_cost(b, axis=1) + y.apply_square_cost(b, axis=0)
    cross = y.apply_cost(x.apply_cost(p, axis=1).T, axis=1).T
    cross_t = y.apply_cost(x.apply_cost(p, axis=0).T, axis=0).T
    grad = left + right.T - 2 * (cross + cross_t)
    return 0.5 * jnp.sum(p * grad), grad


def _residual(p: jax.Array, r: jax.Array, s: jax.Array, q: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.maximum(jnp.max(jnp.abs(p.sum(1) + s - q)), jnp.max(jnp.abs(p.sum(0) + r.sum(0) - b)))


def _linear_solve(
    d: jax.Array,
    cr: jax.Array,
    cs: jax.Array,
    q: jax.Array,
    b: jax.Array,
    epsilon: float,
    threshold: float,
    max_iterations: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dtype = jnp.result_type(d, cr, cs, q, b)
    d, cr, cs, q, b = (jnp.asarray(t, dtype=dtype) for t in (d, cr, cs, q, b))
    kp, kr, ks = -d / epsilon, -cr / epsilon, -cs / epsilon
    recruitment = logsumexp(kr, axis=0)

    def coupling(u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jnp.exp(kp + u[:, None] + v[None, :]), jnp.exp(kr + v[None, :]), jnp.exp(ks + u)

    def step(state: tuple[Any, ...]) -> tuple[Any, ...]:
        i, _, v, _ = state
        denominator = jnp.logaddexp(logsumexp(kp + v[None, :], axis=1), ks)
        u = jnp.where(q > 0, jnp.log(q) - denominator, -jnp.inf)
        denominator = jnp.logaddexp(logsumexp(kp + u[:, None], axis=0), recruitment)
        v = jnp.where(b > 0, jnp.log(b) - denominator, -jnp.inf)
        p, r, s = coupling(u, v)
        return i + 1, u, v, _residual(p, r, s, q, b)

    initial = (jnp.asarray(0), jnp.zeros_like(q), jnp.zeros_like(b), jnp.asarray(jnp.inf, dtype=q.dtype))
    i, u, v, error = jax.lax.while_loop(lambda z: (z[0] < max_iterations) & (z[3] > threshold), step, initial)
    p, r, s = coupling(u, v)
    return p, r, s, error, i


class OpenSystemOutput(MatrixSolverOutput):
    """Open-system solution in units of the target tissue count.

    ``transport_matrix``, ``a``, ``b``, push and pull refer to resident transport only.
    Recruitment and loss are available explicitly; the augmented matrix is built on demand.
    """

    def __init__(self, p: jax.Array, r: jax.Array, s: jax.Array, *, diagnostics: dict[str, Any], **kwargs: Any):
        super().__init__(p, **kwargs)
        self.recruitment = r
        self.loss = s
        self.diagnostics = diagnostics

    @property
    def P(self) -> jax.Array:  # noqa: N802
        """Resident-to-resident coupling."""
        return cast(jax.Array, self.transport_matrix)

    @property
    def R(self) -> jax.Array:  # noqa: N802
        """Circulating-to-resident coupling."""
        return self.recruitment

    @property
    def S(self) -> jax.Array:  # noqa: N802
        """Resident loss vector."""
        return self.loss

    @property
    def augmented_transport_matrix(self) -> jax.Array:
        """Materialize [[P, S], [R, 0]] with a structural zero lower-right block."""
        return jnp.concatenate(
            (jnp.column_stack((self.P, self.S)), jnp.column_stack((self.R, jnp.zeros_like(self.R[:, 0]))))
        )

    @property
    def P_hat(self) -> jax.Array:  # noqa: N802
        """Augmented coupling, materialized on demand."""
        return self.augmented_transport_matrix

    @property
    def recruitment_mass(self) -> jax.Array:
        """Recruitment mass per target cell."""
        return self.R.sum(0)

    @property
    def resident_contribution(self) -> jax.Array:
        """Resident contribution per target cell."""
        return self.P.sum(0)

    @property
    def total_recruitment(self) -> float:
        """Total recruitment mass."""
        return float(self.R.sum())

    @property
    def total_loss(self) -> float:
        """Total loss mass."""
        return float(self.S.sum())

    def to(self, device: Optional[Device_t] = None, dtype: Optional[DTypeLike] = None) -> "OpenSystemOutput":
        """Transfer all three transport blocks to a JAX device."""
        if isinstance(device, str):
            platform, _, index = device.partition(":")
            device = jax.devices(platform)[int(index or 0)]
        result = copy.copy(self)
        for name in ("_transport_matrix", "recruitment", "loss"):
            value = getattr(self, name)
            setattr(result, name, jax.device_put(value if dtype is None else value.astype(dtype), device))
        return result


class OpenSystemSolver(OTTJaxSolver):
    """Generalized Sinkhorn subproblems with a damped resident FGW outer iteration.

    Spatial geometry products use OTT's online/batched operators. Resident and recruitment
    couplings and their linear costs are dense. The host controls the outer line search;
    geometry products and inner scaling can be JIT compiled.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        epsilon: float = 1e-3,
        threshold: float = 1e-6,
        max_iterations: int = 200,
        min_iterations: int = 1,
        linear_solver_kwargs: Optional[dict[str, Any]] = None,
        jit: bool = True,
    ):
        super().__init__(jit=jit)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("`alpha` must be in [0, 1].")
        if not isinstance(epsilon, (int, float, np.floating)) or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("`epsilon` must be a finite positive scalar.")
        linear = dict(linear_solver_kwargs or {})
        self.inner_threshold = linear.pop("threshold", 1e-7)
        self.inner_iterations = linear.pop("max_iterations", 10000)
        if linear:
            raise TypeError(f"Unsupported open-system linear solver arguments: {tuple(linear)}.")
        for name, value in (("threshold", threshold), ("inner threshold", self.inner_threshold)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"`{name}` must be finite and positive.")
        for value in (max_iterations, min_iterations, self.inner_iterations):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Iteration counts must be positive integers.")
        if min_iterations > max_iterations:
            raise ValueError("`min_iterations` exceeds `max_iterations`.")
        self.alpha, self.epsilon, self.threshold = alpha, float(epsilon), threshold
        self.max_iterations, self.min_iterations = max_iterations, min_iterations

    @property
    def problem_kind(self) -> str:
        """Kind of the resident transport problem."""
        return "quadratic"

    def _prepare(
        self,
        *,
        xy: TaggedArray,
        x: TaggedArray,
        y: TaggedArray,
        a: ArrayLike,
        b: ArrayLike,
        recruitment: TaggedArray,
        recruitment_penalty: ArrayLike,
        sink_cost: ArrayLike,
        batch_size: Optional[int] = None,
    ) -> tuple[Any, ...]:
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("`batch_size` must be a positive integer.")
        for tagged in (x, y, xy, recruitment):
            if not (tagged.is_point_cloud or tagged.is_cost_matrix):
                raise ValueError("Open-system geometries must be point clouds or explicit cost matrices.")
            for value in (tagged.data_src, tagged.data_tgt):
                if value is not None and not np.all(np.isfinite(value)):
                    raise ValueError("Resident and recruitment geometry data must be finite.")
        geoms = [self._create_geometry(t, is_linear_term=False, batch_size=batch_size) for t in (x, y)]
        c = self._create_geometry(xy, is_linear_term=True, batch_size=batch_size).cost_matrix
        if recruitment.shape[0]:
            cr = (1 - self.alpha) * self._create_geometry(
                recruitment, is_linear_term=True, batch_size=batch_size
            ).cost_matrix + jnp.asarray(recruitment_penalty)
        else:
            cr = jnp.empty((0, len(b)), dtype=c.dtype)
        cr = cr.astype(c.dtype)
        q, b, cs = (jnp.asarray(v, dtype=c.dtype) for v in (a, b, sink_cost))
        if c.shape != (len(q), len(b)) or cr.shape[1] != len(b) or cs.shape != q.shape:
            raise ValueError("Open-system cost and mass shapes do not match.")
        if not len(q) or not len(b):
            raise ValueError("Empty resident subsets are not supported.")
        if geoms[0].shape != (len(q), len(q)) or geoms[1].shape != (len(b), len(b)):
            raise ValueError("Resident spatial geometry shapes do not match the masses.")
        for name, value in (("source mass", q), ("target mass", b), ("resident cost", c)):
            if not np.all(np.isfinite(value)) or np.any(np.asarray(value) < 0):
                raise ValueError(f"`{name}` must be finite and nonnegative.")
        for value in (cr, cs):
            if np.any(np.isnan(value)) or np.any(np.asarray(value) < 0):
                raise ValueError("Recruitment and sink costs must be nonnegative, possibly +infinity.")
        # P is fully connected; these two capacity inequalities are necessary and sufficient.
        mandatory_source = np.isinf(cs)
        mandatory_target = np.all(np.isinf(cr), axis=0)
        source_min, target_min = float(q[mandatory_source].sum()), float(b[mandatory_target].sum())
        source_total, target_total = float(q.sum()), float(b.sum())
        tol = self.inner_threshold * 0.1
        if source_min > target_total + tol or target_min > source_total + tol:
            raise ValueError("Infeasible open-system constraints: insufficient recruitment or loss capacity.")
        allowed = np.ones(c.shape, dtype=bool)
        # Remove forced zeros on boundary faces, where finite dual potentials need not exist.
        if source_min == target_total:
            allowed[~mandatory_source, :] = False
            cr = jnp.full_like(cr, jnp.inf)
        if target_min == source_total:
            allowed[:, ~mandatory_target] = False
            cs = jnp.full_like(cs, jnp.inf)
        data = (geoms[0], geoms[1], c, cr, cs, q, b, jnp.asarray(allowed))
        self._problem = data
        return data

    def _solve(self, data: tuple[Any, ...], **_: Any) -> OpenSystemOutput:
        x, y, c, cr, cs, q, b, allowed = data
        quadratic = partial(_quadratic, x=x, y=y)
        inner = partial(
            _linear_solve,
            cr=cr,
            cs=cs,
            q=q,
            b=b,
            epsilon=self.epsilon,
            # Leave rounding headroom for interpolation and independently evaluated residuals.
            threshold=0.5 * self.inner_threshold,
            max_iterations=self.inner_iterations,
        )
        if self._jit:
            quadratic, inner = jax.jit(quadratic), jax.jit(inner)

        def objective(p: jax.Array, r: jax.Array, s: jax.Array) -> jax.Array:
            value = (1 - self.alpha) * jnp.sum(c * p) + self.alpha * quadratic(p)[0]
            value += jnp.sum(jnp.where(jnp.isfinite(cr), cr, 0) * r)
            value += jnp.sum(jnp.where(jnp.isfinite(cs), cs, 0) * s)
            return value + self.epsilon * sum(jnp.sum(xlogy(t, t) - t) for t in (p, r, s))

        if self._jit:
            objective = jax.jit(objective)
        p, r, s, error, count = inner(jnp.where(allowed, (1 - self.alpha) * c, jnp.inf))
        if not np.isfinite(error) or float(error) > self.inner_threshold:
            raise RuntimeError(
                "Initial open-system scaling did not reach feasibility; increase inner iterations or epsilon."
            )
        history = [float(objective(p, r, s))]
        if not np.isfinite(history[0]):
            raise ValueError("Non-finite open-system objective; check spatial costs and their numerical scale.")
        inner_counts = [int(count)]
        converged, status, gap = False, "max_iterations", np.inf
        for iteration in range(self.max_iterations):
            grad = (1 - self.alpha) * c + self.alpha * quadratic(p)[1]
            pn, rn, sn, error, count = inner(jnp.where(allowed, grad, jnp.inf))
            inner_counts.append(int(count))
            if not np.isfinite(error) or float(error) > self.inner_threshold:
                status = "inner_not_converged"
                break
            # Generalized conditional-gradient gap, keeping entropy in the convex subproblem.
            linear_delta = jnp.sum(grad * (pn - p))
            linear_delta += jnp.sum(jnp.where(jnp.isfinite(cr), cr, 0) * (rn - r))
            linear_delta += jnp.sum(jnp.where(jnp.isfinite(cs), cs, 0) * (sn - s))
            entropy_delta = sum(jnp.sum(xlogy(t, t) - t) for t in (pn, rn, sn))
            entropy_delta -= sum(jnp.sum(xlogy(t, t) - t) for t in (p, r, s))
            gap = float(-linear_delta - self.epsilon * entropy_delta)
            if abs(gap) <= self.threshold and iteration + 1 >= self.min_iterations:
                converged, status = True, "converged"
                break
            step = 1.0
            for _ in range(40):
                candidate = tuple((1 - step) * old + step * new for old, new in zip((p, r, s), (pn, rn, sn)))
                value = float(objective(*candidate))
                if np.isfinite(value) and value <= history[-1] - 1e-4 * step * max(gap, 0):
                    p, r, s = candidate
                    history.append(value)
                    break
                step *= 0.5
            else:
                status = "line_search_failed"
                break
        source_error = float(jnp.max(jnp.abs(p.sum(1) + s - q)))
        target_error = float(jnp.max(jnp.abs(p.sum(0) + r.sum(0) - b)))
        if max(source_error, target_error) > self.inner_threshold:
            converged, status = False, "feasibility_not_converged"
        diagnostics = {
            "objective": np.asarray(history),
            "gap": gap,
            "status": status,
            "iterations": iteration + 1,
            "inner_iterations": np.asarray(inner_counts),
            "source_residual": source_error,
            "target_residual": target_error,
            "mass_balance_residual": float(b.sum() - q.sum() - r.sum() + s.sum()),
        }
        return OpenSystemOutput(
            p, r, s, diagnostics=diagnostics, cost=history[-1], converged=converged, is_linear=self.alpha == 0
        )

    @classmethod
    def _call_kwargs(cls) -> tuple[set[str], set[str]]:
        return {"xy", "x", "y", "a", "b", "recruitment", "recruitment_penalty", "sink_cost", "batch_size"}, set()
