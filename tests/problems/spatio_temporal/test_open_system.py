import dataclasses
from itertools import product

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from scipy.optimize import Bounds, LinearConstraint, brentq, linprog, minimize
from scipy.special import xlogy

from anndata import AnnData

from moscot.backends.ott._open_system import OpenSystemSolver, _linear_solve, _quadratic
from moscot.problems.spatiotemporal import OpenSystemConfig, SpatioTemporalProblem
from moscot.utils.tagged_array import Tag, TaggedArray


def solve(q=(0.3, 0.4), b=(0.5, 0.5), cr=None, cs=(0.4, 0.7), **kwargs):
    c = np.array([[0.1, 0.5], [0.3, 0.2]])
    cx = np.array([[0, 0.7], [0.7, 0]])
    cy = np.array([[0, 0.9], [0.9, 0]])
    cr = np.array([[0.2, 0.6]]) if cr is None else np.asarray(cr)
    solver = OpenSystemSolver(
        epsilon=kwargs.pop("epsilon", 0.2),
        threshold=1e-9,
        linear_solver_kwargs={"threshold": 1e-11, "max_iterations": 30000},
        **kwargs,
    )
    tag = lambda x: TaggedArray(np.asarray(x), tag=Tag.COST_MATRIX)
    out = solver(
        xy=tag(c),
        x=tag(cx),
        y=tag(cy),
        a=np.asarray(q),
        b=np.asarray(b),
        recruitment=tag(np.zeros_like(cr)),
        recruitment_penalty=cr,
        sink_cost=np.asarray(cs),
    )
    return out, (c, cx, cy, cr, np.asarray(cs))


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("q", [(0.3, 0.4), (0.8, 0.9), (0.0, 0.7)])
def test_constraints(q, jit):
    out, _ = solve(q=q, jit=jit)
    assert out.converged
    np.testing.assert_allclose(out.P.sum(1) + out.S, q, atol=2e-11)
    np.testing.assert_allclose(out.P.sum(0) + out.R.sum(0), [0.5, 0.5], atol=2e-11)
    assert abs(1 - sum(q) - out.total_recruitment + out.total_loss) < 3e-11
    for block in (out.P, out.R, out.S):
        assert np.all(np.isfinite(block))
        assert np.all(block >= 0)
    assert np.all(out.augmented_transport_matrix[2:, -1] == 0)
    assert np.all(np.diff(out.diagnostics["objective"]) <= 0)
    np.testing.assert_allclose(out.push(np.ones(2)), out.resident_contribution)
    np.testing.assert_allclose(out.pull(np.ones(2)), out.P.sum(1))
    np.testing.assert_allclose(out.recruitment_mass, out.R.sum(0))
    moved = out.to("cpu")
    np.testing.assert_array_equal(moved.R, out.R)
    np.testing.assert_array_equal(moved.S, out.S)


@pytest.mark.parametrize(("n", "m", "symmetric"), [(3, 2, False), (2, 4, False), (3, 2, True), (1, 3, True)])
def test_gradient(n, m, symmetric):
    rng = np.random.default_rng(42)
    # Asymmetric matrices also test both transpose terms in the general derivative.
    cx, cy, p = rng.random((n, n)), rng.random((m, m)), rng.random((n, m))
    if symmetric:
        cx, cy = (cx + cx.T) / 2, (cy + cy.T) / 2
    gx, gy = geometry.Geometry(cost_matrix=jnp.asarray(cx)), geometry.Geometry(cost_matrix=jnp.asarray(cy))
    tensor = (cx[:, :, None, None] - cy[None, None, :, :]) ** 2
    objective = lambda t: np.einsum("abij,ai,bj->", tensor, t, t)
    q, grad = _quadratic(jnp.asarray(p), gx, gy)
    np.testing.assert_allclose(q, objective(p), rtol=1e-12)
    for i, j in np.ndindex(p.shape):
        perturbation = np.zeros_like(p)
        perturbation[i, j] = 1e-5
        finite_diff = (objective(p + perturbation) - objective(p - perturbation)) / 2e-5
        np.testing.assert_allclose(grad[i, j], finite_diff, rtol=1e-8)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("q", [0.6, 1.0, 1.4])
def test_exact_scalar_optimum(alpha, q):
    # Eliminating R=1-P and S=q-P gives a strictly convex one-dimensional objective.
    epsilon, c, recruitment_expression, entry, sink, spatial_difference = 0.4, 0.4, 0.3, 0.1, 0.2, 0.7
    cr = (1 - alpha) * recruitment_expression + entry

    def derivative(p):
        return (
            (1 - alpha) * c
            + 2 * alpha * spatial_difference**2 * p
            - cr
            - sink
            + epsilon * (np.log(p) - np.log(1 - p) - np.log(q - p))
        )

    exact = brentq(derivative, 1e-14, min(1.0, q) - 1e-14, xtol=1e-14)
    tag = lambda value: TaggedArray(np.array([[value]]), tag=Tag.COST_MATRIX)
    out = OpenSystemSolver(alpha=alpha, epsilon=epsilon, threshold=1e-12, linear_solver_kwargs={"threshold": 1e-13})(
        x=tag(0.0),
        y=tag(spatial_difference),
        xy=tag(c),
        a=np.array([q]),
        b=np.ones(1),
        recruitment=tag(recruitment_expression),
        recruitment_penalty=np.array([[entry]]),
        sink_cost=np.array([sink]),
    )
    assert out.converged
    np.testing.assert_allclose([out.P.item(), out.R.item(), out.S.item()], [exact, 1 - exact, q - exact], atol=2e-6)


@pytest.mark.parametrize("seed", range(4))
def test_nonlinear_objective_and_kkt(seed):
    rng = np.random.default_rng(seed)
    n, m, k = 3, 4, 2
    cx, cy = rng.uniform(0, 0.3, (n, n)), rng.uniform(0, 0.3, (m, m))
    c, cr, cs = rng.random((n, m)), rng.random((k, m)), rng.random(n)
    q, b = rng.uniform(0.15, 0.5, n), rng.uniform(0.1, 0.4, m)
    cr[0, 1] = np.inf
    alpha, epsilon = 0.4, 0.4
    tag = lambda value: TaggedArray(value, tag=Tag.COST_MATRIX)
    out = OpenSystemSolver(alpha=alpha, epsilon=epsilon, threshold=1e-11, linear_solver_kwargs={"threshold": 1e-12})(
        x=tag(cx),
        y=tag(cy),
        xy=tag(c),
        a=q,
        b=b,
        recruitment=tag(np.zeros((k, m))),
        recruitment_penalty=cr,
        sink_cost=cs,
    )
    assert out.converged
    p, r, s = (np.asarray(block) for block in (out.P, out.R, out.S))
    loss = (cx[:, :, None, None] - cy[None, None, :, :]) ** 2
    q_value = np.einsum("abij,ai,bj->", loss, p, p)
    q_gradient = np.einsum("abij,bj->ai", loss, p) + np.einsum("baji,bj->ai", loss, p)
    expected = (
        (1 - alpha) * np.sum(c * p)
        + alpha * q_value
        + np.sum(np.where(np.isfinite(cr), cr, 0) * r)
        + np.sum(cs * s)
        + epsilon * sum(np.sum(xlogy(t, t) - t) for t in (p, r, s))
    )
    np.testing.assert_allclose(out.cost, expected, atol=1e-12)
    # KKT: each recruitment row shares the same target multiplier; loss gives the source multiplier.
    gp = (1 - alpha) * c + alpha * q_gradient + epsilon * np.log(p)
    gr = cr + epsilon * np.log(np.where(r > 0, r, 1))
    gs = cs + epsilon * np.log(s)
    np.testing.assert_allclose(gp, gs[:, None] + gr[1][None, :], atol=3e-6)
    np.testing.assert_allclose(gr[0, [0, 2, 3]], gr[1, [0, 2, 3]], atol=1e-12)
    assert r[0, 1] == 0


@pytest.mark.parametrize("shift", [0.0, 100.0, 10000.0])
def test_cost_shift_invariance(shift):
    # Adding a source/target potential to each incident edge changes only a constant.
    tag = lambda value: TaggedArray(np.array([[value]]), tag=Tag.COST_MATRIX)
    out = OpenSystemSolver(alpha=0, epsilon=1.0, linear_solver_kwargs={"threshold": 1e-11})(
        x=tag(0.0),
        y=tag(0.0),
        xy=tag(2 * shift),
        a=np.ones(1),
        b=np.ones(1),
        recruitment=tag(0.0),
        recruitment_penalty=np.array([[shift]]),
        sink_cost=np.array([shift]),
    )
    exact = (3 - np.sqrt(5)) / 2
    assert out.converged
    np.testing.assert_allclose([out.P.item(), out.R.item(), out.S.item()], [exact, 1 - exact, 1 - exact], atol=1e-10)
    expected_entropy = xlogy(exact, exact) - exact + 2 * (xlogy(1 - exact, 1 - exact) - (1 - exact))
    np.testing.assert_allclose(out.cost, 2 * shift + expected_entropy, atol=1e-7)


@pytest.mark.parametrize("source_scale", [0.5, 1.0, 1.5])
def test_feasibility_against_linear_program(source_scale):
    q, b = source_scale * np.array([0.4, 0.6]), np.array([0.5, 0.5])
    # Variables are P.ravel(), R.ravel(), S; these are the original constraints, without dummy marginals.
    constraints = np.array(
        [[1, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0]]
    )
    tag = lambda shape: TaggedArray(np.zeros(shape), tag=Tag.COST_MATRIX)
    for support in product([False, True], repeat=4):
        active = np.array([True] * 4 + list(support))
        bounds = [(0, None if allowed else 0) for allowed in active]
        reference = linprog(np.zeros(8), A_eq=constraints, b_eq=np.concatenate((q, b)), bounds=bounds)
        solver = OpenSystemSolver()
        arguments = {
            "x": tag((2, 2)),
            "y": tag((2, 2)),
            "xy": tag((2, 2)),
            "a": q,
            "b": b,
            "recruitment": tag((1, 2)),
            "recruitment_penalty": np.where(active[4:6], 0.0, np.inf)[None, :],
            "sink_cost": np.where(active[6:], 0.0, np.inf),
        }
        if reference.success:
            _, _, _, cr, cs, _, _, allowed = solver._prepare(**arguments)
            retained = np.concatenate((np.asarray(allowed).ravel(), np.isfinite(cr).ravel(), np.isfinite(cs)))
            for index in np.flatnonzero(active & ~retained):
                objective = np.zeros(8)
                objective[index] = -1
                maximum = linprog(objective, A_eq=constraints, b_eq=np.concatenate((q, b)), bounds=bounds)
                assert maximum.success
                assert -maximum.fun < 1e-12  # Removed entries really are structurally zero.
        else:
            with pytest.raises(ValueError, match="Infeasible"):
                solver._prepare(**arguments)


@pytest.mark.parametrize(
    ("cr", "cs", "q"),
    [
        (np.full((1, 2), np.inf), (0.4, 0.7), (0.8, 0.8)),
        (np.empty((0, 2)), (0.4, 0.7), (0.8, 0.8)),
        (np.array([[0.2, np.inf]]), (0.0, 0.0), (0.4, 0.4)),
        (np.array([[0.2, 0.4]]), (1e6, 1e6), (0.3, 0.4)),
    ],
)
def test_forbidden_and_sink_extremes(cr, cs, q):
    out, _ = solve(q=q, cr=cr, cs=cs, alpha=0)
    assert out.converged
    assert np.all(np.asarray(out.R)[np.isinf(cr)] == 0)
    np.testing.assert_allclose(out.P.sum(1) + out.S, q, atol=2e-11)


@pytest.mark.parametrize(
    ("q", "b", "cr", "cs"),
    [
        ((0.3, 0.4), (0.5, 0.5), np.empty((0, 2)), (1, 1)),
        ((0.8, 0.8), (0.5, 0.5), np.zeros((1, 2)), (np.inf, np.inf)),
        ((0.1, 0.1), (0.5, 0.5), np.array([[0, np.inf]]), (1, 1)),
    ],
)
def test_infeasible(q, b, cr, cs):
    with pytest.raises(ValueError, match="Infeasible"):
        solve(q=q, b=b, cr=cr, cs=cs)


@pytest.mark.parametrize(
    ("q", "b", "cr", "cs"),
    [
        ((0, 0), (0.5, 0.5), np.zeros((1, 2)), (1, 1)),
        ((0.4, 0.6), (0, 0), np.zeros((1, 2)), (1, 1)),
        ((0, 0), (0, 0), np.zeros((1, 2)), (1, 1)),
        ((0.4, 0.6), (0.5, 0.5), np.full((1, 2), np.inf), (np.inf, np.inf)),
        ((0.5, 0.7), (0.5, 0), np.zeros((1, 2)), (np.inf, 1)),
        ((0.2, 0.3), (0.5, 0.5), np.array([[np.inf, 0]]), (1, 1)),
    ],
)
def test_boundary_and_zero_mass(q, b, cr, cs):
    out, _ = solve(q=q, b=b, cr=cr, cs=cs, alpha=0)
    assert out.converged
    np.testing.assert_allclose(out.P.sum(1) + out.S, q, atol=2e-11)
    np.testing.assert_allclose(out.P.sum(0) + out.R.sum(0), b, atol=2e-11)


def test_small_epsilon():
    args = tuple(
        jnp.asarray(x)
        for x in (
            [[0.1, 0.9], [0.8, 0.2]],
            [[0.3, np.inf]],
            [0.4, 0.4],
            [0.7, 0.7],
            [0.5, 0.5],
        )
    )
    p, r, s, error, _ = jax.jit(_linear_solve)(*args, 1e-3, 1e-9, 100000)
    assert float(error) < 1e-9
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(s))
    assert r[0, 1] == 0


def test_batched_geometry_and_float32():
    rng = np.random.default_rng(11)
    x, y, z = (rng.random((n, 2)).astype(np.float32) for n in (3, 4, 2))
    arguments = {
        "x": TaggedArray(x),
        "y": TaggedArray(y),
        "xy": TaggedArray(x, y),
        "recruitment": TaggedArray(z, y),
        "recruitment_penalty": np.zeros((2, 4)),
        "sink_cost": np.ones(3),
        "a": np.array([0.2, 0.3, 0.4]),
        "b": np.full(4, 0.25),
    }
    with jax.experimental.enable_x64(False):
        dense = OpenSystemSolver(epsilon=0.3, threshold=1e-6)(**arguments)
        batched = OpenSystemSolver(epsilon=0.3, threshold=1e-6)(**arguments, batch_size=2)
    assert dense.converged
    assert batched.converged
    assert batched.P.dtype == jnp.float32
    assert batched.diagnostics["source_residual"] <= 1e-7
    assert batched.diagnostics["target_residual"] <= 1e-7
    np.testing.assert_allclose(dense.P, batched.P, atol=1e-6)
    np.testing.assert_array_equal(batched.P_hat[-2:, -1], [0, 0])


def test_failed_convergence_is_reported():
    out, _ = solve(max_iterations=1)
    assert not out.converged
    assert out.diagnostics["status"] == "max_iterations"
    assert out.diagnostics["source_residual"] < 1e-11


@pytest.mark.parametrize("arguments", [{"epsilon": 0}, {"alpha": 1.1}, {"max_iterations": 0}, {"threshold": -1}])
def test_invalid_solver_settings(arguments):
    with pytest.raises(ValueError, match="epsilon|alpha|counts|threshold"):
        OpenSystemSolver(**arguments)


def test_invalid_mass_and_inner_failure():
    with pytest.raises(ValueError, match="nonnegative"):
        solve(q=(-0.1, 0.5))
    solver = OpenSystemSolver(epsilon=0.01, linear_solver_kwargs={"max_iterations": 1})
    with pytest.raises(RuntimeError, match="feasibility"):
        solver(
            x=TaggedArray(np.array([[0.0], [1.0]])),
            y=TaggedArray(np.array([[0.0], [1.0]])),
            xy=TaggedArray(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]])),
            a=np.array([0.7, 0.7]),
            b=np.array([0.5, 0.5]),
            recruitment=TaggedArray(np.array([[0.0]]), np.array([[0.0], [1.0]])),
            recruitment_penalty=np.zeros((1, 2)),
            sink_cost=np.ones(2),
        )


def test_empty_resident_subset():
    tag = lambda shape: TaggedArray(np.zeros(shape), tag=Tag.COST_MATRIX)
    with pytest.raises(ValueError, match="Empty resident"):
        OpenSystemSolver()(
            x=tag((0, 0)),
            y=tag((2, 2)),
            xy=tag((0, 2)),
            a=np.empty(0),
            b=np.full(2, 0.5),
            recruitment=tag((1, 2)),
            recruitment_penalty=np.zeros((1, 2)),
            sink_cost=np.empty(0),
        )


@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_trusted_optimizer(alpha):
    out, (c, cx, cy, cr, cs) = solve(alpha=alpha)
    tensor = (cx[:, :, None, None] - cy[None, None, :, :]) ** 2

    def objective(t):
        p, r, s = t[:4].reshape(2, 2), t[4:6].reshape(1, 2), t[6:]
        return (
            (1 - alpha) * np.sum(c * p)
            + alpha * np.einsum("abij,ai,bj->", tensor, p, p)
            + np.sum(cr * r)
            + np.sum(cs * s)
            + 0.2 * np.sum(xlogy(t, t) - t)
        )

    constraints = np.array(
        [
            [1, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0],
        ]
    )
    rhs = [0.3, 0.4, 0.5, 0.5]
    result = minimize(
        objective,
        np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]),
        method="SLSQP",
        bounds=Bounds(1e-14, np.inf),
        constraints=LinearConstraint(constraints, rhs, rhs),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert result.success, result.message
    assert out.converged
    np.testing.assert_allclose(out.cost, result.fun, atol=1e-8)
    np.testing.assert_allclose(np.concatenate((out.P.ravel(), out.R.ravel(), out.S)), result.x, atol=5e-5)


def test_balanced_fgw_limit():
    alpha, epsilon = 0.4, 0.2
    out, (c, cx, cy, _, _) = solve(q=(0.5, 0.5), cr=np.empty((0, 2)), cs=(np.inf, np.inf), alpha=alpha, epsilon=epsilon)
    geom = lambda c: geometry.Geometry(cost_matrix=jnp.asarray(c))
    problem = quadratic_problem.QuadraticProblem(
        geom(cx),
        geom(cy),
        geom(c),
        fused_penalty=(1 - alpha) / (2 * alpha),
        a=jnp.array([0.5, 0.5]),
        b=jnp.array([0.5, 0.5]),
    )
    # OTT's squared-loss linearization is grad(Q)/2; match both linear and entropy coefficients.
    reference = gromov_wasserstein.GromovWasserstein(
        epsilon=epsilon / (2 * alpha),
        threshold=1e-10,
        linear_solver=sinkhorn.Sinkhorn(threshold=1e-11),
        max_iterations=200,
    )(problem)
    np.testing.assert_allclose(out.P, reference.matrix, atol=2e-5)
    assert np.all(out.S == 0)


@pytest.fixture
def residents():
    adata = AnnData(np.zeros((6, 2)))
    adata.obs["time"] = [0, 0, 1, 1, 2, 2]
    adata.obs["lineage"] = ["immune", "malignant"] * 3
    adata.obs["beta"] = [np.log(2), 0] * 3
    adata.obs["death"] = 100
    adata.obsm["state"] = np.tile([[0.0, 0.0], [0.3, 0.1]], (3, 1))
    adata.obsm["spatial"] = np.tile([[0.0, 0.0], [3.0, 4.0]], (3, 1))
    return adata


def configuration():
    reference = AnnData(np.zeros((2, 2)))
    reference.obsm["state"] = np.array([[0.0, 0.0], [0.3, 0.1]])
    reference.obs["lineage"] = ["immune", "malignant"]
    reference.obs["entry"] = [0.2, 0.4]
    return OpenSystemConfig(
        {0: 100, 1: 200, 2: 400},
        circulating=reference,
        entry_cost="entry",
        proliferation_rates="beta",
        vascular_coordinates={1: [[0.0, 0.0]], 2: [[0.0, 0.0]]},
        vascular_weight=0.1,
    )


def prepare(residents, config=None, **kwargs):
    return SpatioTemporalProblem(residents).prepare(
        "time",
        joint_attr="state",
        normalize_spatial=False,
        open_system=configuration() if config is None else config,
        **kwargs,
    )


def test_integration_and_serialization(residents, tmp_path):
    model = SpatioTemporalProblem(residents)
    model.apoptosis_key = "death"
    model.prepare("time", joint_attr="state", normalize_spatial=False, open_system=configuration())
    assert set(model.problems) == {(0, 1), (1, 2)}
    for problem in model.problems.values():
        np.testing.assert_allclose(problem.a, [0.5, 0.25])
        np.testing.assert_allclose(problem.b, [0.5, 0.5])
        np.testing.assert_allclose(problem.prior_growth_rates, [2, 1])
        np.testing.assert_allclose(problem.vascular_distances, [0, 5])
        np.testing.assert_allclose(np.diag(problem.recruitment_penalty), [0.2, 0.9])
        assert np.isinf(problem.recruitment_penalty[0, 1])
    model.solve(alpha=0, epsilon=0.2, batch_size=1, threshold=1e-8, device="cpu")
    for problem in model.problems.values():
        out = problem.solution
        assert out.converged
        assert out.R[0, 1] == out.R[1, 0] == 0
        np.testing.assert_allclose(problem.push(normalize=False).ravel(), out.P.sum(0))
        np.testing.assert_allclose(problem.pull(normalize=False).ravel(), out.P.sum(1))
    path = tmp_path / "open.pkl"
    model.save(path)
    loaded = SpatioTemporalProblem.load(path)
    assert loaded.stage == "solved"
    for key in model:
        np.testing.assert_array_equal(model[key].solution.R, loaded[key].solution.R)
        np.testing.assert_array_equal(model[key].solution.S, loaded[key].solution.S)
        np.testing.assert_array_equal(model[key].circulating_names, loaded[key].circulating_names)
        assert loaded[key].open_system.cell_counts == configuration().cell_counts
    loaded.solve(alpha=0, epsilon=0.2)


def test_explicit_growth_and_array_reference(residents):
    config = dataclasses.replace(
        configuration(),
        proliferation_rates=None,
        growth_weights={0: [4.0, 1.0]},
        circulating=np.array([[0.0, 0.0]]),
        circulating_lineages=["immune"],
        entry_cost=0.2,
        compatibility=[("immune", "malignant")],
    )
    model = prepare(residents, config, policy="explicit", subset=[(0, 2)])
    assert list(model.problems) == [(0, 2)]
    np.testing.assert_allclose(model[0, 2].a, [0.5, 0.125])
    rates_model = prepare(residents, policy="explicit", subset=[(0, 2)])
    np.testing.assert_allclose(model[0, 2].a, rates_model[0, 2].a)
    model.solve(alpha=0, epsilon=0.2)
    assert model[0, 2].solution.R[0, 0] == 0


def test_nonuniform_baseline_weights(residents):
    residents.obs["weight"] = [1.0, 3.0] * 3
    model = prepare(residents, a="weight", b="weight")
    np.testing.assert_allclose(model[0, 1].a, [0.25, 0.375])
    np.testing.assert_allclose(model[0, 1].b, [0.25, 0.75])


@pytest.mark.parametrize("scale", [1e-308, 1.0, 1e308])
def test_weight_normalization_is_scale_invariant(residents, scale):
    residents.obs["weight"] = np.full(residents.n_obs, scale)
    model = prepare(residents, a="weight", b="weight")
    np.testing.assert_allclose(model[0, 1].a, [0.5, 0.25])
    np.testing.assert_allclose(model[0, 1].b, [0.5, 0.5])


def test_counts_rescale_masses_not_costs(residents):
    config = configuration()
    scaled = dataclasses.replace(config, cell_counts={t: 1000 * n for t, n in config.cell_counts.items()})
    original = prepare(residents, config, policy="explicit", subset=[(0, 1)])
    rescaled = prepare(residents, scaled, policy="explicit", subset=[(0, 1)])
    for model in (original, rescaled):
        model.solve(alpha=0.01, epsilon=0.3, threshold=1e-8)
        assert model[0, 1].solution.converged
    np.testing.assert_allclose(original[0, 1].a, rescaled[0, 1].a)
    np.testing.assert_allclose(original[0, 1].solution.P_hat, rescaled[0, 1].solution.P_hat)
    np.testing.assert_allclose(original[0, 1].solution.cost, rescaled[0, 1].solution.cost)


@pytest.mark.parametrize("options", [{"tau_a": 0.9}, {"tau_b": 0.9}, {"rank": 2}, {"scale_cost": "mean"}])
def test_reject_incompatible_solver(residents, options):
    with pytest.raises(ValueError, match="Open-system|Low-rank"):
        prepare(residents).solve(**options)


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_counts": {0: 0, 1: 2, 2: 3}},
        {"vascular_coordinates": None},
        {"growth_weights": 2},
        {"sink_cost": -1},
        {"entry_cost": -1},
        {"vascular_weight": -1},
        {"circulating": np.ones((1, 3))},
        {"vascular_coordinates": {1: [], 2: []}},
    ],
)
def test_invalid_inputs(residents, changes):
    with pytest.raises(ValueError, match="counts|coordinates|either|nonnegative|dimension"):
        prepare(residents, dataclasses.replace(configuration(), **changes))


def test_missing_embedding_and_zero_baseline(residents):
    with pytest.raises(ValueError, match="shared joint_attr"):
        SpatioTemporalProblem(residents).prepare("time", open_system=configuration())
    residents.obs["weights"] = 0
    with pytest.raises(ValueError, match="positive total"):
        prepare(residents, a="weights")
    with pytest.raises(ValueError, match="increasing"):
        prepare(residents, policy="explicit", subset=[(1, 0)])


def test_no_reference_or_vessels_and_reprepare(residents):
    config = OpenSystemConfig({0: 200, 1: 100, 2: 100})
    model = prepare(residents, config)
    model.solve(alpha=0, epsilon=0.2)
    assert model[0, 1].solution.R.shape == (0, 2)
    assert model[0, 1].solution.total_loss >= 1 - 1e-6
    model.prepare("time", joint_attr="state", normalize_spatial=False)
    model.solve(epsilon=0.2)
    assert not hasattr(model[0, 1].solution, "R")


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="GPU unavailable")
def test_gpu():
    with jax.default_device(jax.devices("gpu")[0]):
        out, _ = solve()
    assert out.P.devices().pop().platform == "gpu"
    assert out.converged
