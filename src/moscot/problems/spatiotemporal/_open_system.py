"""Preparation of open-system timepoint pairs."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from anndata import AnnData

from moscot._types import ArrayLike, Device_t, Numeric_t
from moscot.backends.ott._open_system import OpenSystemSolver
from moscot.base.problems._utils import wrap_solve
from moscot.base.problems.birth_death import BirthDeathProblem
from moscot.utils.tagged_array import TaggedArray

ResidentValue = Union[str, float, Mapping[Numeric_t, ArrayLike]]


@dataclass
class OpenSystemConfig:
    """Inputs for explicit recruitment and loss in spatiotemporal transport.

    Parameters
    ----------
    cell_counts
        Positive segmented tissue counts A_t indexed by timepoint (not sampled AnnData sizes).
    circulating
        Circulating reference AnnData or an array of shared state embeddings. None disables recruitment.
    circulating_attr
        Reference obsm key. Defaults to the resident ``joint_attr`` key. Both must use the same feature basis.
    lineage_key
        Resident obs key with broad lineage labels. Also used for reference AnnData unless
        ``circulating_lineages`` is supplied.
    circulating_lineages
        Reference obs key or labels in circulating row order; required for array input.
    compatibility
        Allowed (circulating lineage, target lineage) pairs. Defaults to immune-to-immune and
        malignant-to-malignant, with exact lowercase labels. An empty sequence forbids all recruitment.
    proliferation_rates
        Nonnegative log proliferation rates beta, supplied as an obs key, scalar, or timepoint-to-vector mapping.
        Growth is exp(beta * delta_t); apoptosis is never subtracted.
    growth_weights
        Alternative nonnegative, precomputed *interval* multipliers (not exponentiated again).
        Same formats as proliferation_rates; mutually exclusive with it.
    sink_cost
        Nonnegative loss costs, in the same formats. +infinity prohibits loss for a source cell.
    entry_cost
        Nonnegative circulating entry penalties: scalar, vector, or reference obs key.
        +infinity prohibits entry for that reference cell.
    vascular_coordinates
        Target timepoint-to-coordinate-array mapping, in the original target spatial coordinate frame and units.
    vascular_weight
        Nonnegative coefficient multiplying Euclidean distance to the nearest vascular entry.
        Positive values require coordinates for every selected target timepoint.
    """

    cell_counts: Mapping[Numeric_t, float]
    circulating: Optional[Union[AnnData, ArrayLike]] = None
    circulating_attr: Optional[str] = None
    lineage_key: str = "lineage"
    circulating_lineages: Optional[Union[str, Sequence[str]]] = None
    compatibility: Optional[Sequence[tuple[str, str]]] = None
    proliferation_rates: Optional[ResidentValue] = None
    growth_weights: Optional[ResidentValue] = None
    sink_cost: ResidentValue = 1.0
    entry_cost: Union[str, float, ArrayLike] = 0.0
    vascular_coordinates: Optional[Mapping[Numeric_t, ArrayLike]] = None
    vascular_weight: float = 0.0


def _vector(value: Any, adata: AnnData, time: Numeric_t, name: str, *, allow_inf: bool = False) -> np.ndarray[Any, Any]:
    if isinstance(value, Mapping):
        value = value[time]
    if isinstance(value, str):
        value = adata.obs[value].to_numpy()
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        value = np.full(adata.n_obs, value)
    if value.shape != (adata.n_obs,) or np.any(value < 0) or np.any(np.isnan(value)):
        raise ValueError(f"`{name}` must be a nonnegative vector of length {adata.n_obs}.")
    if not allow_inf and not np.all(np.isfinite(value)):
        raise ValueError(f"`{name}` must be finite.")
    return value


class OpenSystemProblem(BirthDeathProblem):
    """A resident timepoint pair with explicitly prepared recruitment and loss metadata."""

    def configure(self, config: OpenSystemConfig, *, joint_attr: str, spatial_key: str) -> None:
        """Attach shared reference geometry and unnormalized growth-adjusted source masses."""
        assert self._src_key is not None
        assert self._tgt_key is not None
        assert self.xy is not None
        dt = self.delta
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Open-system pairs require numeric increasing timepoints (delta_t > 0).")
        if config.proliferation_rates is not None and config.growth_weights is not None:
            raise ValueError("Specify either proliferation_rates or growth_weights, not both.")
        counts = np.asarray([config.cell_counts[self._src_key], config.cell_counts[self._tgt_key]], dtype=float)
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
            raise ValueError("Segmented cell counts must be finite and positive.")
        if not np.isfinite(config.vascular_weight) or config.vascular_weight < 0:
            raise ValueError("`vascular_weight` must be finite and nonnegative.")
        a, b = np.asarray(self.a), np.asarray(self.b)
        for weights in (a, b):
            if not len(weights) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("Resident weights must be finite, nonnegative, and have positive total mass.")
        # Normalize ratios without overflowing the sum of finite baseline weights.
        a, b = a / a.max(), b / b.max()
        if config.growth_weights is not None:
            growth = _vector(config.growth_weights, self.adata_src, self._src_key, "growth_weights")
        else:
            beta = _vector(
                0.0 if config.proliferation_rates is None else config.proliferation_rates,
                self.adata_src,
                self._src_key,
                "proliferation_rates",
            )
            with np.errstate(over="ignore"):
                growth = np.exp(beta * dt)
        q = counts[0] / counts[1] * (a / a.sum()) * growth
        self._a = q
        self._b = b / b.sum()
        if not np.all(np.isfinite(q)):
            raise ValueError("Growth-adjusted source mass overflowed.")
        self._prior_growth = growth
        self.sink_cost = _vector(config.sink_cost, self.adata_src, self._src_key, "sink_cost", allow_inf=True)
        reference = config.circulating
        reference_attr = config.circulating_attr or joint_attr
        target: np.ndarray[Any, Any] = np.asarray(self.adata_tgt.obsm[joint_attr])
        source: np.ndarray[Any, Any] = np.asarray(self.adata_src.obsm[joint_attr])
        if reference is None:
            z = np.empty((0, target.shape[1]))
        else:
            z = np.asarray(reference.obsm[reference_attr] if isinstance(reference, AnnData) else reference)
        for embedding in (source, target, z):
            if embedding.ndim != 2 or embedding.shape[1] != target.shape[1] or not np.all(np.isfinite(embedding)):
                raise ValueError(
                    "Resident and circulating embeddings must be finite and share a feature dimension/basis."
                )
        self.circulating_names = (
            np.asarray(reference.obs_names) if isinstance(reference, AnnData) else np.arange(len(z))
        )
        entry = config.entry_cost
        if isinstance(entry, str):
            if not isinstance(reference, AnnData):
                raise ValueError("An entry_cost obs key requires circulating AnnData.")
            entry = reference.obs[entry].to_numpy()
        entry = np.asarray(entry, dtype=float)
        if entry.ndim == 0:
            entry = np.full(len(z), entry)
        if entry.shape != (len(z),) or np.any(np.isnan(entry)) or np.any(entry < 0):
            raise ValueError("`entry_cost` must be nonnegative and match circulating rows.")
        allowed = np.zeros((len(z), len(target)), dtype=bool)
        if len(z):
            labels = config.circulating_lineages
            if isinstance(reference, AnnData) and (labels is None or isinstance(labels, str)):
                labels = reference.obs[config.lineage_key if labels is None else labels].to_numpy()
            if labels is None or isinstance(labels, str) or np.asarray(labels).shape != (len(z),):
                raise ValueError("Provide one lineage label per circulating cell.")
            target_labels = self.adata_tgt.obs[config.lineage_key].to_numpy()
            pairs = config.compatibility
            if pairs is None:
                pairs = (("immune", "immune"), ("malignant", "malignant"))
            for src, tgt in pairs:
                allowed |= (np.asarray(labels)[:, None] == src) & (target_labels[None, :] == tgt)
        distances = np.zeros(len(target))
        if config.vascular_coordinates is not None and self._tgt_key in config.vascular_coordinates:
            vessels = np.asarray(config.vascular_coordinates[self._tgt_key], dtype=float)
            coords: np.ndarray[Any, Any] = np.asarray(self.adata_tgt.obsm[spatial_key])
            if vessels.ndim != 2 or not len(vessels) or vessels.shape[1] != coords.shape[1]:
                raise ValueError("Vascular coordinates must be nonempty and match target spatial dimensions.")
            if not np.all(np.isfinite(vessels)) or not np.all(np.isfinite(coords)):
                raise ValueError("Spatial and vascular coordinates must be finite.")
            distances = cKDTree(vessels).query(coords)[0]
        elif config.vascular_weight > 0:
            raise ValueError(f"Missing vascular coordinates for target {self._tgt_key!r}.")
        self.vascular_distances = distances
        self.recruitment_penalty = np.where(
            allowed, entry[:, None] + config.vascular_weight * distances[None, :], np.inf
        )
        self.recruitment_geometry = TaggedArray(z, target, cost=self.xy.cost)
        self.open_system = config

    @wrap_solve
    def solve(
        self,
        *,
        alpha: float = 0.5,
        epsilon: float = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: Any = 1.0,
        batch_size: Optional[int] = None,
        initializer: Any = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        jit: bool = True,
        min_iterations: int = 1,
        max_iterations: int = 200,
        threshold: float = 1e-6,
        linear_solver_kwargs: Optional[Mapping[str, Any]] = None,
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "OpenSystemProblem":
        """Solve the explicit mass-balance problem using the OTT geometry/JAX backend."""
        if tau_a != 1 or tau_b != 1:
            raise ValueError("Open-system constraints require tau_a=tau_b=1; KL relaxation is not supported.")
        if rank != -1:
            raise ValueError("Low-rank open-system FGW is not supported.")
        if scale_cost != 1:
            raise ValueError("Open-system costs require scale_cost=1; preprocess coordinates/embeddings explicitly.")
        if initializer is not None or initializer_kwargs or kwargs:
            raise TypeError("Unsupported open-system initializer or solver arguments.")
        solver = OpenSystemSolver(
            alpha=alpha,
            epsilon=epsilon,
            jit=jit,
            threshold=threshold,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            linear_solver_kwargs=dict(linear_solver_kwargs or {}),
        )
        self._solver = solver
        self._solution = solver(
            xy=self.xy,
            x=self.x,
            y=self.y,
            a=self.a,
            b=self.b,
            batch_size=batch_size,
            recruitment=self.recruitment_geometry,
            recruitment_penalty=self.recruitment_penalty,
            sink_cost=self.sink_cost,
            device=device,
        )
        return self

    @property
    def posterior_growth_rates(self) -> None:
        """Not identifiable from persistence alone when recruitment and loss are explicit."""
        return None
