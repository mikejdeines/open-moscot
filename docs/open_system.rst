Open-system spatiotemporal transport
====================================

``SpatioTemporalProblem.prepare(open_system=OpenSystemConfig(...))`` adds explicit
circulating recruitment and resident death/egress to spatiotemporal FGW. Omitting
``open_system`` preserves the original preparation, birth/death estimation, solver,
and cost-scaling defaults. No additional dependency is required.

Preparing MERFISH and a circulating reference
---------------------------------------------

The resident AnnData contains all sampled timepoints. Its ``obs["time"]`` must be
numeric, ``obsm["spatial"]`` contains spatial coordinates, and ``obsm["state"]``
contains an expression/latent embedding in **the same basis** as the circulating
reference's ``obsm["state"]``. Equal dimensionality alone does not establish this:
use shared genes with identical preprocessing, or project both datasets through
one fitted embedding. The open-system API requires an explicit embedding key and
rejects local PCA/expression callbacks.

For example, with preprocessed MERFISH and reference files:

.. code-block:: python

    import numpy as np
    import scanpy as sc
    from moscot.problems.spatiotemporal import OpenSystemConfig, SpatioTemporalProblem

    resident = sc.read_h5ad("merfish_timepoints.h5ad")
    circulating = sc.read_h5ad("circulating_reference.h5ad")
    # obs['beta'] contains calibrated log proliferation rates per time unit.
    # obs['lineage'] contains broad 'immune', 'malignant', or other lineage labels.
    config = OpenSystemConfig(
        circulating=circulating,
        cell_counts={0: 10000, 2: 14000},  # segmented tissue counts, not sample sizes
        proliferation_rates="beta",
        sink_cost=0.8,  # alternatively, an obs key or {time: per-cell costs}
        entry_cost=0.3,  # alternatively, a reference obs key or per-reference-cell array
        vascular_coordinates={2: np.array([[20.0, 30.0], [70.0, 80.0]])},
        vascular_weight=0.01,
    )
    model = SpatioTemporalProblem(resident).prepare(
        "time",
        joint_attr="state",
        normalize_spatial=False,
        policy="explicit",
        subset=[(0, 2)],
        open_system=config,
    )
    model.solve(alpha=0.2, epsilon=0.1, threshold=1e-7, batch_size=256)
    result = model[0, 2].solution
    P, R, S = result.P, result.R, result.S
    print(result.converged, result.diagnostics)
    resident_target_mass = result.resident_contribution
    recruited_target_mass = result.recruitment_mass
    lost_source_mass = result.loss
    # Convert normalized masses to estimated cell counts using A2:
    recruited_cell_count = 14000 * result.total_recruitment
    model.save("open_system.pkl")
    restored = SpatioTemporalProblem.load("open_system.pkl")

The default policy solves all adjacent pairs. Explicit pairs must have positive
time differences. A common circulating reference is reused for each pair; its
rows have no supply limit. For pair-specific references, prepare separate models.

``a`` and ``b`` may be obs keys defining nonnegative baseline weights; each is
normalized independently. Uniform weights are the default. Zero entries are
allowed; empty resident subsets and zero-total baseline weights are rejected.
``growth_weights`` can replace ``proliferation_rates`` with precomputed interval
multipliers; they are not exponentiated again. Resident vector inputs use obs keys,
scalars, or timepoint-to-vector mappings in subset observation order. Zero growth
weights are supported. Do not pass net birth-minus-death estimates: loss is a
separate optimized variable. Scored proliferation/apoptosis keys are not used
automatically in this mode; supply calibrated proliferation explicitly.

The default compatibility permits only exact lowercase ``immune`` to ``immune``
and ``malignant`` to ``malignant`` transitions. Other labels, including immune
subtypes, are forbidden unless listed in ``compatibility=[(source_label,
target_label), ...]``. An empty list forbids all recruitment. The resident
``lineage_key`` defaults to ``"lineage"``. Reference labels can use that key, another
obs key via ``circulating_lineages``, or an explicit sequence. Array references
require explicit labels. ``circulating_attr`` can select a different reference
obsm key, provided it represents the same feature basis.

Vascular penalties use Euclidean nearest-entry distances in the **original**
target coordinate frame, even when ``normalize_spatial=True`` standardizes the
resident spatial geometries. Coordinates and the distance coefficient must have
consistent physical units. No vascular coordinates are needed when the coefficient
is zero; a positive coefficient requires a nonempty entry array for every target.
Positive infinity in a sink/entry cost prohibits that transition. ``circulating=None``
disables recruitment. Both loss and recruitment can be prohibited to recover a
balanced resident problem when the growth-adjusted totals match.

Objective, normalization, and algorithm
---------------------------------------

For source/target segmented counts :math:`A_1,A_2`, normalized resident baseline
weights :math:`a_0,b_0`, and interval :math:`\Delta t>0`, the solver uses

.. math::

    \widetilde q_i = (A_1/A_2) a_{0i}\exp(\beta_i\Delta t),\qquad
    \widetilde\nu_j=b_{0j}.

It jointly optimizes nonnegative resident transport :math:`P\in\mathbb R^{N\times M}`,
recruitment :math:`R\in\mathbb R^{K\times M}`, and loss :math:`S\in\mathbb R^N`:

.. math::

    P\mathbf1+S=\widetilde q,\qquad
    P^T\mathbf1+R^T\mathbf1=\widetilde\nu.

The objective is exactly

.. math::

    (1-\alpha)\langle C,P\rangle + \alpha Q(P)
    + \langle C_R,R\rangle + \langle c_S,S\rangle
    + \epsilon\sum_{T\in\{P,R,S\}}\sum_{ij} T_{ij}(\log T_{ij}-1),

where :math:`Q(P)=\sum_{ii'jj'}(C_{x,ii'}-C_{y,jj'})^2P_{ij}P_{i'j'}` and
:math:`C_{R,kj}=(1-\alpha)c(z_k,y_j)+\lambda_v d(s_j,V)+e_k+m_{kj}`.
The lineage penalty :math:`m` is zero or positive infinity. The continuous entropy
convention is :math:`0\log0=0`. The augmented matrix has a structural zero block
from circulating cells to the sink. No circulating or sink spatial distances exist.

Only masses are divided by :math:`A_2`; costs are not. This is an objective defined
on normalized masses: applying the same coefficients directly to cell-count
couplings would change the relative quadratic weight, because
:math:`Q(A_2P)=A_2^2Q(P)`. Interpret and tune ``alpha`` and ``epsilon`` in the normalized
units. Open mode defaults to ``scale_cost=1`` and rejects other scale settings;
preprocess features and spatial coordinates explicitly if rescaling is desired.
``alpha`` may range from zero (linear open transport) to one (resident GW plus
entry/loss costs). Epsilon must be a finite positive scalar.

For symmetric spatial costs and current resident marginals :math:`r=P\mathbf1`,
:math:`c=P^T\mathbf1`, the full derivative is

.. math::

    \nabla Q(P)=2\left[C_x^{\circ2}r\mathbf1^T
                    +\mathbf1c^T(C_y^{\circ2})^T-2C_xPC_y^T\right].

The implementation also handles asymmetric costs using the corresponding transpose
terms. It recomputes resident marginals at each iteration. With
:math:`D=(1-\alpha)C+\alpha\nabla Q(P)`, generalized Sinkhorn scaling solves the
joint entropic linear subproblem. Updates are performed with log-sum-exp:

.. math::

    u=\widetilde q/(K_Pv+K_S),\qquad
    v=\widetilde\nu/(K_P^Tu+K_R^T\mathbf1),

followed by :math:`P=\mathrm{diag}(u)K_P\mathrm{diag}(v)`,
:math:`R=K_R\mathrm{diag}(v)`, and :math:`S=uK_S`.
The outer generalized conditional-gradient method backtracks along the joint
feasible direction and monitors the actual nonlinear objective. ``threshold``
controls the absolute generalized gap; ``max_iterations`` and ``min_iterations``
control outer iterations. ``linear_solver_kwargs`` accepts only ``threshold``
(absolute maximum marginal residual, default ``1e-7``) and ``max_iterations``
(default ``10000``). Use JAX 64-bit mode for tolerances below float32 precision.

For fully connected finite resident costs, feasibility requires that source mass
which cannot enter the sink fit into the total target mass, and that target mass
which cannot be recruited fit into total resident source mass. The solver checks
these inequalities and removes forced zeros on saturated boundary faces. Constraints
are enforced to the requested numerical tolerance, without dummy prescribed masses.
Failure to obtain an initially feasible coupling raises an error. Subsequent inner
or outer failures return the last feasible iterate with ``converged=False`` and an
explanatory diagnostic status. Very small epsilon or extreme costs may require more
inner iterations and float64; log-domain arithmetic does not guarantee convergence.

OTT's standard squared-loss GW linearization is half of this full :math:`\nabla Q`.
For a balanced-limit comparison against OTT, use
``fused_penalty=(1-alpha)/(2*alpha)`` and ``epsilon_ott=epsilon/(2*alpha)``
for positive alpha. Merely copying alpha/epsilon between those conventions does
not compare the same stationary equations. The original moscot solver is unchanged.

Mathematical checks
-------------------

The tests in ``tests/problems/spatio_temporal/test_open_system.py`` evaluate the
four-index definition of :math:`Q` independently of the geometry-based implementation.
Finite differences check its full gradient for symmetric and asymmetric costs and
unequal matrix dimensions, with no fixed resident marginals.

At a positive allowed entry, the first-order optimality conditions are

.. math::

    (1-\alpha)C_{ij}+\alpha\nabla Q(P)_{ij}+\epsilon\log P_{ij}&=f_i+g_j,\\
    C_{R,kj}+\epsilon\log R_{kj}&=g_j,\\
    c_{S,i}+\epsilon\log S_i&=f_i.

Substituting :math:`u_i=\exp(f_i/\epsilon)` and
:math:`v_j=\exp(g_j/\epsilon)` gives precisely the three scaling formulas above.
The tests check these conditions on rectangular problems with forbidden recruitment,
reconstruct the full nonlinear objective independently, and compare small problems
with SciPy's constrained optimizer.

A separate one-cell test eliminates :math:`R=b-P` and :math:`S=q-P`, then solves
the exact scalar derivative with a bracketed root finder. With zero costs,
:math:`q=b=1`, and one circulating cell, the solution is
:math:`P=(3-\sqrt5)/2` and :math:`R=S=1-P`. This also checks the entropy's
:math:`-T` term: it cannot be dropped because augmented total mass is variable.
Adding source/target potentials to all incident costs must leave this coupling
unchanged; the tests check that invariance even at large cost offsets.

Feasibility and boundary-face structural zeros are checked against independent linear
programs over all recruitment/loss support patterns of a two-by-two problem.
Normalization tests cover unequal counts, proliferation over non-unit time intervals,
and invariance under common count or baseline-weight rescaling. Baseline weights
are scaled by their maximum before summation to avoid overflow.

These checks establish consistency with the specified objective and constraints on
the tested cases. They do not prove that the nonconvex solver finds a global minimum.
The generalized gap is an objective-scale stopping measure, not an entrywise coupling
error bound or a bound of the same size on first-order residuals; inner solves are
also approximate to their specified feasibility tolerance.

Outputs and limitations
-----------------------

``result.transport_matrix`` and ``result.P`` are resident transport. Existing
push/pull and the output marginals ``result.a``/``result.b`` use only this block.
They do not automatically include recruited sources. ``result.R``, ``result.S``,
``recruitment_mass``, ``resident_contribution``, ``total_recruitment``, and
``total_loss`` provide explicit open-system analyses. ``augmented_transport_matrix``
materializes the full block matrix only when requested. Multiply masses by the
pair's target count to recover cell-count units. Loss is death **or** egress; these
mechanisms cannot be distinguished by transport alone. Posterior growth rates are
not inferred from resident persistence and return ``None`` in this mode.

Diagnostics include objective history, the generalized gap, outer/inner iteration
counts, status, maximum source/target residuals, and the signed mass-balance residual
:math:`\sum b-\sum\widetilde q-\sum R+\sum S`. Prepared metadata and all output blocks
are retained by the existing ``save``/``load`` methods.

FGW is nonconvex: convergence means a stationary solution, not a certified global
minimum. The circulating pool has unlimited supply; duplicating reference rows
changes its entropic recruitment prior. Reference sampling and penalty calibration
therefore matter. There is no reference-capacity estimation or death-versus-egress
identification. Explicit sources/sinks do not implement KL-relaxed marginals:
``tau_a``/``tau_b`` must equal one.

Spatial products reuse OTT geometry operators, including ``batch_size`` for online
point clouds. Spatial costs need not be materialized; however, resident and
recruitment couplings and linear costs remain dense, requiring :math:`O(NM+KM)`
storage. With general dense spatial costs, gradient work is
:math:`O(N^2M+NM^2)`; each inner scaling step is :math:`O(NM+KM)`. Squared-Euclidean
geometry products can exploit OTT's structure. ``batch_size`` does not remove the
dense coupling requirement. Low-rank solvers and cost approximations are rejected.
Inner scaling and geometry operations support JIT and JAX devices; the outer
line search runs on the host and synchronizes objective values. This is not an
end-to-end differentiable solver. ``device`` follows moscot's output-transfer
convention; select a JAX default device to control where optimization runs.

Runnable example
----------------

Download :download:`examples/open_system_merfish.py` and run it with no input files
to create and load a small synthetic MERFISH-like dataset and circulating reference:

.. code-block:: console

    python docs/examples/open_system_merfish.py --output /tmp/open-system-demo

It solves a two-timepoint problem, checks constraints, saves the model, and plots
recruitment, persistence, and loss grouped by cell type. Replace the synthetic data
with prepared files using ``--residents merfish.h5ad --circulating blood.h5ad``.
The script documents the required metadata; its demo counts and costs are illustrative.

.. autoclass:: moscot.problems.spatiotemporal.OpenSystemConfig
    :members:
