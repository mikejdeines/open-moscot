"""Run an open-system MERFISH example, or load two prepared h5ad files.

Resident input: obs time (0/1), lineage, beta; obsm state, spatial;
uns cell_counts (string time keys) and vascular_coordinates (string target keys).
Reference input: obs lineage, entry; obsm state in the SAME basis as resident state.
With no files supplied, generate small synthetic inputs and load them from disk.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import anndata as ad

from moscot.problems.spatiotemporal import OpenSystemConfig, SpatioTemporalProblem


def synthetic_inputs(directory: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(10)
    resident = ad.AnnData(np.zeros((24, 3)))
    resident.obs["time"] = np.repeat([0, 1], 12)
    resident.obs["lineage"] = np.tile(np.repeat(["immune", "malignant", "stromal"], 4), 2)
    resident.obs["beta"] = np.tile(np.repeat([0.05, 0.2, 0.0], 4), 2)
    centers = np.tile(np.repeat(np.eye(3), 4, axis=0), (2, 1))
    resident.obsm["state"] = centers + rng.normal(0, 0.05, (24, 3))
    resident.obsm["spatial"] = rng.uniform(0, 1, (24, 2))
    resident.uns["cell_counts"] = {"0": 1000, "1": 1300}
    resident.uns["vascular_coordinates"] = {"1": np.array([[0.2, 0.4], [0.8, 0.7]])}
    reference = ad.AnnData(np.zeros((4, 3)))
    reference.obs["lineage"] = ["immune", "immune", "malignant", "malignant"]
    reference.obs["entry"] = [0.2, 0.2, 0.3, 0.3]
    reference.obsm["state"] = np.repeat(np.eye(3)[:2], 2, axis=0) + rng.normal(0, 0.05, (4, 3))
    resident_path, reference_path = directory / "merfish.h5ad", directory / "circulating.h5ad"
    resident.write_h5ad(resident_path)
    reference.write_h5ad(reference_path)
    return resident_path, reference_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residents", type=Path)
    parser.add_argument("--circulating", type=Path)
    parser.add_argument("--output", type=Path, default=Path("open-system-demo"))
    args = parser.parse_args()
    if (args.residents is None) != (args.circulating is None):
        parser.error("Supply both --residents and --circulating, or neither for synthetic data.")
    args.output.mkdir(parents=True, exist_ok=True)
    resident_path, reference_path = (
        synthetic_inputs(args.output) if args.residents is None else (args.residents, args.circulating)
    )
    resident, reference = ad.read_h5ad(resident_path), ad.read_h5ad(reference_path)
    counts = {float(t): float(n) for t, n in resident.uns["cell_counts"].items()}
    vessels = {float(t): coords for t, coords in resident.uns["vascular_coordinates"].items()}
    model = (
        SpatioTemporalProblem(resident)
        .prepare(
            "time",
            joint_attr="state",
            normalize_spatial=False,
            open_system=OpenSystemConfig(
                cell_counts=counts,
                circulating=reference,
                proliferation_rates="beta",
                sink_cost=0.8,
                entry_cost="entry",
                vascular_coordinates=vessels,
                vascular_weight=0.2,
            ),
        )
        .solve(alpha=0.2, epsilon=0.1, threshold=1e-6, batch_size=16)
    )
    for (source, target), problem in model.problems.items():
        result = problem.solution
        P, R, S = result.P, result.R, result.S
        assert result.converged, result.diagnostics
        np.testing.assert_allclose(P.sum(1) + S, problem.a, atol=1e-6)
        np.testing.assert_allclose(P.sum(0) + R.sum(0), problem.b, atol=1e-6)
        target_frame = (
            pd.DataFrame(
                {
                    "lineage": problem.adata_tgt.obs["lineage"].to_numpy(),
                    "recruitment": counts[target] * np.asarray(result.recruitment_mass),
                    "persistence": counts[target] * np.asarray(result.resident_contribution),
                }
            )
            .groupby("lineage")
            .sum()
        )
        loss = (
            pd.DataFrame(
                {
                    "lineage": problem.adata_src.obs["lineage"].to_numpy(),
                    "loss": counts[target] * np.asarray(S),
                }
            )
            .groupby("lineage")
            .sum()
        )
        summary = target_frame.join(loss, how="outer").fillna(0)
        ax = summary.plot.bar(rot=0, ylabel="Estimated cells", title=f"{source} → {target}")
        ax.figure.tight_layout()
        ax.figure.savefig(args.output / f"mass-{source}-{target}.png", dpi=150)
        plt.close(ax.figure)
        summary.to_csv(args.output / f"mass-{source}-{target}.csv")
    model.save(args.output / "open-system.pkl", overwrite=True)


if __name__ == "__main__":
    main()
