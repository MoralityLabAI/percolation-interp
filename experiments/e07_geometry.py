"""Exploratory component/neuron percolation from saved VPD factors; CPU only."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from percinterp.polysemanticity import (bond_curve, configuration_giant, degree_summary,
                                      direction_energy, mix_basis, support_edges)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("p-7e116244", "p-06edd191")
BASE = Path(r"C:\Users\patri\param_decomp_out\decompositions")
KS = (1, 2, 3, 4, 6, 8, 12, 16)
PS = np.linspace(0, 1, 41)
DRAWS, K_DRAWS, BASE_K, SEED = 32, 8, 8, 71022


def distribution(a):
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "q05": float(np.quantile(a, .05)), "q95": float(np.quantile(a, .95))}


def half_point(p, giant):
    ids = np.flatnonzero(giant >= .5)
    if not len(ids):
        return None
    i = int(ids[0])
    return float(p[i]) if i == 0 else float(np.interp(.5, giant[i-1:i+1], p[i-1:i+1]))


def uniform_support(n_left, n_right, k, rng):
    return np.repeat(np.arange(n_left), k), np.concatenate([
        rng.choice(n_right, k, replace=False) for _ in range(n_left)])


def summarize_curves(curves):
    giant = np.array([x["giant_left_fraction"] for x in curves])
    chi = np.array([x["susceptibility_left"] for x in curves])
    return {"mean": giant.mean(0).tolist(), "q05": np.quantile(giant, .05, axis=0).tolist(),
            "q95": np.quantile(giant, .95, axis=0).tolist(),
            "mean_susceptibility": chi.mean(0).tolist(),
            "p50_mean_curve": half_point(PS, giant.mean(0)),
            "p50_draws": [half_point(PS, g) for g in giant],
            "auc_mean": float(np.trapezoid(giant.mean(0), PS))}


def analyze(a, seed):
    rng = np.random.default_rng(seed)
    q = direction_energy(a)
    nl, nr = q.shape
    normalized = np.asarray(a, dtype=float) / np.linalg.norm(a, axis=1, keepdims=True)
    gram = normalized @ normalized.T
    off = gram[np.triu_indices(nl, 1)]
    rotated = mix_basis(normalized, rng)
    rows = np.arange(min(64, nl))
    gram_error = float(np.max(np.abs(rotated[rows] @ rotated[rows].T - gram[np.ix_(rows, rows)])))
    assert gram_error < 1e-10
    rotated_q = direction_energy(rotated)
    positive_columns = (q * q).sum(0) > 0
    coordinate_pr = q.sum(0)[positive_columns] ** 2 / (q * q).sum(0)[positive_columns]
    record = {"n_components": nl, "n_neurons": nr,
              "component_coordinate_participation": distribution(1 / (q * q).sum(1)),
              "coordinate_component_participation": distribution(coordinate_pr),
              "mean_pair_cosine_squared": float(np.mean(off * off)),
              "basis_mix_max_gram_error": gram_error,
              "basis_mix_component_coordinate_participation": distribution(1 / (rotated_q * rotated_q).sum(1)),
              "top_k": []}
    for k in KS:
        left, right = support_edges(q, k)
        actual = bond_curve(left, right, nl, nr, [1.], np.zeros(len(left)))
        degree = degree_summary(left, right, nl, nr)
        uniform, config = [], []
        for _ in range(K_DRAWS):
            sl, sr = uniform_support(nl, nr, k, rng)
            uniform.append(float(bond_curve(sl, sr, nl, nr, [1.], np.zeros(len(sl)))["giant_left_fraction"][0]))
            shuffled = rng.permutation(right)
            config.append(float(bond_curve(left, shuffled, nl, nr, [1.], np.zeros(len(left)))["giant_left_fraction"][0]))
        kept = np.bincount(left, weights=q[left, right], minlength=nl)
        record["top_k"].append({"k": k, "energy_captured": distribution(kept),
            "giant_left_fraction": float(actual["giant_left_fraction"][0]),
            "branching_product": degree["branching_product"],
            "configuration_null_giant": distribution(config),
            "uniform_null_giant": distribution(uniform)})
    left, right = support_edges(q, BASE_K)
    rl, rr = support_edges(rotated_q, BASE_K)
    degree = degree_summary(left, right, nl, nr)
    hubs = np.argsort(-degree["right"], kind="stable")[:10]
    record["top_contact_coordinates"] = [{
        "neuron_index": int(j), "top8_component_contacts": int(degree["right"][j]),
        "effective_components_unit_row_energy": float(q[:, j].sum() ** 2 / np.sum(q[:, j] ** 2)) if np.sum(q[:, j] ** 2) else 0,
        "largest_energy_component_rows": np.argsort(-q[:, j], kind="stable")[:5].tolist(),
    } for j in hubs]
    groups = {key: [] for key in ("learned", "configuration", "uniform", "rotated")}
    duplicates = []
    for _ in range(DRAWS):
        uniforms = rng.random(len(left))
        groups["learned"].append(bond_curve(left, right, nl, nr, PS, uniforms))
        shuffled = rng.permutation(right)
        duplicates.append(1 - len(np.unique(left * nr + shuffled)) / len(left))
        groups["configuration"].append(bond_curve(left, shuffled, nl, nr, PS, uniforms))
        sl, sr = uniform_support(nl, nr, BASE_K, rng)
        groups["uniform"].append(bond_curve(sl, sr, nl, nr, PS, uniforms))
        groups["rotated"].append(bond_curve(rl, rr, nl, nr, PS, uniforms))
    bond = {key: summarize_curves(curves) for key, curves in groups.items()}
    bond["configuration_theory"] = configuration_giant(degree["left"], degree["right"], PS).tolist()
    bond["configuration_pc"] = degree["configuration_bond_pc"]
    bond["configuration_parallel_edge_fraction"] = distribution(duplicates)
    bond["left_degree_histogram"] = np.bincount(degree["left"]).tolist()
    bond["right_degree_histogram"] = np.bincount(degree["right"]).tolist()
    bond["auc_learned_minus_configuration"] = bond["learned"]["auc_mean"] - bond["configuration"]["auc_mean"]
    bond["auc_learned_minus_uniform"] = bond["learned"]["auc_mean"] - bond["uniform"]["auc_mean"]
    bond["configuration_theory_max_abs_error"] = float(np.max(np.abs(
        np.asarray(bond["configuration"]["mean"]) - bond["configuration_theory"])))
    record["bond"] = bond
    return record


def calibration():
    rng = np.random.default_rng(SEED + 999)
    nl, nr, k = 3000, 12000, 8
    left, right = uniform_support(nl, nr, k, rng)
    degree = degree_summary(left, right, nl, nr)
    mean = np.mean([bond_curve(left, right, nl, nr, PS, rng.random(len(left)))["giant_left_fraction"]
                    for _ in range(8)], axis=0)
    theory = configuration_giant(degree["left"], degree["right"], PS)
    error = float(np.max(np.abs(mean[PS >= .4] - theory[PS >= .4])))
    assert error < .05, f"Random-support calibration error {error}"
    return {"n_components": nl, "n_coordinates": nr, "k": k, "draws": 8,
            "empirical": mean.tolist(), "configuration_theory": theory.tolist(),
            "max_abs_error_for_p_ge_0_4": error,
            "configuration_pc": degree["configuration_bond_pc"]}


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), sharex=True, sharey=True)
    colors = {"learned": "#184e77", "configuration": "#f08a24", "uniform": "#78918a", "rotated": "#9a6baf"}
    for ax, (name, rec) in zip(axes.ravel(), result["maps"].items()):
        for key in colors:
            curve = rec["bond"][key]
            ax.plot(PS, curve["mean"], color=colors[key], label=key, lw=1.7)
            if key in ("learned", "configuration"):
                ax.fill_between(PS, curve["q05"], curve["q95"], color=colors[key], alpha=.12)
        ax.plot(PS, rec["bond"]["configuration_theory"], color="black", ls=":", lw=1, label="degree PGF theory")
        ax.axvline(rec["bond"]["configuration_pc"], color="black", ls="--", lw=.7, alpha=.45)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.grid(alpha=.16)
    for ax in axes[-1]:
        ax.set_xlabel("Independent bond retention p")
    for ax in axes[:, 0]:
        ax.set_ylabel("Largest cluster / all components")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("VPD component–neuron sharing: structural mixing, not semantic labels\n"
                 "Top 8 coordinates per component; shaded ranges: 5–95% of 32 Monte Carlo draws", fontsize=13)
    fig.tight_layout(rect=(0, .045, 1, .95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = {"exploratory": True, "rng_seed": SEED, "k_values": list(KS),
              "primary_k": BASE_K, "bond_draws": DRAWS, "k_null_draws": K_DRAWS,
              "p_values": PS.tolist(), "population": "all nonzero parameter subcomponents, not firing-filtered",
              "semantic_labels_available": False, "maps": {}, "checkpoints": []}
    retrieval = json.loads((ROOT / "results/e05b_retrieval.json").read_text())
    result["random_support_calibration"] = calibration()
    for seed, run in enumerate(RUNS):
        checkpoint = BASE / run / "model_20000.pth"
        assert checkpoint.stat().st_size == retrieval["runs"][str(seed)]["final_checkpoint_bytes"]
        result["checkpoints"].append({"seed": seed, "run_id": run, "path": str(checkpoint),
                                     "verified_sha256": retrieval["runs"][str(seed)]["sha256"]})
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        for layer in (0, 1):
            for role, suffix in (("gate_write", "gate_proj.U"), ("up_write", "up_proj.U"), ("down_read", "down_proj.V")):
                tensor = state[f"_components.model-layers-{layer}-mlp-{suffix}"]
                a = tensor.numpy().T if role == "down_read" else tensor.numpy()
                nonzero = np.max(np.abs(a), axis=1) > 0
                a = a[nonzero]
                name = f"seed{seed}_layer{layer}_{role}"
                rec = analyze(a, SEED + 100 * seed + 10 * layer + len(result["maps"]))
                rec["excluded_zero_components"] = int(np.sum(~nonzero))
                rec["original_component_ids"] = np.flatnonzero(nonzero).tolist()
                result["maps"][name] = rec
                print(f"{name}: neuron participation {rec['coordinate_component_participation']['median']:.1f}; "
                      f"p50 {rec['bond']['learned']['p50_mean_curve']}; "
                      f"AUC learned-config {rec['bond']['auc_learned_minus_configuration']:+.4f}; "
                      f"elapsed {time.monotonic()-started:.0f}s", flush=True)
        del state
    result["elapsed_seconds"] = time.monotonic() - started
    directory = ROOT / "results/e07"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "geometry.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite a completed exploratory result")
    path.write_text(json.dumps(result, indent=1) + "\n")
    plot(result, args.figure)
    print(f"Wrote {path} and {args.figure}", flush=True)


if __name__ == "__main__":
    main()
