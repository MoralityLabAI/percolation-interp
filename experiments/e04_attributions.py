"""e04: dataset attributions for the Qwen slice with param-decomp's harvester, embedding excluded.

Same as param_decomp.dataset_attributions.harvest.harvest_attributions, except:
  - the embedding is not a source of components (the slice's input is the residual after
    layers 0-1; vocabulary x components accumulators would also need ~9 GB);
  - alive masks come from e04_stats.py (CI > 0.1 on >= 1e-5 of held-out tokens) instead of a
    harvest run;
  - tokens are the "attributions" part of the held-out rows, disjoint from the rows the
    causal test (e04_causal.py) measures on.

  python experiments/e04_attributions.py [--step N]
Writes out/e04/dataset_attributions.pt (param-decomp's storage format).
"""

from __future__ import annotations

import argparse
import time

import torch

from e04_common import MIN_RATE, OUT, find_run, heldout_part, install_qwen3_schema, load_component_model, load_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()

    from param_decomp.dataset_attributions.harvester import AttributionHarvester
    from param_decomp.topology import TransformerTopology, get_sources_by_target

    install_qwen3_schema()
    run_id = find_run(args.step)
    cm, run_info = load_component_model(run_id, args.device)
    stats = load_stats()
    assert stats["run_id"] == run_id, "stats are from a different run"
    alive = {m: (stats["n_fire"][m] / stats["tokens"] >= MIN_RATE) for m in cm.target_module_paths}

    topology = TransformerTopology(cm.target_model)
    unembed_path = topology.path_schema.unembed_path
    raw = get_sources_by_target(cm, topology, args.device, run_info.config.sampling)
    components = set(cm.target_module_paths)
    embed_path = topology.path_schema.embedding_path
    sources_by_target = {}
    for target, sources in raw.items():
        if target not in components | {unembed_path}:
            continue
        # The embedding is excluded as a source of components (vocab x C accumulators, ~9 GB).
        # The harvester requires it as a source of the output, where it only feeds the
        # (d_model x vocab) straight-through term, ~0.6 GB.
        kept = [s for s in sources if s in components or (target == unembed_path and s == embed_path)]
        if any(s in components for s in kept):
            sources_by_target[target] = kept
    print(f"{len(sources_by_target)} targets with component sources", flush=True)

    harvester = AttributionHarvester(model=cm, topology=topology, sources_by_target=sources_by_target,
                                     component_alive=alive, sampling=run_info.config.sampling)
    it, t0 = heldout_part(args.batch, "attributions"), time.time()
    for b in range(args.n_batches):
        harvester.process_batch(next(it).to(args.device))
        print(f"batch {b + 1}/{args.n_batches}: {time.time() - t0:.0f}s", flush=True)  # the driver's watchdog reads this
    storage = harvester.finalize(0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    storage.save(OUT / "dataset_attributions.pt")
    print(f"saved; {harvester.n_tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
