"""Depth-arm hyperparameter search (docs/plan.md, "Depth arm").

Trains the given settings sequentially and exports each run's history JSON to
results/e02_search/. With --report, applies the plan's accuracy-only selection
rule to whatever results exist.

  python experiments/e02_search.py --settings A B --device cuda
  python experiments/e02_search.py --report
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from percinterp.modadd import Config, train

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "out" / "e02" / "runs" / "search"
RESULTS = ROOT / "results" / "e02_search"
BASE = dict(P=113, width=128, seed=0, steps=40000)

# The plan's table. Keys are Config fields; depths per setting.
SETTINGS = {
    "A": ({}, [8]),
    "B": ({"init": "kaiming"}, [4, 8]),
    "C": ({"init": "kaiming", "lr": 3e-4}, [4, 8]),
    "D": ({"init": "kaiming", "init_scale": 0.5}, [4, 8]),
    "E": ({"frac_train": 0.5}, [4, 8]),
    "F": ({"init": "kaiming", "frac_train": 0.5}, [4, 8]),
}


def config(setting: str, depth: int, steps: int | None = None) -> Config:
    overrides, _ = SETTINGS[setting]
    base = dict(BASE, steps=steps) if steps else BASE
    return Config(depth=depth, **base, **overrides)


def first_step(history, key, thresh):
    return next((h["step"] for h in history if h[key] >= thresh), None)


def run(settings, device, progress, steps=None, results=RESULTS, runs=RUNS):
    results.mkdir(parents=True, exist_ok=True)
    for s in settings:
        for depth in SETTINGS[s][1]:
            cfg = config(s, depth, steps)
            if (results / f"{cfg.name}.json").exists():
                print(f"skip {s} {cfg.name}", flush=True)
                continue
            r = train(cfg, runs, device=device, progress=progress)
            shutil.copyfile(runs / f"{cfg.name}.json", results / f"{cfg.name}.json")
            h = r["history"]
            print(f"{s} {cfg.name}: {r['seconds']:.0f}s, train acc 99% at {first_step(h, 'train_acc', 0.99)}, "
                  f"test acc 95% at {first_step(h, 'test_acc', 0.95)}, final test acc {r['final']['test_acc']:.3f}",
                  flush=True)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()


def report():
    rows, qualifying = [], []
    for s, (overrides, depths) in SETTINGS.items():
        t = {}
        for depth in depths:
            path = RESULTS / f"{config(s, depth).name}.json"
            if not path.exists():
                t[depth] = "missing"
                continue
            h = json.loads(path.read_text())["history"]
            t[depth] = first_step(h, "test_acc", 0.95)
            rows.append((s, depth, t[depth], h[-1]["test_acc"]))
        # depth 4 under setting A is the step 2 run that did not grok
        if s == "A":
            t[4] = None
        if all(isinstance(v, int) for v in t.values()):
            qualifying.append((max(t.values()), len(overrides), s))
    print("setting depth  t_grok  final_test_acc")
    for s, d, tg, acc in rows:
        print(f"   {s}      {d}   {str(tg):>6}  {acc:.3f}")
    if qualifying:
        best = min(qualifying)
        print(f"selected: {best[2]} (max t_grok {best[0]}, {best[1]} change(s) from step 2)")
    else:
        print("no setting qualifies: both depths must reach 95% test accuracy within 40k steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", choices=sorted(SETTINGS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress", type=int, default=5000)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="smoke test: this many steps, outputs under out/e02/search_smoke (never results/)")
    args = ap.parse_args()
    if args.settings:
        if args.smoke:
            smoke = ROOT / "out" / "e02" / "search_smoke"
            run(args.settings, args.device, args.progress, args.smoke, smoke / "results", smoke / "runs")
        else:
            run(args.settings, args.device, args.progress)
    if args.report:
        report()


if __name__ == "__main__":
    main()
