"""Build the self-contained PercInterp E08 static site for GitHub Pages.

Calibration fixtures are synthetic. The E07 reference table is transcribed from
saved measurements, and is explicitly not used to drive the demo globe.

    python experiments/e08_build_site.py --output out/e08_site
    python -m http.server 8000 -d out/e08_site
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e08_holonomy_percolation import make_fixture  # noqa: E402


def build_site(output: Path) -> dict:
    source = ROOT / "ui" / "godel_globe"
    output = output.resolve()
    if output == source.resolve() or source.resolve() in output.parents:
        raise ValueError("output must be outside the tracked source UI directory")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html",):
        shutil.copy2(source / name, output / name)
    library_page = output / "library" / "index.html"
    library_page.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "ui" / "library" / "index.html", library_page)

    fixtures = {mode: make_fixture(mode) for mode in ("planted", "flat", "tangent")}
    for mode, data in fixtures.items():
        if data["provenance"]["type"] != "SYNTHETIC_CALIBRATION":
            raise AssertionError(f"unexpected provenance for {mode}")
        if data["n_nodes"] != 6 or data["n_edges"] != 7:
            raise AssertionError(f"unexpected graph for {mode}")
    (output / "calibration_data.js").write_text(
        "/* Synthetic fixtures built by experiments/e08_build_site.py. */\n"
        "window.PERCINTERP_CALIBRATIONS = " +
        json.dumps(fixtures, indent=2, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return {
        "page": str(output / "index.html"),
        "library_page": str(library_page),
        "fixtures": {mode: {"cycles": len(data["loops"]), "p_h": data["p_h"]}
                     for mode, data in fixtures.items()},
        "provenance": "synthetic only; E07 statistics displayed separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "e08_site")
    args = parser.parse_args()
    print(json.dumps(build_site(args.output), indent=2))


if __name__ == "__main__":
    main()
