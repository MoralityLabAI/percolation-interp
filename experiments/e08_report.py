"""Render computed e08 measurements; final scientific review remains explicit."""
from pathlib import Path
import numpy as np
from percinterp.e08_io import read_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    geometry = read_json(ROOT / 'results/e08/weighted.json')
    contexts = read_json(ROOT / 'results/e08/contextual.json')
    labels = read_json(ROOT / 'out/e08/labels.json')
    lines = ['# e08: contextual contributions and energy-weighted percolation', '',
        'Computed locally from the two saved VPD decompositions of one Qwen3-0.6B target.',
        'Exploratory selected cases; no new training, cloud rental or model download.', '',
        '## Contextual intervention cases', '',
        '| Case | Status | Directed KL interactions | Holm p, six cases |',
        '|---|---|---|---|']
    for name, case in contexts['cases'].items():
        effect = ', '.join(f'{v:.5g}' for v in case.get('directed_interactions', [])) or 'unscored'
        lines.append(f"| {name} | {case['status']} | {effect} | {case['p_holm_six_cases']:.5g} |")
    lines.extend(['', 'The directed interactions compare matched contextual families on KL to the',
        'unmodified CI-masked, delta-retaining model. A positive value in both',
        'directions supports selective disruption in the reviewed contextual',
        'families. It does not establish a semantic feature count, interference,',
        'or generality across independently trained models. Unscored cases remain',
        'in the six-case multiplicity correction with p=1.', '',
        f"Label reviewer: {labels['reviewer']}. Method: {labels['method']}.",
        'No independent inter-annotator agreement was measured. Cluster labels',
        'were reviewed on discovery prefixes before evaluating interventions.',
        'Semantic coherence on evaluation rows remains an unmeasured limitation.', '',
        'Identity hooks, retained-delta reconstruction and fixed-CI paired',
        'effects are checked by the runner. The original e05 gate is reported',
        'separately in every case; e08 does not revise it.', '',
        '## All-energy geometry', '',
        '| Map | Learned t50 | Row-permuted t50 | Uniform t50 | Median contacts for 90% energy |',
        '|---|---:|---:|---:|---:|'])
    for name, item in geometry['maps'].items():
        values = [item['curves'][key]['t50_mean_curve'] for key in ('learned', 'row_permuted', 'uniform')]
        formatted = ['not reached' if v is None else f'{v:.5g}' for v in values]
        contacts = np.median(item['curves']['learned']['contacts_for_90pct_energy'])
        lines.append(f"| {name} | {' | '.join(formatted)} | {contacts:.0f} |")
    lines.extend(['', 'These are descriptive finite-graph crossovers on a fixed logarithmic',
        'exposure grid, with 32 nested draws. Exposure t is not e07 bond probability p.',
        'Each edge has retention 1-exp(-t*N*q), where q is normalized squared',
        'direction energy. Row permutation preserves row energy spectra, not',
        'coordinate strengths. All directions have equal total weight; activity,',
        'component scale and the delta term do not enter this geometric graph.', '',
        'The binary degree-moment branching formula was not applied to these',
        'nonuniform edges. No weighted critical-point prediction is claimed.', '',
        '![Weighted percolation curves](../results/e08/weighted_curves.png)', '',
        'Protocol: `docs/e08_design.md`. Raw results: `results/e08/weighted.json`',
        'and `results/e08/contextual.json`. Contexts, frozen labels, matching plans',
        'and row-level paired effects are retained locally in `out/e08/`.', '',
        'This generated report requires final agent review before reporting the',
        'workflow complete. Read the raw per-family effects and matched controls',
        'before making any positive functional interpretation.', ''])
    path = ROOT / 'docs/e08_results.md'
    if path.exists():
        raise FileExistsError('refusing to overwrite an existing reviewed report')
    path.write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
    for ax, (name, item) in zip(axes.flat, geometry['maps'].items()):
        for key, color in (('learned', '#1565c0'), ('row_permuted', '#d17c00'), ('uniform', '#666666')):
            curve = item['curves'][key]
            t = np.array(item['exposures'])
            ax.plot(t[1:], np.array(curve['mean_giant'])[1:], color=color, label=key.replace('_', ' '))
        ax.set_xscale('log'); ax.set_ylim(0, 1.03); ax.grid(alpha=.2)
        ax.set_title(name.replace('_', ' '), fontsize=9)
    handles, legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=3, frameon=False)
    fig.supxlabel('Energy-weighted contact exposure t'); fig.supylabel('Largest cluster / all components')
    fig.suptitle('e08: all-energy geometric percolation, two decompositions of one target model')
    fig.tight_layout(rect=(.02, .07, 1, .94))
    fig.savefig(ROOT / 'results/e08/weighted_curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
