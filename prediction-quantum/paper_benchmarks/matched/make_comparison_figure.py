"""Figure 9.2: the best GNB model against the best variational quantum classifier on the
identical recording-grouped splits of the quantum benchmark (three seeds).

Reads ../data/matched_benchmark_summary.csv and ../data/matched_per_class_summary.csv
(written by matched_gnb_vs_vqc.py) and writes ../figures/fig_9_2_gnb_vs_vqc.png plus the
plotted values to ../data/fig_9_2_source.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import pandas as pd
from matplotlib import font_manager

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[2]   # prediction-quantum
DATA = ROOT / 'results/paper/matched'
FIG = DATA / 'figures'
FIG.mkdir(exist_ok=True)

# entity -> (label, colour, hatch); categorical slots 1-3 of the validated default palette,
# hatching as the secondary encoding for grayscale print
# first two fields are the model / variant keys that pick each row out of the two summary CSVs
MODELS = [
    ('GNB subject-specific chain (PyTorch)', 'train', 'Best GNB: subject-specific chain', '#2a78d6', ''),
    ('GNB improved global chain (PyTorch, argmax)', 'train', 'GNB improved global chain', '#1baf7a', '///'),
    ('VQC 6q x 4 layers', 'train (val for checkpoint selection)', 'Best VQC: 6 qubits × 4 layers', '#eb6834', '\\\\\\'),
]
METRICS = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1'), ('balanced_accuracy', 'Balanced\naccuracy'),
           ('macro_precision', 'Macro\nprecision'), ('macro_recall', 'Macro\nrecall')]
COMMANDS = ['backward', 'forward', 'landing', 'left', 'right', 'takeoff']
CHANCE = 1 / 6
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'

# Arial where installed, otherwise its metric-compatible twin Liberation Sans
_FAMILY = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'Liberation Sans'
plt.rcParams.update({'font.family': _FAMILY, 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6,
                     'hatch.linewidth': 0.6, 'hatch.color': 'white'})


def main():
    """Build the two-panel figure from the matched summary CSVs under DATA: panel A shows the five overall
    metrics, panel B the per-command F1, both as mean ± sd across seeds. Saves the 300 dpi PNG and a CSV of
    every plotted value so the numbers in the figure can be checked against the text."""
    summ = pd.read_csv(DATA / 'matched_benchmark_summary.csv')
    pcs = pd.read_csv(DATA / 'matched_per_class_summary.csv')
    rows = []
    # 6.5 in is the full text width of the manuscript; panel B is wider because it holds six groups to A's five
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={'width_ratios': [5.4, 6]})
    width = 0.26
    for j, (model, variant, label, colour, hatch) in enumerate(MODELS):
        # the summary is already averaged over seeds, so each (model, variant) pair is exactly one row
        s = summ[(summ.model == model) & (summ.variant == variant)].iloc[0]
        xs, ys, es = [], [], []
        for k, (m, _) in enumerate(METRICS):
            # j - 1 centres the middle model on the tick and puts the other two one bar width either side
            xs.append(k + (j - 1) * width)
            ys.append(s[f'{m}_mean'])
            es.append(s[f'{m}_sd'])
            rows.append({'panel': 'A', 'model': label, 'metric': m, 'mean': s[f'{m}_mean'], 'sd': s[f'{m}_sd'],
                         'n_seeds': int(s['n_seeds']), 'n_test_windows_mean': s['n_test_mean']})
        # bars are drawn 0.03 narrower than their slot so neighbours within a group do not touch
        ax1.bar(xs, ys, width - 0.03, color=colour, hatch=hatch, label=label, zorder=2, edgecolor='white',
                linewidth=0.4)
        ax1.errorbar(xs, ys, yerr=es, fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=3)
        # selective direct labels: accuracy only
        ax1.text(xs[0], ys[0] + es[0] + 0.02, f'{ys[0]:.2f}', ha='center', va='bottom', fontsize=7, color=INK)
        p = pcs[(pcs.model == model) & (pcs.variant == variant)].set_index('cls')
        xs2 = [c + (j - 1) * width for c in range(len(COMMANDS))]
        ys2 = [p.loc[c, 'f1_mean'] for c in COMMANDS]
        es2 = [p.loc[c, 'f1_sd'] for c in COMMANDS]
        for c, y, e in zip(COMMANDS, ys2, es2):
            rows.append({'panel': 'B', 'model': label, 'metric': f'f1_{c}', 'mean': y, 'sd': e,
                         'n_seeds': int(s['n_seeds']), 'n_test_windows_mean': s['n_test_mean']})
        ax2.bar(xs2, ys2, width - 0.03, color=colour, hatch=hatch, zorder=2, edgecolor='white', linewidth=0.4)
        ax2.errorbar(xs2, ys2, yerr=es2, fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=3)
    for ax in (ax1, ax2):
        ax.axhline(CHANCE, color=MUTED, ls=(0, (4, 3)), lw=0.8, zorder=1)
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    ax1.set_xticks(range(len(METRICS)))
    ax1.set_xticklabels([t for _, t in METRICS], fontsize=7)
    ax1.set_ylabel('Score on the held-out test windows')
    ax1.set_title('A  Overall scores', loc='left', fontsize=9, fontweight='bold', color=INK)
    ax2.set_xticks(range(len(COMMANDS)))
    ax2.set_xticklabels(COMMANDS, fontsize=7.5)
    ax2.set_ylabel('F1 score per command')
    ax2.set_title('B  F1 per command', loc='left', fontsize=9, fontweight='bold', color=INK)
    handles, labels = ax1.get_legend_handles_labels()
    # axhline carries no label, so a proxy line stands in for the chance level in the shared legend
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=MUTED, ls=(0, (4, 3)), lw=0.8))
    labels.append('Chance (1/6)')
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=7, columnspacing=1.2, handlelength=1.8,
               bbox_to_anchor=(0.5, -0.01))
    # keep the bottom 7 % of the canvas free for the figure-level legend
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG / 'fig_9_2_gnb_vs_vqc.png'
    fig.savefig(out, dpi=300)
    pd.DataFrame(rows).to_csv(DATA / 'fig_9_2_source.csv', index=False)
    print(out)


if __name__ == '__main__':
    main()
