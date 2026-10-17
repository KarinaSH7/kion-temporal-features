"""Plot saved results across all recorded training seeds."""
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tlab-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import ROOT


def main():
    results = ROOT / 'results/main'
    df = pd.read_csv(results / 'test_metrics.csv')
    ci = json.loads((results / 'paired_bootstrap.json').read_text())
    seeds = [42, 17, 2026]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout='constrained')
    for arm, label, color, shift in [('manual', 'Manual', '#294768', -.09),
                                     ('auto_selected', 'Selected', '#147d6f', .09)]:
        values = [float(df[(df.arm == arm) & (df.seed == s)].ndcg10.iloc[0]) for s in seeds]
        axes[0].scatter(values, np.arange(3)+shift, s=55, color=color, label=label, zorder=3)
    axes[0].set_xlim(.112, .139)
    axes[0].set_xlabel('Test NDCG@10')
    axes[0].set_title('Quality across training seeds', loc='left', pad=15)
    axes[0].legend(frameon=False, loc='lower right')
    delta = np.array([ci[str(s)]['delta'] for s in seeds])
    lower = np.array([ci[str(s)]['ci95_low'] for s in seeds])
    upper = np.array([ci[str(s)]['ci95_high'] for s in seeds])
    axes[1].axvline(0, color='#8897a5', linestyle='--', linewidth=1)
    axes[1].errorbar(delta, np.arange(3), xerr=[delta-lower, upper-delta],
                     fmt='o', color='#147d6f', capsize=5, markersize=6)
    axes[1].set_xlabel('Selected − manual · paired 95% CI')
    axes[1].set_title('Uncertainty of the difference', loc='left', pad=15)
    for ax in axes:
        ax.set_yticks(range(3), [f'Seed {s}' for s in seeds])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.savefig(ROOT / 'reports/seed_comparison.png', dpi=190, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
