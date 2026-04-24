"""Plot model parameter count vs Test F1."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SHOW_EXTRAPOLATION = True

# ── Data ──────────────────────────────────────────────────────────────────────
# Each entry: (label, params_M, test_f1, style_kwargs)
# style_kwargs are passed directly to ax.scatter — extend as needed.
MODELS = [
    ("ResNet Fusion",            23.6,   0.5407, {'c':'green'}),
    ("Two-Stream Convolution",   47.2,   0.7008, {'c':'green'}),
    ("Pose Estimation",          28.8,   0.4316, {'c':'green'}),
    ("VideoMAE",                 86.2,   0.6253, {'c':'blue'}),
    ("ViViT",                    88.7,   0.7685, {'c':'blue'}),
    ("VideoPrism (vision only)", 114,    0.8466, {'c':'blue','marker':'s'}),
    ("LLaVA-OV, No prompt",      894,    0.6767, {'c':'orange','marker':'s'}),
    ("LLaVA-OV, Prompt",         894,    0.6970, {'c':'orange','marker':'s'}),
    ("Qwen3-VL",                 7065,   0.7358, {'c':'orange','marker':'*'}),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

for label, params, f1, style in MODELS:
    kw = dict(s=80, zorder=3)
    kw.update(style)
    ax.scatter(params, f1, **kw)
    ax.annotate(
        label,
        xy=(params, f1),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
    )

ax.set_xscale("log")
ax.set_xlabel("Parameters (Million)")
ax.set_ylabel("Test F1")
ax.set_title("Model size vs. Test F1")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_xlim(10, 20000)
ax.set_ylim(-0.0, 1)

if SHOW_EXTRAPOLATION:
    # Pareto frontier: models where no other model has fewer params AND higher F1.
    sorted_by_params = sorted(MODELS, key=lambda m: m[1])
    pareto, best_f1 = [], -1.0
    for _, params, f1, _ in sorted_by_params:
        if f1 > best_f1:
            pareto.append((params, f1))
            best_f1 = f1

    pp = np.array([m[0] for m in pareto])
    pf = np.array([m[1] for m in pareto])
    log_pp = np.log(pp)

    x_line = np.logspace(np.log10(10), np.log10(20000), 400)
    log_x  = np.log(x_line)

    # Regression 1: F1 ~ a·ln(params) + b  (linear on this log-scale plot)
    #c1 = np.polyfit(log_pp, pf, 1)
    #ax.plot(x_line, np.polyval(c1, log_x),
    #        'k--', linewidth=1, label='Linear fit')
    #print(f"Linear fit:    F1 = {c1[0]:.4f} * ln(params) + {c1[1]:.4f}")

    # Regression 2: ln(1-F1) ~ a·ln(params) + b  (power-law error decay)
    c2 = np.polyfit(log_pp, np.log(1.0 - pf), 1)
    y2 = np.polyval(c2, log_x)
    ax.plot(x_line, 1.0 - np.exp(y2),
            'k:', linewidth=1, label='Log-error fit')
    print(f"Log-error fit: ln(1-F1) = {c2[0]:.4f} * ln(params) + {c2[1]:.4f}")

    fit_handles = [
    #    Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='Linear fit'),
        Line2D([0], [0], color='k', linestyle=':',  linewidth=1, label='Log-error fit'),
    ]
    leg3 = ax.legend(handles=fit_handles, title='Pareto extrapolation',
                     loc='upper left', fontsize=7, title_fontsize=7)
    ax.add_artist(leg3)

color_handles = [
    Patch(facecolor='green',  label='CNN'),
    Patch(facecolor='blue',   label='Vision Transformer'),
    Patch(facecolor='orange', label='Vision Language Model'),
]
leg1 = ax.legend(handles=color_handles, title='Architecture', loc='lower left')
ax.add_artist(leg1)

shape_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Full Fine Tune'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=8, label='Classifier Only'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='grey', markersize=10, label='Zero-shot'),
]
#ax.legend(handles=shape_handles, title='Training Regime', loc='lower center')
ax.legend(handles=shape_handles, title='Training Regime', loc='lower left', bbox_to_anchor=(0.44, 0.0))

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()
print("Saved results.png")
