"""The three benchmark figures (matplotlib, Agg). Colours follow a validated categorical palette in fixed order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed order
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAMILY_STYLE = {"VQC": (SERIES[0], "o"), "Logistic regression": (SERIES[1], "^"), "GaussianNB": (SERIES[2], "s"), "MLP": (SERIES[3], "D")}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "legend.frameon": False, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax, title, subtitle, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=18)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom", ha="left")


def spread_labels(ys, min_gap):
    """Nudge label positions apart (sorted order kept) so neighbours are at least min_gap apart."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def short_vqc(n, L):
    return f"{int(n)}q x {int(L)}"


def end_labels(ax, ends, x_pad_frac=0.01, gap_frac=0.055):
    """Direct labels at the right end of lines, nudged apart by a gap relative to the y-range."""
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    ys = spread_labels([e[1] for e in ends], gap_frac * (y1 - y0))
    x0, x1 = ax.get_xlim()
    for (x, _, name, color), y in zip(ends, ys):
        ax.text(x + x_pad_frac * (x1 - x0), y, name, color=color, fontsize=8, va="center")
    lo, hi = min(ys.min(), y0), max(ys.max() + gap_frac * (y1 - y0) / 2, y1)
    ax.set_ylim(lo, hi)


def annotate_without_overlap(fig, ax, points, texts, color=INK2, fontsize=7.5):
    """Annotate (x, y) points, trying a few offsets per label so that label boxes do not overlap."""
    candidates = [(6, 4), (6, -11), (-6, 4), (-6, -11), (6, 14), (-6, 14), (6, -22), (0, 18)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for (x, y), text in zip(points, texts):
        for dx, dy in candidates:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color,
                            ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top")
            box = t.get_window_extent(renderer).padded(3)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            t.remove()
        else:
            t = ax.annotate(text, (x, y), xytext=candidates[0], textcoords="offset points", fontsize=fontsize, color=color)
            placed.append(t.get_window_extent(renderer))


def fig_convergence(conv, cfg, chance, path):
    n_ref, depths = cfg["report"]["reference_qubits"], list(cfg["vqc"]["n_layers"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    curves, handles = [], []
    for i, L in enumerate(depths):
        d = conv[(conv.model == "vqc") & (conv.n_qubits == n_ref) & (conv.n_layers == L)]
        if len(d):
            curves.append((d, SERIES[i % len(SERIES)], "-", f"VQC {short_vqc(n_ref, L)} layer{'s' if L != 1 else ''}"))
    d = conv[(conv.model == "mlp_pca") & (conv.n_qubits == n_ref)]
    if len(d):
        curves.append((d, INK2, "--", f"MLP-64 (PCA-{n_ref})"))
    n_seeds = conv.seed.nunique()
    ends = []
    for d, color, ls, name in curves:
        g = d.groupby("epoch")[["train_loss", "val_loss", "train_acc", "val_acc"]].mean()
        axes[0].plot(g.index, g.val_loss, color=color, ls=ls, lw=2)
        axes[0].plot(g.index, g.train_loss, color=color, ls=":", lw=1.4, alpha=0.9)
        axes[1].plot(g.index, g.val_acc, color=color, ls=ls, lw=2)
        axes[1].plot(g.index, g.train_acc, color=color, ls=":", lw=1.4, alpha=0.9)
        handles.append(Line2D([], [], color=color, ls=ls, lw=2, label=name))
        ends.append((g.index[-1], g.val_acc.iloc[-1], name, color))
    axes[1].axhline(chance, color=MUTED, ls=":", lw=1)
    axes[1].text(1, chance + 0.004, "chance", color=MUTED, fontsize=8, va="bottom")
    handles += [Line2D([], [], color=INK2, ls="-", lw=2, label="validation"), Line2D([], [], color=INK2, ls=":", lw=1.4, label="training")]
    style(axes[0], "Cross-entropy per epoch", f"mean over {n_seeds} recording-level split{'s' if n_seeds != 1 else ''}; solid = validation, dotted = training",
          "epoch", "cross-entropy")
    style(axes[1], "Accuracy per epoch", f"same runs; the dotted chance line is the majority-class share ({chance:.3f})", "epoch", "accuracy")
    last = max(e[0] for e in ends) if ends else 1
    axes[1].set_xlim(0, last * 1.32)
    axes[1].set_xticks([t for t in axes[1].get_xticks() if 0 <= t <= last])
    end_labels(axes[1], ends)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_accuracy_latency(summary, cfg, chance, path):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    n_seeds = int(summary.n_seeds.max())
    points, texts = [], []
    for fam, (color, marker) in FAMILY_STYLE.items():
        d = summary[summary.family == fam]
        if not len(d):
            continue
        ax.errorbar(d.latency_ms_mean, d.accuracy_mean, xerr=d.latency_ms_sd.fillna(0), yerr=d.accuracy_sd.fillna(0), fmt=marker,
                    color=color, ms=8, mec=SURFACE, mew=1.2, ecolor=color, elinewidth=1, capsize=2, alpha=0.95, label=fam, ls="none")
        for _, r in d.sort_values("accuracy_mean", ascending=False).iterrows():
            if fam == "VQC":
                text = short_vqc(r.n_qubits, r.n_layers)
            elif r.input.startswith("570"):
                text = f"{fam.replace('Logistic regression', 'LogReg')} 570"
            else:
                text = f"{fam.replace('Logistic regression', 'LogReg').replace('GaussianNB', 'GNB')} PCA-{int(r.n_qubits)}"
            points.append((r.latency_ms_mean, r.accuracy_mean))
            texts.append(text)
    ax.set_xscale("log")
    ax.axhline(chance, color=MUTED, ls=":", lw=1)
    style(ax, "Test accuracy against inference latency",
          f"mean +/- sd over {n_seeds} recording-level splits; latency = median of {cfg['latency']['repeats']} single-window calls on the CPU "
          f"({cfg['compute']['torch_threads']} torch threads)", "inference latency per window (ms, log scale)", "test accuracy")
    ax.text(ax.get_xlim()[0], chance + 0.004, " chance", color=MUTED, fontsize=8, va="bottom")
    ax.legend(loc="center right")
    fig.tight_layout()
    annotate_without_overlap(fig, ax, points, texts)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_noise(noise, cfg, chance, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = list(cfg["vqc"]["n_layers"])
    qubits = sorted(noise.n_qubits.unique())
    ls_by_q = {q: ls for q, ls in zip(qubits, ["--", "-", "-."])}
    mk_by_q = {q: m for q, m in zip(qubits, ["s", "o", "^"])}
    ends = []
    for (n, L), d in noise.groupby(["n_qubits", "n_layers"]):
        g = d.groupby("noise_p").accuracy.mean()
        color = SERIES[depths.index(L) % len(SERIES)] if L in depths else INK2
        ax.plot(g.index, g.values, color=color, ls=ls_by_q.get(n, "-"), marker=mk_by_q.get(n, "o"), ms=7, mec=SURFACE, mew=1.2, lw=2)
        ends.append((g.index[-1], g.values[-1], short_vqc(n, L), color))
    ax.axhline(chance, color=MUTED, ls=":", lw=1)
    ax.text(0, chance + 0.002, "chance", color=MUTED, fontsize=8, va="bottom")
    handles = [Line2D([], [], color=SERIES[i], lw=2, label=f"{L} layer{'s' if L != 1 else ''}") for i, L in enumerate(depths)]
    handles += [Line2D([], [], color=INK2, ls=ls_by_q[q], marker=mk_by_q[q], ms=6, lw=1.5, label=f"{q} qubits") for q in qubits]
    ax.set_xticks(sorted(noise.noise_p.unique()))
    ax.set_xlim(left=-0.004, right=max(noise.noise_p) * 1.18)
    n_seeds = noise.seed.nunique()
    style(ax, "Trained circuits under depolarising noise", f"inference on default.mixed; depolarising channel on every qubit after the embedding and after each layer; "
          f"{n_seeds} seed{'s' if n_seeds != 1 else ''}", "depolarising probability p", "test accuracy")
    end_labels(ax, ends, x_pad_frac=0.012)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_all(out_dir, cfg, chance):
    out_dir = out_dir
    conv = pd.read_csv(out_dir / "convergence.csv")
    summary = pd.read_csv(out_dir / "benchmark_summary.csv")
    fig_convergence(conv, cfg, chance, out_dir / "fig_convergence.png")
    fig_accuracy_latency(summary, cfg, chance, out_dir / "fig_accuracy_latency.png")
    noise_path = out_dir / "noise_sweep.csv"
    if noise_path.exists():
        fig_noise(pd.read_csv(noise_path), cfg, chance, out_dir / "fig_noise.png")
