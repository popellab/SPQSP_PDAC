"""Full CAF diagnostic: steady-state analysis + spatial organization stability."""
import sys, pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

audit_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("outputs/class_audit_42.csv")
phi_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("outputs/phi_42.csv")
out_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else pathlib.Path(".")

df = pd.read_csv(audit_path)
phi = pd.read_csv(phi_path)

df["day"] = df["step"] * 0.25
phi["day"] = phi["step"] * 0.25

# ── Panel layout: 2×3 ──
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# ── Panel A: All cell types vs QSP target ──
ax_a = fig.add_subplot(gs[0, 0])
types = [
    ("CD8 T",   "abm.CD8",    "tgt.CD8",    "#e41a1c"),
    ("MAC",     "abm.MAC",    "tgt.MAC",    "#377eb8"),
    ("B cell",  "abm.B",      "tgt.B",      "#ff7f00"),
    ("iCAF",    "abm.iCAF",   "tgt.iCAF",   "#FF9800"),
    ("myCAF",   "abm.myCAF",  "tgt.myCAF",  "#E91E63"),
    ("apCAF",   "abm.apCAF",  "tgt.apCAF",  "#9C27B0"),
    ("MDSC",    "abm.MDSC",   "tgt.MDSC",   "#4daf4a"),
]
for label, abm_col, tgt_col, color in types:
    if abm_col in df.columns and tgt_col in df.columns:
        ratio = df[abm_col] / df[tgt_col].clip(lower=1)
        ax_a.plot(df["day"], ratio, color=color, lw=1.2, label=label, alpha=0.85)
ax_a.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax_a.set_ylabel("ABM / QSP target")
ax_a.set_xlabel("Day")
ax_a.set_title("A. ABM fill fraction (ABM/QSP)")
ax_a.legend(fontsize=6, ncol=2, loc="upper right")
ax_a.set_ylim(0, 6.5)
ax_a.grid(True, alpha=0.2)

# ── Panel B: CAF substate absolute counts ──
ax_b = fig.add_subplot(gs[0, 1])
substates = [
    ("qPSC",  "abm.qPSC",  "#2196F3"),
    ("iCAF",  "abm.iCAF",  "#FF9800"),
    ("myCAF", "abm.myCAF", "#E91E63"),
    ("apCAF", "abm.apCAF", "#9C27B0"),
]
bottom = np.zeros(len(df))
for name, col, color in substates:
    vals = df[col].values
    ax_b.fill_between(df["day"].values, bottom, bottom + vals, alpha=0.7,
                       color=color, label=name)
    bottom += vals
total_tgt = df["tgt.iCAF"] + df["tgt.myCAF"] + df["tgt.apCAF"] + df["tgt.qPSC"]
ax_b.plot(df["day"], total_tgt, "k--", lw=1.5, alpha=0.4, label="QSP target")
ax_b.set_ylabel("Cell count")
ax_b.set_xlabel("Day")
ax_b.set_title("B. Fibroblast sub-states (ABM)")
ax_b.legend(fontsize=6, loc="upper right")
ax_b.grid(True, alpha=0.2)

# ── Panel C: Steady-state theory ──
ax_c = fig.add_subplot(gs[0, 2])
alphas = np.logspace(-4, -0.5, 200)
k_iCAF_death, k_myCAF_death, k_apCAF_death = 0.05, 0.015, 0.03
n_icaf, n_mycaf, n_apcaf = 1372, 2329, 415
n_fib = n_icaf + n_mycaf + n_apcaf
d_avg = (k_iCAF_death * n_icaf + k_myCAF_death * n_mycaf + k_apCAF_death * n_apcaf) / n_fib * 0.25
T = 42803
N_ss = alphas * T / (alphas + d_avg)
fill_pct = N_ss / T * 100

ax_c.semilogx(alphas, fill_pct, "k-", lw=2)
ax_c.axvline(0.01, color="#E91E63", ls="--", lw=1.5, label=r"$\alpha$=0.01 (current)")
ax_c.axhline(58.8, color="#E91E63", ls=":", lw=1, alpha=0.5)
ax_c.axvline(d_avg, color="#377eb8", ls="--", lw=1.5, label=f"d_avg={d_avg:.4f}")
ax_c.axhline(50, color="#377eb8", ls=":", lw=1, alpha=0.5)
ax_c.set_xlabel(r"Gap-closing rate $\alpha$")
ax_c.set_ylabel("Steady-state fill (%)")
ax_c.set_title(r"C. $N_{ss} = \alpha T / (\alpha + d)$")
ax_c.legend(fontsize=7)
ax_c.set_ylim(0, 105)
ax_c.grid(True, alpha=0.2)
ax_c.annotate("58.8%", xy=(0.01, 58.8), xytext=(0.003, 45),
              fontsize=8, color="#E91E63",
              arrowprops=dict(arrowstyle="->", color="#E91E63", lw=0.8))

# ── Panel D: phi_icaf_c1 trajectory ──
ax_d = fig.add_subplot(gs[1, 0])
valid_icaf = phi[phi["valid_icaf_c1"] == 1]
ax_d.plot(valid_icaf["day"], valid_icaf["phi_icaf_c1"], color="#FF9800", lw=1.2, alpha=0.8)
ax_d.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5, label="random (1.0)")
last200 = valid_icaf.tail(200)
mean_late = last200["phi_icaf_c1"].mean()
std_late = last200["phi_icaf_c1"].std()
ax_d.axhline(mean_late, color="#FF9800", ls="--", lw=1, alpha=0.6,
             label=f"late mean={mean_late:.3f}")
ax_d.fill_between([valid_icaf["day"].min(), valid_icaf["day"].max()],
                   mean_late - 2*std_late, mean_late + 2*std_late,
                   alpha=0.1, color="#FF9800")
ax_d.set_ylabel(r"$\phi_{iCAF \times C1}$")
ax_d.set_xlabel("Day")
ax_d.set_title(r"D. iCAF$\times$cancer spatial organization")
ax_d.legend(fontsize=7)
ax_d.grid(True, alpha=0.2)
ax_d.set_ylim(0.8, 1.5)
ax_d.annotate("Random recruitment\nperiod", xy=(25, 1.08), fontsize=7, color="gray",
              ha="center", style="italic")

# ── Panel E: phi stability (CV by window) ──
ax_e = fig.add_subplot(gs[1, 1])
phi_cols = [
    ("CD8×C1",    "phi_cd8_c1",    "valid_cd8_c1",    "#e41a1c"),
    ("M2×C1",     "phi_m2_c1",     "valid_m2_c1",     "#377eb8"),
    ("iCAF×C1",   "phi_icaf_c1",   "valid_icaf_c1",   "#FF9800"),
    ("MDSC×CD8",  "phi_mdsc_cd8",  "valid_mdsc_cd8",  "#4daf4a"),
]
max_step = int(phi["step"].max())
if max_step > 500:
    windows = [(0, 200), (200, 500), (500, 900), (900, 1300), (1300, max_step+1)]
else:
    n5 = max_step // 5
    windows = [(i*n5, (i+1)*n5) for i in range(5)]
    windows[-1] = (windows[-1][0], max_step+1)
window_labels = [f"{s0*0.25:.0f}-{s1*0.25:.0f}" for s0, s1 in windows]
x_pos = np.arange(len(windows))
width = 0.18
for i, (label, col, vcol, color) in enumerate(phi_cols):
    cvs = []
    for s0, s1 in windows:
        w = phi[(phi["step"] >= s0) & (phi["step"] < s1) & (phi[vcol] == 1)]
        if len(w) > 5:
            cv = 100 * w[col].std() / abs(w[col].mean()) if w[col].mean() != 0 else 0
        else:
            cv = 0
        cvs.append(cv)
    ax_e.bar(x_pos + i * width, cvs, width, color=color, alpha=0.8, label=label)

ax_e.set_xticks(x_pos + 1.5 * width)
ax_e.set_xticklabels(window_labels, fontsize=7)
ax_e.set_ylabel("CV (%)")
ax_e.set_xlabel("Day window")
ax_e.set_title("E. Spatial metric stability (CV)")
ax_e.legend(fontsize=6, loc="upper right")
ax_e.grid(True, alpha=0.2, axis="y")

# ── Panel F: Volume budget ──
ax_f = fig.add_subplot(gs[1, 2])
last = df.iloc[-1]

# Get cancer count from phi CSV (n_c1 column)
last_phi = phi.iloc[-1]
n_cancer = last_phi.get("n_c1", 107534)

categories = ["Cancer", "CD8", "TCD4", "B", "MAC", "MDSC", "iCAF", "myCAF", "apCAF", "Free"]
counts = [n_cancer, last["abm.CD8"], last["abm.TCD4"], last["abm.B"],
          last["abm.MAC"], last["abm.MDSC"],
          last["abm.iCAF"], last["abm.myCAF"], last["abm.apCAF"]]
volumes_um3 = [2145, 185, 185, 185, 2145, 185, 2145, 2145, 2145]

vox_vol = 20**3  # 8000 um3
nx, ny, nz = 100, 100, 12
total_vol = nx * ny * nz * vox_vol

vol_used = [c * v / total_vol * 100 for c, v in zip(counts, volumes_um3)]
labels_used = list(categories[:-1])
colors_bar = ["#333", "#e41a1c", "#984ea3", "#ff7f00", "#377eb8", "#4daf4a",
              "#FF9800", "#E91E63", "#9C27B0", "#ddd"]

free = 100 - sum(vol_used)
vol_used.append(free)
labels_used.append("Free")

bars = ax_f.barh(range(len(labels_used)), vol_used, color=colors_bar, alpha=0.85)
ax_f.set_yticks(range(len(labels_used)))
ax_f.set_yticklabels(labels_used, fontsize=8)
ax_f.set_xlabel("Volume fraction (%)")
ax_f.set_title("F. Domain volume budget")
ax_f.grid(True, alpha=0.2, axis="x")
for bar, v in zip(bars, vol_used):
    if v > 2:
        ax_f.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                  f"{v:.1f}%", va="center", fontsize=7)

fig.suptitle("CAF Gap-Closing Diagnostic — Direct Recruitment (α=0.01)", fontsize=14, y=0.98)

out_path = out_dir / "caf_diagnostic_full.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
