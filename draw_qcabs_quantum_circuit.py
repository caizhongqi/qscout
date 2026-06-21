"""Draw the publication circuit diagram for the implemented Q-CABS VQC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).parent
FIG = ROOT / "figures"


PALETTE = {
    "encode": "#2f6b5f",
    "train": "#6b5c9b",
    "entangle": "#b56c3d",
    "noise": "#b54b4b",
    "readout": "#3b6f9c",
    "ink": "#252525",
}


def box(ax, x, y, w, h, label, color, dashed=False, fontsize=8.5):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=color, facecolor="white",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=PALETTE["ink"])


def cnot(ax, x, y_control, y_target):
    ax.plot([x, x], [y_control, y_target], color=PALETTE["entangle"], lw=1.45)
    ax.plot(x, y_control, "o", color=PALETTE["entangle"], ms=5)
    radius = 0.12
    circle = plt.Circle((x, y_target), radius, edgecolor=PALETTE["entangle"], facecolor="white", lw=1.35)
    ax.add_patch(circle)
    ax.plot([x - radius * 0.65, x + radius * 0.65], [y_target, y_target], color=PALETTE["entangle"], lw=1.0)
    ax.plot([x, x], [y_target - radius * 0.65, y_target + radius * 0.65], color=PALETTE["entangle"], lw=1.0)


def draw_member(ax):
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.8, 4.4)
    ax.axis("off")
    ys = [3.5, 2.55, 1.6, 0.65]
    for i, y in enumerate(ys):
        ax.text(0.12, y, rf"$q_{i}: |0\rangle$", ha="left", va="center", fontsize=10)
        ax.plot([1.35, 12.7], [y, y], color=PALETTE["ink"], lw=0.95)

    # Initial data encoding.
    for i, y in enumerate(ys):
        box(ax, 1.55, y - 0.23, 1.05, 0.46, rf"$D^0(z_{i})$", PALETTE["encode"])
        box(ax, 2.68, y - 0.19, 0.30, 0.38, r"$\mathcal{N}$", PALETTE["noise"], fontsize=7)
    ax.text(2.07, 4.04, "initial feature encoding", ha="center", fontsize=8.5, color=PALETTE["encode"])
    ax.text(2.83, -0.22, "noise", ha="center", fontsize=7.5, color=PALETTE["noise"])

    # Repeat block.
    ax.add_patch(Rectangle((3.05, 0.18), 7.0, 3.85, fill=False, edgecolor="#777777", linestyle="--", lw=1.15))
    ax.text(6.5, 4.12, r"repeat for $l=1,ldots,L$", ha="center", fontsize=10, fontweight="bold")
    for i, y in enumerate(ys):
        box(ax, 3.18, y - 0.23, 0.96, 0.46, rf"$D^l(z)$", PALETTE["encode"])
        box(ax, 4.58, y - 0.23, 1.30, 0.46, r"$R_XR_YR_Z$", PALETTE["train"], fontsize=8)
        box(ax, 8.45, y - 0.23, 0.90, 0.46, r"$\mathcal{N}_p$", PALETTE["noise"])
    ax.text(3.66, -0.22, "re-upload", ha="center", fontsize=8, color=PALETTE["encode"])
    ax.text(5.23, -0.22, r"trainable $\theta_{l,i}$", ha="center", fontsize=8, color=PALETTE["train"])
    cnot(ax, 6.55, ys[0], ys[1])
    cnot(ax, 7.08, ys[1], ys[2])
    cnot(ax, 7.61, ys[2], ys[3])
    # Circular closing edge is drawn on the far side to keep the circuit readable.
    ax.plot([8.0, 8.0], [ys[3], 0.27], color=PALETTE["entangle"], lw=1.1)
    ax.plot([8.0, 6.18], [0.27, 0.27], color=PALETTE["entangle"], lw=1.1)
    ax.plot([6.18, 6.18], [0.27, ys[0]], color=PALETTE["entangle"], lw=1.1)
    ax.plot(6.18, ys[0], "o", color=PALETTE["entangle"], ms=5)
    ax.add_patch(plt.Circle((8.0, ys[3]), 0.12, edgecolor=PALETTE["entangle"], facecolor="white", lw=1.2))
    ax.text(7.05, -0.55, "circular CNOT entanglement", ha="center", fontsize=8, color=PALETTE["entangle"])

    # Readout.
    for i, y in enumerate(ys):
        ax.plot(10.8, y, marker="o", markersize=7, markerfacecolor="white", markeredgecolor=PALETTE["readout"], markeredgewidth=1.3)
        ax.text(11.06, y + 0.16, rf"$\langle Z_{i}\rangle$", fontsize=8.5, color=PALETTE["readout"])
    ax.text(11.65, 0.0, r"plus $\langle Z_iZ_{i+1}\rangle$", ha="center", fontsize=8.5, color=PALETTE["readout"])
    ax.text(6.6, -0.78, r"$D^l(z_i)=R_Y(s_lz_{(i+ln)\mathrm{mod}\ d})R_Z(s_lz_{(i+ln)\mathrm{mod}\ d}^2/\pi)$", ha="center", fontsize=8.5)
    ax.text(0.12, 4.28, "(b) Gate-level circuit of one Q-CABS committee member", fontsize=11, fontweight="bold")


def draw_committee(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.15, 5.65, "(a) Q-CABS quantum committee for query ranking", fontsize=11, fontweight="bold")
    box(ax, 0.35, 2.35, 1.45, 0.85, "public candidate\nfeature $z=P_q(x)$", PALETTE["encode"], fontsize=9)
    member_y = [4.35, 2.65, 0.95]
    for m, y in enumerate(member_y, start=1):
        box(ax, 3.0, y - 0.36, 2.15, 0.72, rf"VQC member {m}\n$U_m(z;\theta_m)$", PALETTE["train"], fontsize=9)
        ax.annotate("", xy=(2.94, y), xytext=(1.82, 2.78), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": PALETTE["ink"]})
        ax.annotate("", xy=(6.14, y), xytext=(5.18, y), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": PALETTE["ink"]})
        box(ax, 6.15, y - 0.28, 1.18, 0.56, rf"$p_{m}(y|z)$", PALETTE["readout"], fontsize=9)
    box(ax, 7.78, 2.05, 1.78, 1.48, "query score\nuncertainty\n+ disagreement\n+ coverage - risk", PALETTE["entangle"], fontsize=8.5)
    for y in member_y:
        ax.annotate("", xy=(7.72, 2.79), xytext=(7.35, y), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PALETTE["ink"]})
    ax.annotate("", xy=(9.88, 2.79), xytext=(9.58, 2.79), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": PALETTE["ink"]})
    ax.text(9.83, 3.22, "top-$k$\nclassical\nqueries", ha="left", va="center", fontsize=8.8)
    ax.text(5.02, 0.18, "All quantum processing is local to the attacker; the target API receives only classical inputs.", ha="center", fontsize=8.5, color="#555555")


def main() -> None:
    fig = plt.figure(figsize=(14.0, 7.4))
    left = fig.add_axes([0.04, 0.10, 0.38, 0.82])
    right = fig.add_axes([0.45, 0.10, 0.52, 0.82])
    draw_committee(left)
    draw_member(right)
    fig.text(0.5, 0.02, "Q-CABS combines M noisy variational circuits for ranking, then queries a classical hard-label victim with the selected inputs.", ha="center", fontsize=10)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "qcabs_quantum_circuit.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "qcabs_quantum_circuit.svg", bbox_inches="tight")
    fig.savefig(FIG / "qcabs_quantum_circuit.pdf", bbox_inches="tight")
    plt.close(fig)
    print(FIG / "qcabs_quantum_circuit.png")


if __name__ == "__main__":
    main()
