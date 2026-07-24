"""
markov_chain_pretty_graphs.py
==============================
A polished, reusable Markov-chain visualization toolkit.

Given any valid transition matrix P (rows summing to 1), this produces a
single, publication-quality figure with four panels:

  1. Transition-matrix heatmap
  2. Directed network diagram (edge width + color ~ probability,
     node size + color ~ stationary probability)
  3. Convergence plot: distribution vs. time step, starting from several
     different initial conditions, all converging to the same stationary line
  4. Stationary distribution bar chart

Works in Google Colab or any local Python environment with
numpy / matplotlib / networkx installed.

Usage
-----
    python markov_chain_pretty_graphs.py

or, in a notebook:

    from markov_chain_pretty_graphs import visualize_markov_chain
    visualize_markov_chain(P, labels=["Sunny", "Cloudy", "Rainy"])
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
def _apply_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.dpi": 110,
    })


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def stationary_distribution(P):
    """Eigenvector method (assumes a regular/ergodic chain with a unique
    stationary distribution)."""
    evals, evecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(evals - 1))
    vec = np.real(evecs[:, idx])
    return vec / vec.sum()


def evolve(x0, P, n_steps):
    """Return the trajectory of a distribution x0 under repeated application
    of P, shape (n_steps + 1, n_states)."""
    traj = [x0]
    x = x0.copy()
    for _ in range(n_steps):
        x = x @ P
        traj.append(x)
    return np.array(traj)


# ---------------------------------------------------------------------------
# Panel 1: heatmap
# ---------------------------------------------------------------------------
def _plot_heatmap(ax, P, labels):
    n = len(labels)
    im = ax.imshow(P, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    ax.set_title("Transition matrix  P")
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            color = "white" if P[i, j] > 0.55 else "black"
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probability")


# ---------------------------------------------------------------------------
# Panel 2: network diagram
# ---------------------------------------------------------------------------
def _plot_network(ax, P, pi, labels):
    n = len(labels)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-9:
                G.add_edge(i, j, weight=P[i, j])

    pos = nx.circular_layout(G)

    node_sizes = 1200 + 3800 * np.asarray(pi)
    node_colors = plt.get_cmap("plasma")(0.15 + 0.75 * np.asarray(pi) / max(pi.max(), 1e-9))

    self_loops = [(u, v) for u, v in G.edges() if u == v]
    normal_edges = [(u, v) for u, v in G.edges() if u != v]

    for (u, v) in normal_edges:
        w = G[u][v]["weight"]
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            connectionstyle="arc3,rad=0.18",
            width=0.6 + 4.0 * w, arrowsize=14,
            edge_color=[plt.get_cmap("Greys")(0.35 + 0.55 * w)],
            alpha=0.9,
        )
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=self_loops,
        connectionstyle="arc3,rad=0.7", arrowsize=14,
        width=1.5, edge_color="#999999",
    )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                            node_color=node_colors, edgecolors="black", linewidths=1.6)
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={i: f"{labels[i]}\n\u03c0={pi[i]:.2f}" for i in range(n)},
        font_size=9, font_weight="bold")

    edge_labels = {(u, v): f"{P[u, v]:.2f}" for u, v in G.edges() if u != v}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=7.5)

    ax.set_title("Transition diagram\n(node size/color = stationary prob., edge width = P(i\u2192j))")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel 3: convergence plot
# ---------------------------------------------------------------------------
def _plot_convergence(ax, P, pi, labels, n_steps=25):
    n = len(labels)
    cmap = plt.get_cmap("tab10")

    # A handful of informative starting points: each corner (certain of one
    # state) plus the uniform distribution.
    starts = [np.eye(n)[i] for i in range(n)] + [np.ones(n) / n]
    start_names = [f"start: certain {labels[i]}" for i in range(n)] + ["start: uniform"]

    for k, (x0, name) in enumerate(zip(starts, start_names)):
        traj = evolve(x0, P, n_steps)
        # plot distance from stationary distribution (L1) -- a clean single
        # line per start, regardless of how many states there are
        dist = np.abs(traj - pi).sum(axis=1)
        ax.plot(range(n_steps + 1), dist, marker=".", label=name,
                color=cmap(k % 10), linewidth=1.6)

    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel("distance to stationary dist. (L1, log scale)")
    ax.set_title("Convergence to the stationary distribution")
    ax.legend(fontsize=7.5, loc="upper right")


# ---------------------------------------------------------------------------
# Panel 4: stationary distribution bar chart
# ---------------------------------------------------------------------------
def _plot_stationary_bar(ax, pi, labels):
    colors = plt.get_cmap("plasma")(0.15 + 0.75 * np.asarray(pi) / max(pi.max(), 1e-9))
    bars = ax.bar(labels, pi, color=colors, edgecolor="black", linewidth=1.2)
    for rect, p in zip(bars, pi):
        ax.text(rect.get_x() + rect.get_width() / 2, p + 0.01, f"{p:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(pi) * 1.25)
    ax.set_ylabel("stationary probability")
    ax.set_title("Long-run stationary distribution \u03c0")
    ax.tick_params(axis="x", rotation=20)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def visualize_markov_chain(P, labels=None, n_steps=25, title=None, savepath=None):
    """Build the full 4-panel figure for a transition matrix P.

    Parameters
    ----------
    P : (n, n) array_like
        Transition matrix. Rows must sum to 1.
    labels : list of str, optional
        State names.
    n_steps : int
        Number of steps to show in the convergence panel.
    title : str, optional
        Overall figure title.
    savepath : str, optional
        If given, saves the figure (e.g. "chain.png").
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if labels is None:
        labels = [f"S{i}" for i in range(n)]
    assert P.shape == (n, n), "P must be square"
    assert np.allclose(P.sum(axis=1), 1, atol=1e-6), "each row of P must sum to 1"

    _apply_style()
    pi = stationary_distribution(P)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _plot_heatmap(axes[0, 0], P, labels)
    _plot_network(axes[0, 1], P, pi, labels)
    _plot_convergence(axes[1, 0], P, pi, labels, n_steps=n_steps)
    _plot_stationary_bar(axes[1, 1], pi, labels)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {savepath}")

    return fig, pi


# ---------------------------------------------------------------------------
# Demo (runs if this file is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Same weather chain used in markov_chain_new_example.ipynb
    P_weather = np.array([
        [0.70, 0.25, 0.05],
        [0.30, 0.40, 0.30],
        [0.15, 0.35, 0.50],
    ])
    fig, pi = visualize_markov_chain(
        P_weather,
        labels=["Sunny", "Cloudy", "Rainy"],
        title="Weather Markov Chain",
        savepath="weather_chain.png",
    )
    print("Stationary distribution:", np.round(pi, 3))
    plt.show()
