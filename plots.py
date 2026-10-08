import math

import matplotlib

SHOW = False  # set to True to open a window for each figure as well as saving it
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import files
from analysis import relative_position
from data import AU, DAY, YEAR

plt.style.use("dark_background")


def finish(fig, name):
    fig.savefig(files.out_path(name), dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print("saved outputs/" + name)


def draw_orbits(ax, result, indices):
    for i in indices:
        x = result["pos"][:, i, 0] / AU
        y = result["pos"][:, i, 1] / AU
        size = max(2, 2 + 0.8 * (math.log10(result["masses"][i]) - 22))  # bigger mass, bigger dot
        ax.plot(x, y, color=result["colors"][i], linewidth=0.8)
        ax.plot(x[-1], y[-1], "o", color=result["colors"][i], markersize=size, label=result["names"][i])
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.legend(fontsize=7, loc="upper right")


def plot_orbits(result):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_orbits(ax, result, range(len(result["names"])))
    ax.set_title("Orbits")
    finish(fig, "orbits_2d.png")


def plot_inner_outer(result):
    """Two panels, because Neptune's orbit would make the inner planets a dot."""
    sun = result["parents"].index(None)
    inner = []
    outer = []
    for i in range(len(result["names"])):
        if i == sun:
            inner.append(i)
        elif result["parents"][i] == result["names"][sun]:
            if result["a"][i] > 2.5 * AU:
                outer.append(i)
            else:
                inner.append(i)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7))
    draw_orbits(left, result, inner)
    left.set_title("Inner planets")
    draw_orbits(right, result, outer)
    right.set_title("Outer planets")
    finish(fig, "orbits_inner_outer.png")


def plot_moon(result):
    """The Moon's path around Earth during the first year."""
    rel = relative_position(result, "Moon") / 1e3
    first_year = result["times"] - result["times"][0] <= 365.25 * DAY
    rel = rel[first_year]
    dist = np.sqrt(np.sum(rel**2, axis=1))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(rel[:, 0], rel[:, 1], color="#d9d9d9", linewidth=0.7)
    ax.plot(0, 0, "o", color="#4f9dde", markersize=8, label="Earth")
    ax.annotate("closest\n%.0f km" % dist.min(), rel[dist.argmin(), :2], xytext=(10, 10), textcoords="offset points")
    ax.annotate("farthest\n%.0f km" % dist.max(), rel[dist.argmax(), :2], xytext=(10, 10), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("x relative to Earth (km)")
    ax.set_ylabel("y relative to Earth (km)")
    ax.set_title("The Moon around Earth")
    ax.legend()
    finish(fig, "moon_relative.png")


def plot_energy(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        drift = np.abs((r["energy"] - r["energy"][0]) / abs(r["energy"][0]))
        ax.semilogy((r["times"] - r["times"][0]) / YEAR, np.maximum(drift, 1e-18), label=r["method"])
    ax.set_xlabel("time (years)")
    ax.set_ylabel("relative energy change")
    ax.set_title("Energy stays almost constant")
    ax.legend()
    finish(fig, "energy_drift.png")


def plot_kepler(fit):
    a = fit["a"] / AU
    T = fit["T"] / YEAR
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(a, T, "o", label="planets")
    line = np.linspace(a.min() * 0.8, a.max() * 1.2, 50)
    ax.loglog(line, np.exp(fit["intercept"]) * (line * AU) ** fit["slope"] / YEAR,
              label="fit: slope %.4f" % fit["slope"])
    for k in range(len(a)):
        ax.annotate(fit["names"][k], (a[k], T[k]), xytext=(5, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("orbit size (AU)")
    ax.set_ylabel("period (years)")
    ax.set_title("Kepler's third law")
    ax.legend()
    finish(fig, "kepler_third_law.png")


def plot_compare(comp, first, second):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    methods = [first["method"], second["method"]]
    ax1.bar(methods, [first["runtime"], second["runtime"]])
    ax1.set_title("runtime (s)")
    ax2.bar(methods, [first["steps"], second["steps"]])
    ax2.set_title("steps")
    years = (comp["times"] - comp["times"][0]) / YEAR
    for i in range(len(comp["names"])):
        ax3.semilogy(years, np.maximum(comp["diff"][:, i] / 1e3, 1e-9), label=comp["names"][i])
    ax3.set_xlabel("time (years)")
    ax3.set_ylabel("difference (km)")
    ax3.set_title("leapfrog vs solve_ivp")
    ax3.legend(fontsize=7, ncol=2)
    finish(fig, "integrator_comparison.png")
