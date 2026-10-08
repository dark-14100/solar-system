import math
import os

import numpy as np
from scipy.optimize import brentq
from scipy.stats import linregress

import files
import simulate
from data import DAY, G, YEAR, build_system
from physics import orbital_period

MOON_REFERENCE = {"period_days": 27.3217, "perigee_km": 363300.0, "apogee_km": 405500.0}


def relative_position(result, name):
    """Where a body is, measured from its parent (for example the Moon from Earth)."""
    i = result["names"].index(name)
    p = result["names"].index(result["parents"][i])
    return result["pos"][:, i] - result["pos"][:, p]


def energy_drift(result):
    """How much the total energy changed since the start, as a fraction."""
    e = result["energy"]
    return (e - e[0]) / abs(e[0])


def measure_period(times, rel_pos):
    """Measure the orbital period (seconds) from the simulated path."""
    # [NumPy: arctan2 + unwrap] angle around the parent, without jumping back at 2*pi
    angle = np.unwrap(np.arctan2(rel_pos[:, 1], rel_pos[:, 0]))
    angle = angle - angle[0]
    if angle[-1] < 0:
        angle = -angle
    orbits = angle[-1] / (2 * math.pi)
    if orbits < 1.5:
        raise ValueError("the run covers only %.2f orbits, need 1.5; use more years" % orbits)
    n_orbits = int(orbits)
    # the time at which the body finishes orbit 0, 1, 2, ... then fit a straight line through them
    # [NumPy: interp] linear interpolation; [NumPy: polyfit] degree-1 fit, slope = period
    finish_times = np.interp(2 * math.pi * np.arange(n_orbits + 1), angle, times)
    return np.polyfit(np.arange(n_orbits + 1), finish_times, 1)[0]


def solve_kepler(M, e):
    """Solve Kepler's equation M = E - e*sin(E) for E with a root finder."""
    M = M % (2 * math.pi)
    if e == 0 or M == 0:
        return M
    # [SciPy: brentq] bracketed root finder; [Python: lambda] small anonymous function f(E)
    return brentq(lambda E: E - e * math.sin(E) - M, 0, 2 * math.pi, xtol=1e-14)


def kepler_position(a, e, t, mu, inc_deg):
    """Exact position of a planet on a two-body orbit at time t."""
    n = math.sqrt(mu / a**3)
    E = solve_kepler(n * t, e)
    x = a * (math.cos(E) - e)
    y = a * math.sqrt(1 - e**2) * math.sin(E)
    inc = math.radians(inc_deg)
    return np.array([x, y * math.cos(inc), y * math.sin(inc)])


def kepler_fit(result, mode):
    """Fit log(period) against log(orbit size). Kepler's third law says the slope is 1.5."""
    sun = result["parents"].index(None)
    names = []
    a_list = []
    t_list = []
    for i in range(len(result["names"])):
        if result["parents"][i] != result["names"][sun]:
            continue  # skip the Sun and the Moon
        if mode == "analytic":
            mu = G * (result["masses"][sun] + result["masses"][i])
            period = orbital_period(result["a"][i], mu)
        else:
            try:
                period = measure_period(result["times"], relative_position(result, result["names"][i]))
            except ValueError:
                continue  # the run is too short for this planet
        names.append(result["names"][i])
        a_list.append(result["a"][i])
        t_list.append(period)
    if len(names) < 3:
        raise ValueError("only %d planets usable for the fit, need 3; use more years" % len(names))
    fit = linregress(np.log(a_list), np.log(t_list))  # [SciPy: stats.linregress] slope, intercept, r
    return {"mode": mode, "names": names, "a": np.array(a_list), "T": np.array(t_list),
            "slope": fit.slope, "intercept": fit.intercept, "r2": fit.rvalue**2}


def moon_metrics(result):
    """Moon period, closest and farthest distance from Earth."""
    rel = relative_position(result, "Moon")
    dist = np.linalg.norm(rel, axis=1) / 1e3
    return {"period_days": measure_period(result["times"], rel) / DAY,
            "perigee_km": dist.min(), "apogee_km": dist.max()}


def two_body_check(bodies, name):
    """Simulate only the Sun and one planet for one orbit, and compare with the exact answer."""
    planet = next((body for body in bodies if body.name == name), None)  # [Python: generator expression]
    if planet is None or planet.parent != "Sun":
        raise ValueError(name + " must be a planet")
    sun = bodies[0]
    system = build_system([sun, planet])
    mu = G * (sun.mass + planet.mass)
    a = planet.semi_major_axis
    period = orbital_period(a, mu)
    result = simulate.run_leapfrog(system, period, period / 5000, 5000)
    simulated = result["pos"][-1, 1] - result["pos"][-1, 0]
    exact = kepler_position(a, planet.eccentricity, result["times"][-1], mu, planet.inclination_deg)
    error = np.linalg.norm(simulated - exact)
    return error, error / a


def compare(first, second):
    """Distance between the two runs for every body, at every saved time."""
    diff = np.linalg.norm(first["pos"] - second["pos"], axis=2)  # (frames, n, 3) -> (frames, n)
    return {"names": first["names"], "times": first["times"], "diff": diff,
            "max_diff": dict(zip(first["names"], diff.max(axis=0)))}  # [Python: dict(zip(...))]


def check_save_resume(bodies):
    """Save and reload must give identical numbers. Two half runs must match one long run."""
    half = 438 * 3600.0
    whole = 876 * 3600.0
    reference = simulate.run_leapfrog(build_system(bodies), whole, 3600.0, 24)
    first = simulate.run_leapfrog(build_system(bodies), half, 3600.0, 24)

    path = files.out_path("_check.npz")
    files.save_state(first, path)
    saved = files.load_state(path)
    os.remove(path)
    same = np.array_equal(saved["pos"], first["pos"])

    system = simulate.resume_system(build_system(bodies), saved)
    second = simulate.run_leapfrog(system, half, 3600.0, 24, saved["times"][-1])
    error = np.max(np.abs(second["pos"][-1] - reference["pos"][-1])) / np.max(np.abs(reference["pos"][-1]))
    return same, error


def period_check(result, id_, name, body, expected_days, tolerance, target):
    """One report row: is the measured period close enough to the known value?"""
    try:
        days = measure_period(result["times"], relative_position(result, body)) / DAY
    except ValueError:
        return (id_, name, "n/a (run too short)", target, None)
    ok = abs(days - expected_days) / expected_days < tolerance
    return (id_, name, "%.3f days" % days, target, bool(ok))


def run_checks(result, bodies):
    """The 8 checks. Returns rows of (id, name, measured, target, passed)."""
    rows = []

    drift = np.max(np.abs(energy_drift(result)))
    rows.append(("S1", "Energy drift", "%.2e" % drift, "below 1e-6", bool(drift < 1e-6)))

    rows.append(period_check(result, "S2", "Earth period", "Earth", 365.25, 0.001, "365.25 days +/- 0.1%"))
    rows.append(period_check(result, "S3", "Moon period", "Moon", 27.32, 0.005, "27.32 days +/- 0.5%"))

    in_first_year = result["times"] - result["times"][0] <= YEAR  # [NumPy: boolean mask]
    dist = np.linalg.norm(relative_position(result, "Moon")[in_first_year], axis=1) / 1e3
    ok = dist.min() >= 356000 and dist.max() <= 407000
    rows.append(("S4", "Moon distance, year 1", "%.0f-%.0f km" % (dist.min(), dist.max()), "356000-407000 km", bool(ok)))

    fit = kepler_fit(result, "analytic")
    rows.append(("S5", "Kepler slope", "%.4f" % fit["slope"], "1.5 +/- 0.01", bool(abs(fit["slope"] - 1.5) <= 0.01)))

    _, relative = two_body_check(bodies, "Earth")
    rows.append(("S6", "Two-body Earth error", "%.2e of orbit" % relative, "below 1e-4", bool(relative < 1e-4)))

    full_run = abs((result["times"][-1] - result["times"][0]) / YEAR - 10) < 0.01
    passed = bool(result["runtime"] < 60) if full_run else None  # [Python: conditional expression]
    rows.append(("S7", "Run time, 10 years", "%.2f s" % result["runtime"], "below 60 s", passed))

    same, error = check_save_resume(bodies)  # [Python: tuple unpacking]
    rows.append(("S8", "Save, reload, resume", "identical=%s, err %.0e" % (same, error), "identical, below 1e-9", bool(same and error < 1e-9)))
    return rows
