import os
import sys

import numpy as np

import analysis
import data
import files
import simulate
from data import DAY, YEAR
from physics import acceleration

bodies = data.load_bodies()
records = data.load_records()


def fresh():
    return data.build_system(bodies)


def run(years, method="leapfrog"):
    return simulate.run(fresh(), method, years * YEAR, 3600.0, 24)


def test_load_bodies():
    assert len(bodies) == 10
    assert bodies[4].name == "Moon" and bodies[4].parent == "Earth"


def test_classes_describe_themselves():
    assert isinstance(bodies[0], data.Star) and isinstance(bodies[4], data.Moon)
    assert "star" in bodies[0].describe() and "moon" in bodies[4].describe()
    assert bodies[0].describe() != bodies[3].describe()


def test_total_momentum_is_zero():
    s = fresh()
    momentum = np.linalg.norm(np.dot(s["masses"], s["vel"]))
    assert momentum / (s["masses"].sum() * 3e4) < 1e-12


def test_newtons_third_law():
    s = fresh()
    force = s["masses"][:, None] * acceleration(s["pos"], s["masses"])
    assert np.linalg.norm(force.sum(axis=0)) / np.abs(force).sum() < 1e-12


def test_energy_and_moon_one_year():
    r = run(1.0)
    assert np.max(np.abs(analysis.energy_drift(r))) < 1e-6
    dist = np.linalg.norm(analysis.relative_position(r, "Moon"), axis=1) / 1e3
    assert dist.min() > 356000 and dist.max() < 407000


def test_two_methods_agree():
    diff = analysis.compare(run(1.0), run(1.0, "ivp"))["max_diff"]
    assert diff["Earth"] < 1e6 and diff["Moon"] < 5e6


def test_solve_kepler():
    for e in [0, 0.0167, 0.2, 0.9, 0.99]:
        for M in np.linspace(0, 6.2, 12):
            E = analysis.solve_kepler(M, e)
            assert abs(E - e * np.sin(E) - M) < 1e-12


def test_two_body_earth():
    _, relative = analysis.two_body_check(bodies, "Earth")
    assert relative < 1e-4


def test_kepler_slope():
    assert abs(analysis.kepler_fit(run(2.0), "analytic")["slope"] - 1.5) < 0.01
    assert abs(analysis.kepler_fit(run(30.0, "ivp"), "measured")["slope"] - 1.5) < 0.02


def test_periods():
    r = run(2.0)
    earth = analysis.measure_period(r["times"], analysis.relative_position(r, "Earth")) / DAY
    moon = analysis.measure_period(r["times"], analysis.relative_position(r, "Moon")) / DAY
    assert abs(earth - 365.25) / 365.25 < 0.001
    assert abs(moon - 27.32) / 27.32 < 0.005


def test_short_run_raises_error():
    r = run(0.5)
    try:
        analysis.measure_period(r["times"], analysis.relative_position(r, "Earth"))
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_save_and_load():
    r = run(0.2)
    path = files.out_path("_test.npz")
    files.save_state(r, path)
    back = files.load_state(path)
    os.remove(path)
    assert np.array_equal(back["pos"], r["pos"]) and back["names"] == r["names"]


def test_csv_rows():
    r = run(0.2)
    path = files.out_path("_test.csv")
    rows = files.save_csv(r, path)
    with open(path) as f:
        lines = f.readlines()
    os.remove(path)
    assert len(lines) == rows + 1


def test_save_and_resume():
    same, error = analysis.check_save_resume(bodies)
    assert same and error < 1e-9


def test_bad_data_is_rejected():
    bad_versions = []
    for change in range(6):
        copy = [dict(r) for r in records]
        if change == 0:
            del copy[3]["mass_kg"]
        elif change == 1:
            copy[3]["mass_kg"] = -1
        elif change == 2:
            copy[3]["eccentricity"] = 1.5
        elif change == 3:
            copy.append(dict(copy[3]))
        elif change == 4:
            copy[3]["parent"] = "Nope"
        else:
            copy[0]["parent"] = "Earth"
        bad_versions.append(copy)
    for copy in bad_versions:
        try:
            data.check_bodies(copy)
        except ValueError:
            continue
        raise AssertionError("bad data was accepted")


def test_bad_settings_are_rejected():
    for duration, dt in [(0, 3600), (YEAR, -1)]:
        try:
            simulate.run_leapfrog(fresh(), duration, dt, 24)
        except ValueError:
            continue
        raise AssertionError("bad settings were accepted")


tests = [test_load_bodies, test_classes_describe_themselves, test_total_momentum_is_zero, test_newtons_third_law,
         test_energy_and_moon_one_year, test_two_methods_agree, test_solve_kepler,
         test_two_body_earth, test_kepler_slope, test_periods, test_short_run_raises_error,
         test_save_and_load, test_csv_rows, test_save_and_resume, test_bad_data_is_rejected,
         test_bad_settings_are_rejected]

failed = 0
for test in tests:
    try:
        test()
        print("PASS", test.__name__)
    except AssertionError as e:
        failed = failed + 1
        print("FAIL", test.__name__, e)
print("%d of %d passed" % (len(tests) - failed, len(tests)))
sys.exit(1 if failed else 0)
