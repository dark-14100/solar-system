import sys

import analysis
import data
import files
import plots
import simulate
from data import YEAR

DT = 3600.0  # timestep: 1 hour
SAVE_EVERY = 24  # keep one frame per day


def do_tree():
    data.print_tree(data.load_bodies())


def do_simulate(years, method):
    system = data.build_system(data.load_bodies())
    result = simulate.run(system, method, years * YEAR, DT, SAVE_EVERY)
    files.save_state(result)
    print("simulated %g years with %s: %d steps in %.2f s" % (years, method, result["steps"], result["runtime"]))


def do_resume(years):
    saved = files.load_state()
    system = simulate.resume_system(data.build_system(data.load_bodies()), saved)
    result = simulate.run(system, saved["method"], years * YEAR, DT, SAVE_EVERY, saved["times"][-1])
    files.save_state(result, files.out_path("state_resumed.npz"))
    print("continued for %g more years, saved outputs/state_resumed.npz" % years)


def analyze_kepler(result):
    fit = None
    for mode in ["analytic", "measured"]:
        try:
            fit_mode = analysis.kepler_fit(result, mode)
            print("kepler %s: slope %.4f, %d planets" % (mode, fit_mode["slope"], len(fit_mode["names"])))
            if fit is None:
                fit = fit_mode
        except ValueError as e:
            print("kepler %s skipped: %s" % (mode, e))
    if fit is not None:
        plots.plot_kepler(fit)


def analyze_moon(result):
    m = analysis.moon_metrics(result)
    for key in ["period_days", "perigee_km", "apogee_km"]:
        print("moon %s: %.2f (reference %.2f)" % (key, m[key], analysis.MOON_REFERENCE[key]))
    plots.plot_moon(result)


def analyze_two_body(bodies):
    for b in bodies:
        if b.parent == "Sun":
            error, relative = analysis.two_body_check(bodies, b.name)
            print("two-body %-8s error %.0f km (%.1e of its orbit)" % (b.name, error / 1e3, relative))


def analyze_compare(years):
    bodies = data.load_bodies()
    first = simulate.run(data.build_system(bodies), "leapfrog", years * YEAR, DT, SAVE_EVERY)
    second = simulate.run(data.build_system(bodies), "ivp", years * YEAR, DT, SAVE_EVERY)
    comp = analysis.compare(first, second)
    for r in [first, second]:
        drift = analysis.energy_drift(r)[-1]
        print("%-9s %.2f s, %d steps, final energy change %.1e" % (r["method"], r["runtime"], r["steps"], drift))
    worst = max(comp["names"], key=lambda name: comp["max_diff"][name])  # body with the biggest difference
    print("largest difference between the two: %s, %.0f km" % (worst, comp["max_diff"][worst] / 1e3))
    plots.plot_energy([first, second])
    plots.plot_compare(comp, first, second)


def do_analyze(years):
    result = files.load_state()
    bodies = data.load_bodies()
    try:
        analyze_kepler(result)
    except ValueError as e:
        print("skipped:", e)
    try:
        analyze_moon(result)
    except ValueError as e:
        print("skipped:", e)
    try:
        analyze_two_body(bodies)
    except ValueError as e:
        print("skipped:", e)
    try:
        analyze_compare(years)
    except ValueError as e:
        print("skipped:", e)


def do_plot():
    result = files.load_state()
    plots.plot_orbits(result)
    plots.plot_inner_outer(result)
    plots.plot_moon(result)
    plots.plot_energy([result])


def do_export():
    rows = files.save_csv(files.load_state(), files.out_path("trajectory.csv"))
    print("saved outputs/trajectory.csv (%d rows)" % rows)


def do_report():
    result = files.load_state()
    rows = analysis.run_checks(result, data.load_bodies())
    files.write_report(rows, files.out_path("report.txt"))
    for id_, name, measured, target, passed in rows:
        print("%s %-24s %-28s %-24s %s" % (id_, name, measured, target, files.verdict(passed)))
    print("saved outputs/report.txt")


def run_command(command, years):
    """Run one command. Returns False if something went wrong."""
    try:
        if command == "tree":
            do_tree()
        elif command == "simulate":
            do_simulate(years, "leapfrog")
        elif command == "simulate-ivp":
            do_simulate(years, "ivp")
        elif command == "resume":
            do_resume(years)
        elif command == "analyze":
            do_analyze(years)
        elif command == "plot":
            do_plot()
        elif command == "export":
            do_export()
        elif command == "report":
            do_report()
        elif command == "all":
            do_simulate(years, "leapfrog")
            do_analyze(years)
            do_plot()
            do_export()
            do_report()
        else:
            print("unknown command:", command)
            return False
    except ValueError as e:
        print("error:", e)
        return False
    return True


MENU = {
    "1": ("Show bodies", "tree"),
    "2": ("Simulate (leapfrog)", "simulate"),
    "3": ("Simulate (solve_ivp)", "simulate-ivp"),
    "4": ("Continue the last run", "resume"),
    "5": ("Analyze", "analyze"),
    "6": ("Plot", "plot"),
    "7": ("Export CSV", "export"),
    "8": ("Report (checks S1-S8)", "report"),
    "9": ("Run everything", "all"),
}


def ask_years():
    text = input("years [10]: ")
    if text == "":
        return 10.0
    try:
        return float(text)
    except ValueError:
        print("not a number, using 10")
        return 10.0


def menu():
    while True:
        print("\nSolar System Simulator")
        for key in MENU:
            print("  " + key + ") " + MENU[key][0])
        print("  q) Quit")
        try:
            choice = input("> ").strip()
        except EOFError:
            return
        if choice == "q":
            return
        if choice in MENU:
            years = 10.0
            if MENU[choice][1] in ["simulate", "simulate-ivp", "resume", "analyze", "all"]:
                years = ask_years()
            run_command(MENU[choice][1], years)
        else:
            print("choose a number from the menu, or q")


if len(sys.argv) == 1:
    menu()
else:
    years = 10.0
    if len(sys.argv) > 2:
        try:
            years = float(sys.argv[2])
        except ValueError:
            print("years must be a number")
            sys.exit(1)
    ok = run_command(sys.argv[1], years)
    sys.exit(0 if ok else 1)
