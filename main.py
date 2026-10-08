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
    moon = analysis.moon_metrics(result)
    for key, reference in analysis.MOON_REFERENCE.items():
        print("moon %s: %.2f (reference %.2f)" % (key, moon[key], reference))
    plots.plot_moon(result)


def analyze_two_body(bodies):
    for body in bodies:
        if isinstance(body, data.Planet):  # [OOP: isinstance] ask the object what class it is
            error, relative = analysis.two_body_check(bodies, body.name)
            print("two-body %-8s error %.0f km (%.1e of its orbit)" % (body.name, error / 1e3, relative))


def analyze_compare(years):
    bodies = data.load_bodies()
    first = simulate.run(data.build_system(bodies), "leapfrog", years * YEAR, DT, SAVE_EVERY)
    second = simulate.run(data.build_system(bodies), "ivp", years * YEAR, DT, SAVE_EVERY)
    comparison = analysis.compare(first, second)
    for result in [first, second]:
        drift = analysis.energy_drift(result)[-1]
        print("%-9s %.2f s, %d steps, final energy change %.1e"
              % (result["method"], result["runtime"], result["steps"], drift))
    max_diff = comparison["max_diff"]
    worst = max(max_diff, key=max_diff.get)  # [Python: max with key=] body with the biggest difference
    print("largest difference between the two: %s, %.0f km" % (worst, max_diff[worst] / 1e3))
    plots.plot_energy([first, second])
    plots.plot_compare(comparison, first, second)


def do_analyze(years):
    result = files.load_state()
    bodies = data.load_bodies()
    # [Python: functions are first-class objects] keep (function, argument) pairs in a list
    steps = [(analyze_kepler, result), (analyze_moon, result), (analyze_two_body, bodies), (analyze_compare, years)]
    for step, argument in steps:
        try:
            step(argument)
        except ValueError as e:  # one failed analysis does not stop the others
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


NEEDS_YEARS = {"simulate", "simulate-ivp", "resume", "analyze", "all"}  # [Python: set] fast membership test


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
        for key, (label, _) in MENU.items():  # [Python: nested tuple unpacking]
            print("  " + key + ") " + label)
        print("  q) Quit")
        try:
            choice = input("> ").strip()
        except EOFError:
            return
        if choice == "q":
            return
        if choice in MENU:
            command = MENU[choice][1]
            years = ask_years() if command in NEEDS_YEARS else 10.0
            run_command(command, years)
        else:
            print("choose a number from the menu, or q")


def main():
    if len(sys.argv) == 1:
        menu()
        return
    years = 10.0
    if len(sys.argv) > 2:
        try:
            years = float(sys.argv[2])
        except ValueError:
            print("years must be a number")
            sys.exit(1)
    ok = run_command(sys.argv[1], years)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":  # [Python: entry-point guard] runs only when started directly, not on import
    main()
