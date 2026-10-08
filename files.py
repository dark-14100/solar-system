import csv
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
STATE_FILE = os.path.join(OUT_DIR, "state.npz")


def out_path(name):
    """Path of a file inside the outputs folder (the folder is created if needed)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def save_state(result, path=STATE_FILE):
    """Save a whole run to one .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    parents = []
    for p in result["parents"]:
        parents.append(p if p is not None else "")
    try:
        np.savez(path, names=np.array(result["names"]), parents=np.array(parents),
                 colors=np.array(result["colors"]), masses=result["masses"], a=result["a"],
                 times=result["times"], pos=result["pos"], vel=result["vel"],
                 energy=result["energy"], method=np.array(result["method"]),
                 runtime=np.array(result["runtime"]), steps=np.array(result["steps"]))
    except OSError as e:
        raise ValueError("cannot write " + path + ": " + str(e))


def load_state(path=STATE_FILE):
    """Load a run saved by save_state."""
    try:
        data = np.load(path)
        result = {key: data[key] for key in data.files}
    except FileNotFoundError:
        raise ValueError("no saved run found; simulate first")
    except OSError as e:
        raise ValueError("cannot read " + path + ": " + str(e))
    result["names"] = [str(x) for x in result["names"]]
    result["parents"] = [str(p) if str(p) != "" else None for p in result["parents"]]
    result["colors"] = [str(c) for c in result["colors"]]
    result["method"] = str(result["method"])
    result["runtime"] = float(result["runtime"])
    result["steps"] = int(result["steps"])
    return result


def save_csv(result, path):
    """Write one row per body per saved time. Returns the number of rows."""
    rows = 0
    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "body", "x_m", "y_m", "z_m", "vx_ms", "vy_ms", "vz_ms"])
            for k in range(len(result["times"])):
                for i in range(len(result["names"])):
                    row = [result["times"][k], result["names"][i]]
                    row = row + list(result["pos"][k, i]) + list(result["vel"][k, i])
                    writer.writerow(row)
                    rows = rows + 1
    except OSError as e:
        raise ValueError("cannot write " + path + ": " + str(e))
    return rows


def verdict(passed):
    if passed is True:
        return "PASS"
    elif passed is False:
        return "FAIL"
    return "n/a"


def write_report(rows, path):
    """rows: list of (id, name, measured, target, passed)."""
    try:
        with open(path, "w") as f:
            for id_, name, measured, target, passed in rows:
                f.write(f"{id_:<4}{name:<30}{measured:<30}{target:<22}{verdict(passed)}\n")
    except OSError as e:
        raise ValueError("cannot write " + path + ": " + str(e))
