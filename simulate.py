import functools
import time

import numpy as np
from scipy.integrate import solve_ivp

from physics import acceleration, total_energy


def timer(func):
    """[Decorator] Times the function it wraps and stores the time in the result dict."""

    @functools.wraps(func)  # [Decorator: functools.wraps] keeps func's __name__ and docstring
    def wrapper(*args, **kwargs):  # [*args / **kwargs] accept any arguments and pass them on unchanged
        start = time.perf_counter()
        result = func(*args, **kwargs)  # [unpacking] spread the tuple/dict back into the call
        result["runtime"] = time.perf_counter() - start
        return result

    return wrapper  # [closure] wrapper remembers func after timer() has returned


def make_result(system, times, pos, vel, method, steps):
    """Put everything about a run into one dict, and add the energy at each saved time."""
    pos = np.array(pos)  # list of (n, 3) arrays -> one (frames, n, 3) array
    vel = np.array(vel)
    if not np.all(np.isfinite(pos)):  # [NumPy: np.isfinite + np.all] catch NaN/inf anywhere
        raise ValueError("the simulation blew up (NaN); try a smaller timestep")
    energy = [total_energy(p, v, system["masses"]) for p, v in zip(pos, vel)]  # [Python: zip]
    return {
        "names": system["names"],
        "parents": system["parents"],
        "colors": system["colors"],
        "masses": system["masses"],
        "a": system["a"],
        "times": np.array(times),
        "pos": pos,
        "vel": vel,
        "energy": np.array(energy),
        "method": method,
        "steps": steps,
    }


@timer  # [Decorator syntax] same as run_leapfrog = timer(run_leapfrog)
def run_leapfrog(system, duration, dt, save_every, t0=0.0):  # [default argument] t0
    """Our own loop: kick-drift-kick leapfrog with a fixed timestep."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and timestep must be above 0")
    masses = system["masses"]
    pos = system["pos"].copy()  # [NumPy: .copy()] do not change the caller's arrays
    vel = system["vel"].copy()
    acc = acceleration(pos, masses)

    n_steps = round(duration / dt)
    times = [t0]
    all_pos = [pos.copy()]
    all_vel = [vel.copy()]
    for step in range(1, n_steps + 1):
        vel = vel + 0.5 * dt * acc  # half kick
        pos = pos + dt * vel  # drift
        acc = acceleration(pos, masses)
        vel = vel + 0.5 * dt * acc  # half kick
        if step % save_every == 0 or step == n_steps:
            times.append(t0 + step * dt)
            all_pos.append(pos.copy())
            all_vel.append(vel.copy())
    return make_result(system, times, all_pos, all_vel, "leapfrog", n_steps)


def derivative(t, y, masses):
    """What solve_ivp needs: how fast the state changes. y = all positions, then all velocities."""
    n = len(masses)
    pos = y[:3 * n].reshape(n, 3)  # [NumPy: slicing + reshape] flat vector -> (n, 3)
    vel = y[3 * n:]
    return np.concatenate([vel, acceleration(pos, masses).flatten()])  # [NumPy: concatenate/flatten]


@timer
def run_ivp(system, duration, dt, save_every, t0=0.0):
    """SciPy's solver picks its own step size. Used as a second opinion."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and timestep must be above 0")
    masses = system["masses"]
    n = len(masses)

    n_steps = round(duration / dt)
    save_steps = list(range(0, n_steps + 1, save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    times = [s * dt for s in save_steps]

    y0 = np.concatenate([system["pos"].flatten(), system["vel"].flatten()])
    # [SciPy: solve_ivp] adaptive 8th-order Runge-Kutta (DOP853); args= passes extra arguments to derivative
    sol = solve_ivp(derivative, (0, times[-1]), y0, method="DOP853", t_eval=times,
                    args=(masses,), rtol=1e-10, atol=1e-6)
    if not sol.success:
        raise ValueError("solve_ivp failed: " + sol.message)
    pos = sol.y[:3 * n].T.reshape(-1, n, 3)  # [NumPy: transpose + reshape with -1] (frames, n, 3)
    vel = sol.y[3 * n:].T.reshape(-1, n, 3)
    return make_result(system, [t0 + t for t in times], pos, vel, "ivp", sol.nfev)


def run(system, method, duration, dt, save_every, t0=0.0):
    if method == "leapfrog":
        return run_leapfrog(system, duration, dt, save_every, t0)
    elif method == "ivp":
        return run_ivp(system, duration, dt, save_every, t0)
    else:
        raise ValueError("method must be 'leapfrog' or 'ivp'")


def resume_system(system, saved):
    """Start the system from the last saved moment of an earlier run."""
    if saved["names"] != system["names"]:
        raise ValueError("the saved run has different bodies than the data file")
    system["pos"] = saved["pos"][-1].copy()  # [NumPy: negative index] last frame
    system["vel"] = saved["vel"][-1].copy()
    return system
