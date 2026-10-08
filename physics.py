import numpy as np

from data import G


def acceleration(pos, masses):
    """Pull of gravity on every body, from every other body."""
    acc = np.zeros_like(pos)  # [NumPy: zeros_like] same shape and dtype as pos
    for i in range(len(masses)):
        diff = pos - pos[i]  # [NumPy: broadcasting] (n, 3) - (3,): arrow from body i to every body
        dist = np.sqrt(np.sum(diff**2, axis=1))  # [NumPy: vectorised reduction along an axis]
        dist[i] = 1.0  # body i itself: avoid dividing by zero
        weight = masses / dist**3  # [NumPy: element-wise maths, no Python loop]
        weight[i] = 0.0  # a body does not pull on itself
        acc[i] = G * np.dot(weight, diff)  # [NumPy: np.dot] (n,) . (n, 3) -> (3,)
    return acc


def total_energy(pos, vel, masses):
    """Kinetic plus potential energy. It should stay almost constant."""
    kinetic = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    potential = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):  # each pair once
            dist = np.linalg.norm(pos[i] - pos[j])  # [NumPy: np.linalg.norm] vector length
            potential -= G * masses[i] * masses[j] / dist
    return kinetic + potential


def orbital_period(a, mu):
    """Kepler's third law: time for one orbit. Works on a float or a NumPy array."""
    return 2 * np.pi * np.sqrt(a**3 / mu)
