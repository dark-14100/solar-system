import numpy as np

from data import G


def acceleration(pos, masses):
    """Pull of gravity on every body, from every other body."""
    acc = np.zeros_like(pos)
    for i in range(len(masses)):
        diff = pos - pos[i]  # arrow from body i to every body
        dist = np.sqrt(np.sum(diff**2, axis=1))
        dist[i] = 1.0  # body i itself: avoid dividing by zero
        weight = masses / dist**3
        weight[i] = 0.0  # a body does not pull on itself
        acc[i] = G * np.dot(weight, diff)
    return acc


def total_energy(pos, vel, masses):
    """Kinetic plus potential energy. It should stay almost constant."""
    kinetic = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    potential = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            dist = np.linalg.norm(pos[i] - pos[j])
            potential = potential - G * masses[i] * masses[j] / dist
    return kinetic + potential


def orbital_period(a, mu):
    """Kepler's third law: time for one orbit."""
    return 2 * np.pi * np.sqrt(a**3 / mu)
