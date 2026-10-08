import json
import math
import os

import numpy as np
from scipy import constants

G = constants.G
AU = constants.astronomical_unit
DAY = 86400.0
YEAR = 365.25 * DAY

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "planets.json")


class Body:
    """One object in the solar system: Sun, planet or moon."""

    kind = "body"

    def __init__(self, info):
        self.name = info["name"]
        self.parent = info["parent"]  # name of the parent body, None for the Sun
        self.mass = info["mass_kg"]
        self.color = info.get("color", "#cccccc")
        self.a = info.get("semi_major_axis_m", np.nan)  # orbit size
        self.e = info.get("eccentricity", 0)  # how stretched the orbit is
        self.inc = info.get("inclination_deg", 0)  # tilt
        self.w = info.get("arg_periapsis_deg", 0)  # where the closest point is

    def describe(self):
        return self.name


class Star(Body):
    kind = "star"

    def describe(self):
        return self.name + " (star)"


class Planet(Body):
    kind = "planet"

    def describe(self):
        return "%s (planet, orbit %.2f AU)" % (self.name, self.a / AU)


class Moon(Body):
    kind = "moon"

    def describe(self):
        return "%s (moon of %s, orbit %.0f km)" % (self.name, self.parent, self.a / 1e3)


BODY_CLASSES = {"star": Star, "planet": Planet, "moon": Moon}


def load_records():
    """Read planets.json and return the list of body dicts (checked)."""
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError("data file not found: " + DATA_FILE)
    check_bodies(data["bodies"])
    return data["bodies"]


def load_bodies():
    """Return the bodies as Star, Planet and Moon objects."""
    records = load_records()
    return [BODY_CLASSES[r["type"]](r) for r in records]


def check_bodies(bodies):
    """Raise ValueError with a clear message if the data is wrong."""
    seen = []
    for b in bodies:
        for key in ["name", "type", "parent", "mass_kg", "radius_m"]:
            if key not in b:
                raise ValueError("a body is missing the field " + key)
        name = b["name"]
        if b["type"] not in BODY_CLASSES:
            raise ValueError(name + ": type must be star, planet or moon")
        if name in seen:
            raise ValueError("duplicate body name " + name)
        if b["mass_kg"] <= 0:
            raise ValueError(name + ": mass must be above 0")
        if b["parent"] is not None:
            if b["parent"] not in seen:
                raise ValueError(name + ": parent must be listed before it")
            for key in ["semi_major_axis_m", "eccentricity", "inclination_deg"]:
                if key not in b:
                    raise ValueError(name + ": missing the field " + key)
            if not 0 <= b["eccentricity"] < 1:
                raise ValueError(name + ": eccentricity must be from 0 up to 1")
        seen.append(name)
    parents = [b["parent"] for b in bodies]
    if parents.count(None) != 1:
        raise ValueError("exactly one body (the Sun) must have no parent")


def build_system(bodies):
    """Give every body a start position and speed. Returns a dict."""
    names = [b.name for b in bodies]
    masses = np.array([b.mass for b in bodies])
    pos = np.zeros((len(bodies), 3))
    vel = np.zeros((len(bodies), 3))

    for i in range(len(bodies)):
        b = bodies[i]
        if b.parent is None:
            continue
        p = names.index(b.parent)
        inc = math.radians(b.inc)
        w = math.radians(b.w)

        # start at the closest point to the parent, moving at the right speed there
        mu = G * (masses[p] + masses[i])
        r = b.a * (1 - b.e)
        v = math.sqrt(mu * (1 + b.e) / r)
        rel_pos = r * np.array([math.cos(w), math.sin(w), 0])
        rel_vel = v * np.array([-math.sin(w) * math.cos(inc), math.cos(w) * math.cos(inc), math.sin(inc)])
        pos[i] = pos[p] + rel_pos
        vel[i] = vel[p] + rel_vel

        if b.kind == "moon":
            # Earth and Moon circle their shared centre, so move both back by the Moon's share
            share = masses[i] / (masses[p] + masses[i])
            pos[p] = pos[p] - share * rel_pos
            pos[i] = pos[i] - share * rel_pos
            vel[p] = vel[p] - share * rel_vel
            vel[i] = vel[i] - share * rel_vel

    # shift everything so the total momentum is zero (the Sun does not drift away)
    pos = pos - np.dot(masses, pos) / masses.sum()
    vel = vel - np.dot(masses, vel) / masses.sum()

    return {
        "names": names,
        "parents": [b.parent for b in bodies],
        "colors": [b.color for b in bodies],
        "masses": masses,
        "a": np.array([b.a for b in bodies]),
        "pos": pos,
        "vel": vel,
    }


def print_tree(bodies):
    """Print the bodies, indented by how many parents each one has."""
    names = [b.name for b in bodies]
    for b in bodies:
        depth = 0
        parent = b.parent
        while parent is not None:
            depth += 1
            parent = bodies[names.index(parent)].parent
        print("    " * depth + b.describe())  # each class describes itself differently
