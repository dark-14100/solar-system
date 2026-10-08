import json
import math
import os

import numpy as np
from scipy import constants

G = constants.G  # [SciPy: scipy.constants] physical constants, no magic numbers
AU = constants.astronomical_unit
DAY = 86400.0
YEAR = 365.25 * DAY

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "planets.json")


class Body:
    """One object in the solar system: Sun, planet or moon."""

    kind = "body"  # [OOP: class attribute] shared by every instance, overridden by subclasses

    def __init__(self, info):  # [OOP: constructor] builds the instance's own attributes
        self.name = info["name"]
        self.parent = info["parent"]  # name of the parent body, None for the Sun
        self.mass = info["mass_kg"]
        self.color = info.get("color", "#cccccc")
        self.semi_major_axis = info.get("semi_major_axis_m", np.nan)  # orbit size
        self.eccentricity = info.get("eccentricity", 0)  # how stretched the orbit is
        self.inclination_deg = info.get("inclination_deg", 0)  # tilt
        self.arg_periapsis_deg = info.get("arg_periapsis_deg", 0)  # where the closest point is

    def describe(self):
        return self.name


# [OOP: inheritance] Star, Planet and Moon reuse Body.__init__ and only change what differs
class Star(Body):
    kind = "star"

    def describe(self):  # [OOP: method overriding / polymorphism]
        return self.name + " (star)"


class Planet(Body):
    kind = "planet"

    def describe(self):
        return "%s (planet, orbit %.2f AU)" % (self.name, self.semi_major_axis / AU)


class Moon(Body):
    kind = "moon"

    def describe(self):
        return "%s (moon of %s, orbit %.0f km)" % (self.name, self.parent, self.semi_major_axis / 1e3)


# [Python: classes are objects] a dict maps the JSON "type" string straight to a class
BODY_CLASSES = {"star": Star, "planet": Planet, "moon": Moon}


def load_records():
    """Read planets.json and return the list of body dicts (checked)."""
    try:
        with open(DATA_FILE) as f:  # [Python: context manager] file closes itself
            data = json.load(f)
    except FileNotFoundError:  # [Python: exceptions] turn a low-level error into a clear one
        raise ValueError("data file not found: " + DATA_FILE)
    check_bodies(data["bodies"])
    return data["bodies"]


def load_bodies():
    """Return the bodies as Star, Planet and Moon objects."""
    records = load_records()
    return [BODY_CLASSES[record["type"]](record) for record in records]  # [Python: list comprehension]


def check_bodies(records):
    """Raise ValueError with a clear message if the data is wrong."""
    seen = []
    for record in records:
        for key in ["name", "type", "parent", "mass_kg", "radius_m"]:
            if key not in record:
                raise ValueError("a body is missing the field " + key)
        name = record["name"]
        if record["type"] not in BODY_CLASSES:
            raise ValueError(name + ": type must be star, planet or moon")
        if name in seen:
            raise ValueError("duplicate body name " + name)
        if record["mass_kg"] <= 0:
            raise ValueError(name + ": mass must be above 0")
        if record["parent"] is not None:
            if record["parent"] not in seen:
                raise ValueError(name + ": parent must be listed before it")
            for key in ["semi_major_axis_m", "eccentricity", "inclination_deg"]:
                if key not in record:
                    raise ValueError(name + ": missing the field " + key)
            if not 0 <= record["eccentricity"] < 1:  # [Python: chained comparison]
                raise ValueError(name + ": eccentricity must be from 0 up to 1")
        seen.append(name)
    parents = [record["parent"] for record in records]
    if parents.count(None) != 1:
        raise ValueError("exactly one body (the Sun) must have no parent")


def build_system(bodies):
    """Give every body a start position and speed. Returns a dict."""
    names = [body.name for body in bodies]
    masses = np.array([body.mass for body in bodies])
    pos = np.zeros((len(bodies), 3))  # [NumPy: 2D array] one row (x, y, z) per body
    vel = np.zeros((len(bodies), 3))

    for i, body in enumerate(bodies):  # [Python: enumerate] index and item together
        if body.parent is None:
            continue
        p = names.index(body.parent)
        inc = math.radians(body.inclination_deg)
        w = math.radians(body.arg_periapsis_deg)

        # start at the closest point to the parent, moving at the right speed there
        mu = G * (masses[p] + masses[i])
        r = body.semi_major_axis * (1 - body.eccentricity)
        speed = math.sqrt(mu * (1 + body.eccentricity) / r)
        rel_pos = r * np.array([math.cos(w), math.sin(w), 0])
        rel_vel = speed * np.array([-math.sin(w) * math.cos(inc), math.cos(w) * math.cos(inc), math.sin(inc)])
        pos[i] = pos[p] + rel_pos
        vel[i] = vel[p] + rel_vel

        if body.kind == "moon":
            # Earth and Moon circle their shared centre, so move both back by the Moon's share
            share = masses[i] / (masses[p] + masses[i])
            pos[p] -= share * rel_pos
            pos[i] -= share * rel_pos
            vel[p] -= share * rel_vel
            vel[i] -= share * rel_vel

    # shift everything so the total momentum is zero (the Sun does not drift away)
    # [NumPy: np.dot + broadcasting] mass-weighted mean (shape (3,)) subtracted from every row
    pos = pos - np.dot(masses, pos) / masses.sum()
    vel = vel - np.dot(masses, vel) / masses.sum()

    return {
        "names": names,
        "parents": [body.parent for body in bodies],
        "colors": [body.color for body in bodies],
        "masses": masses,
        "a": np.array([body.semi_major_axis for body in bodies]),
        "pos": pos,
        "vel": vel,
    }


def print_tree(bodies):
    """Print the bodies, indented by how many parents each one has."""
    names = [body.name for body in bodies]
    for body in bodies:
        depth = 0
        parent = body.parent
        while parent is not None:
            depth += 1
            parent = bodies[names.index(parent)].parent
        print("    " * depth + body.describe())  # [OOP: polymorphism] each class describes itself differently
