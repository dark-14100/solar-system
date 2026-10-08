# Solar System Simulator

Simulates the Sun, the 8 planets and the Moon under Newton's gravity, then checks the result against Kepler's third law and energy conservation.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python main.py              # menu
.venv/bin/python main.py all 10       # simulate 10 years, analyze, plot, export, report
.venv/bin/python main.py simulate 5   # or one step at a time: tree, simulate, simulate-ivp,
.venv/bin/python main.py analyze      # resume, analyze, plot, export, report
.venv/bin/python test_all.py          # 16 tests
```

Results go to the `outputs/` folder: `state.npz`, `trajectory.csv`, `report.txt` and PNG plots.

## Files

| File | Job |
| --- | --- |
| `data/planets.json` | mass, orbit size and tilt of every body |
| `data.py` | `Body`, `Star`, `Planet`, `Moon` classes; load and check the JSON; give each body a start position and speed |
| `physics.py` | gravity, energy, Kepler's third law |
| `simulate.py` | our leapfrog loop, SciPy's `solve_ivp`, and the `@timer` decorator |
| `analysis.py` | periods, Kepler fit, Kepler's equation (with a `lambda`), Moon numbers, the 8 checks |
| `files.py` | save/load a run, write CSV and report |
| `plots.py` | all the figures |
| `main.py` | menu and commands |
| `test_all.py` | tests |

## Notes

- All planets start at their closest point to the Sun on the +x axis, not at their real position on a calendar date.
- The Moon's closest point is rotated 30 degrees (`arg_periapsis_deg`). At 0 degrees the Sun's pull makes its period 0.8% off.
- Earth and the Moon are shifted so their shared centre follows Earth's orbit. Without this Earth's year is off by 0.15%.
- Timestep is 1 hour. The Moon needs steps under about 4,700 s.
