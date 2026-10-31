# Progress

Resume rule: on restart, read this file first and continue from "Next step".

## Current week: 1 (repo setup, data audit), with the Week 2 InfoDengue pull already running

## Done
- Verified sources (2026-10-30): InfoDengue alertcity API (CSV, about 1 s per call), the three OpenDengue V1.3 zips, the IBGE geoftp 2022 municipal mesh.
- Not reachable: the IBGE REST API (servicodados.ibge.gov.br) times out from this machine, so the geoftp mesh is used instead.
- OpenDengue India coverage: state by year, 1997 to 2024 (no 2013 to 2014), no district or monthly data. Details are in docs/DATA_SOURCES.md.
- Environment: .venv (Python 3.11.9). Direct dependencies pinned in pyproject.toml, full lock in requirements.lock.
- Config (configs/default.yaml), utils (config, cache with manifest, logging, seed), CLI (`python -m dengue.cli <stage>`).
- Brazil boundaries: 5,570 municipalities (`python -m dengue.cli boundaries`).
- InfoDengue parser and tests: 6 tests pass (`pytest`).

## Running
- `python -m dengue.cli cases` (InfoDengue pull for 5,570 municipalities, 2014 to 2025), log in logs/cases_pull.log.

## Open problems
- Earth Engine is not authenticated on this machine. Needed from Week 3.
- No `make` on Windows. The Makefile wraps the Python CLI, which also runs directly.
- Network calls must run outside the tool sandbox (DNS is blocked inside it).

## Next step
- Finish the Week 1 docs, commit, then build the monthly label table once the pull finishes.
