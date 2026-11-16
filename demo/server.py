"""CountGuard demo server: serves the precomputed privacy-utility frontier. Runs locally and offline.

    pip install -e ".[demo]"
    python -m demo.server            # then open http://localhost:8765

The frontier JSON files in demo/data/ come from benchmarks/run_frontier.py
(synthetic kidney-transplant cohorts, seed 2, n=285 and n=20,000).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STATIC = HERE / "static"
COHORTS = (285, 20000)
NO_CACHE = {"Cache-Control": "no-cache"}  # browsers must revalidate, so an edited page never runs stale JS

app = FastAPI(title="CountGuard demo")


class _NoCacheStatic(StaticFiles):
    def file_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        r = super().file_response(*args, **kwargs)
        r.headers.update(NO_CACHE)
        return r


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/api/frontier")
def frontier(n: int = 285) -> JSONResponse:
    p = DATA / f"frontier_n{n}.json"
    if n not in COHORTS or not p.exists():
        raise HTTPException(404, "frontier not computed")
    return JSONResponse(json.loads(p.read_text()))


app.mount("/static", _NoCacheStatic(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    print("CountGuard demo: http://localhost:8765", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
