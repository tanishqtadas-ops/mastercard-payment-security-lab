import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from demo.run_demo import DemoConfig, DemoRunner, serialize_deterministic_demo_json

app = FastAPI(title="Mastercard Payment Security Lab Prototype")

_cached_demo_json: str | None = None

def get_demo_json() -> str:
    global _cached_demo_json
    if _cached_demo_json is None:
        config = DemoConfig(
            seed=42,
            family1_rounds=4,
            family2_rounds=2,
            family3_rounds=2,
            retrain_interval=2,
            quiet=True,
        )
        runner = DemoRunner(config=config)
        result = runner.run()
        _cached_demo_json = serialize_deterministic_demo_json(result)
    return _cached_demo_json

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")

@app.get("/api/demo")
async def get_demo():
    # Retrieve the precomputed deterministic JSON and return as raw json
    # Since serialize_deterministic_demo_json already returns a JSON string,
    # we need to return it as a custom response to avoid double-encoding.
    from fastapi.responses import Response
    return Response(content=get_demo_json(), media_type="application/json")
