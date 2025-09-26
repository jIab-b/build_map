from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAPS_DIR = BASE_DIR / "maps"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _valid_map_id(map_id: str) -> bool:
    return isinstance(map_id, str) and len(map_id) == 4 and map_id.isalpha() and map_id.islower()


def _find_latest_tile_path(map_id: str, z: int, x: int, y: int) -> Optional[Path]:
    tiles_dir = MAPS_DIR / map_id / "tiles" / str(z) / str(x)
    if not tiles_dir.exists():
        return None
    best_seq = -1
    best_path: Optional[Path] = None
    for p in tiles_dir.glob(f"*_{y}.png"):
        stem = p.stem
        try:
            seq_part = stem.split("_", 1)[0]
            seq_val = int(seq_part)
        except Exception:
            seq_val = -1
        if seq_val > best_seq:
            best_seq = seq_val
            best_path = p
    return best_path


@app.get("/maps/{map_id}/tiles/{z}/{x}/{y}.png")
def get_map_tile(map_id: str, z: int, x: int, y: int):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    p = _find_latest_tile_path(map_id, z, x, y)
    if p and p.exists():
        return FileResponse(str(p))
    raise HTTPException(status_code=404, detail="not found")
