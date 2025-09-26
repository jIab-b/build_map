import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAPS_DIR = BASE_DIR / "maps"
TILE_SIZE = 64
COLLAGE_TILES = 3
COLLAGE_SIZE = TILE_SIZE * COLLAGE_TILES

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


class GenerateIn(BaseModel):
    x: int
    y: int
    z: int
    prompt: str


class MapCreateIn(BaseModel):
    map_id: str


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


@app.get("/maps")
def list_maps():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    ids = []
    for d in MAPS_DIR.iterdir():
        if d.is_dir() and _valid_map_id(d.name):
            ids.append(d.name)
    ids.sort()
    return {"maps": ids}


@app.post("/maps")
def create_map(inp: MapCreateIn):
    mid = (inp.map_id or "").strip().lower()
    if not _valid_map_id(mid):
        raise HTTPException(status_code=400, detail="invalid map_id")
    _ensure_map(mid)
    return {"ok": True, "map_id": mid}


def _map_paths(map_id: str) -> Tuple[Path, Path, Path, Path, Path]:
    base = MAPS_DIR / map_id
    tiles = base / "tiles"
    up = base / ".up"
    down = base / ".down"
    seq = base / "seq.txt"
    return base, tiles, up, down, seq


def _ensure_map(map_id: str) -> None:
    base, tiles, up, down, seq = _map_paths(map_id)
    tiles.mkdir(parents=True, exist_ok=True)
    up.mkdir(parents=True, exist_ok=True)
    down.mkdir(parents=True, exist_ok=True)
    if not seq.exists():
        seq.write_text("0", encoding="utf-8")


def _next_seq(map_id: str) -> int:
    _, _, _, _, seq = _map_paths(map_id)
    try:
        current = int(seq.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        current = 0
    nxt = current + 1
    seq.write_text(str(nxt), encoding="utf-8")
    return nxt


def _make_matte(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cell = max(4, size // 8)
    for j in range(0, size, cell):
        for i in range(0, size, cell):
            c = 220 if ((i // cell) + (j // cell)) % 2 == 0 else 200
            d.rectangle([i, j, i + cell, j + cell], fill=(c, c, c, 120))
    return img


def _build_collage_image(map_id: str, z: int, x: int, y: int) -> tuple[Image.Image, list[tuple[int, int, bool]]]:
    canvas = Image.new("RGBA", (COLLAGE_SIZE, COLLAGE_SIZE), (0, 0, 0, 0))
    matte = _make_matte(TILE_SIZE)
    info: list[tuple[int, int, bool]] = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            px = (dx + 1) * TILE_SIZE
            py = (dy + 1) * TILE_SIZE
            pos = (px, py)
            if dx == 0 and dy == 0:
                canvas.paste(matte, pos)
                info.append((dx, dy, False))
                continue
            p = _find_latest_tile_path(map_id, z, x + dx, y + dy)
            exists = bool(p and p.exists())
            info.append((dx, dy, exists))
            if not exists:
                canvas.paste(matte, pos)
                continue
            with Image.open(p) as im:
                tile_img = im.convert("RGBA")
                if tile_img.size != (TILE_SIZE, TILE_SIZE):
                    tile_img = tile_img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                canvas.paste(tile_img, pos)
    return canvas, info


async def _upload_public_image(buf: bytes) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=60) as up:
            files = {"file": ("grid.png", buf, "image/png")}
            print("[upload] starting", flush=True)
            resp = await up.post("https://tmpfiles.org/api/v1/upload", files=files)
            if 200 <= resp.status_code < 300:
                try:
                    payload = resp.json()
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    data = payload.get("data") or {}
                    txt = data.get("url") if isinstance(data, dict) else None
                else:
                    txt = (await resp.aread()).decode(errors="ignore").strip()
                if isinstance(txt, str) and txt.startswith("http"):
                    print(f"[upload] ok url={txt}", flush=True)
                    return txt
            print(f"[upload] failed status={resp.status_code}", flush=True)
            return None
    except Exception:
        print("[upload] exception", flush=True)
        return None


def _parse_resp_json(obj: dict) -> Optional[bytes | str]:
    if not isinstance(obj, dict):
        return None
    if obj.get("image") and isinstance(obj["image"], dict) and obj["image"].get("data"):
        return base64.b64decode(obj["image"]["data"])
    if obj.get("images") and isinstance(obj["images"], list) and obj["images"]:
        first = obj["images"][0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
    if obj.get("output") and isinstance(obj["output"], list) and obj["output"]:
        first = obj["output"][0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
    if obj.get("image") and isinstance(obj["image"], dict) and obj["image"].get("url"):
        return obj["image"]["url"]
    return None


async def fal_generate_edit(png_grid: bytes, prompt: str) -> Optional[bytes]:
    api_key = os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY")
    if not api_key:
        print("[fal] missing api key", flush=True)
        return None
    url = "https://fal.run/fal-ai/nano-banana/edit"
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    public_url = await _upload_public_image(png_grid)
    attempts: list[tuple[str, dict]] = []
    if public_url:
        attempts.extend(
            [
                (
                    "shape_urls",
                    {
                        "prompt": prompt,
                        "image_urls": [public_url],
                        "image_size": {"width": COLLAGE_SIZE, "height": COLLAGE_SIZE},
                    },
                ),
                (
                    "shape_url_single",
                    {
                        "prompt": prompt,
                        "image_url": public_url,
                        "image_size": {"width": COLLAGE_SIZE, "height": COLLAGE_SIZE},
                    },
                ),
            ]
        )
    else:
        print("[fal] skipping url attempt (no public url)", flush=True)

    b64 = base64.b64encode(png_grid).decode()
    attempts.extend(
        [
            (
                "shape_data_url",
                {
                    "prompt": prompt,
                    "image_urls": [f"data:image/png;base64,{b64}"],
                    "image_size": {"width": COLLAGE_SIZE, "height": COLLAGE_SIZE},
                },
            ),
            (
                "shape_image_base64",
                {
                    "prompt": prompt,
                    "image": {"data": b64, "mime_type": "image/png"},
                    "image_size": {"width": COLLAGE_SIZE, "height": COLLAGE_SIZE},
                },
            ),
            (
                "shape_images_base64",
                {
                    "prompt": prompt,
                    "images": [{"data": b64, "mime_type": "image/png"}],
                    "image_size": {"width": COLLAGE_SIZE, "height": COLLAGE_SIZE},
                },
            ),
        ]
    )
    async with httpx.AsyncClient(timeout=120) as client:
        for shape_name, payload in attempts:
            print(f"[fal] attempt {shape_name}", flush=True)
            start = time.time()
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.time() - start) * 1000)
                print(f"[fal] exception {shape_name} after {elapsed_ms}ms: {exc}", flush=True)
                continue
            elapsed_ms = int((time.time() - start) * 1000)
            print(f"[fal] status={resp.status_code} shape={shape_name} elapsed={elapsed_ms}ms", flush=True)
            if not (200 <= resp.status_code < 300):
                try:
                    err_body = await resp.aread()
                except Exception:
                    err_body = b""
                if err_body:
                    snippet = err_body.decode(errors="ignore").strip().replace("\n", " ")
                    print(f"[fal] error body: {snippet[:240]}", flush=True)
                else:
                    print(f"[fal] bad status={resp.status_code}", flush=True)
                continue
            ctype = resp.headers.get("content-type", "")
            body = await resp.aread()
            if "application/json" in ctype:
                try:
                    data = resp.json()
                except Exception:
                    data = None
                img_or_url = _parse_resp_json(data) if isinstance(data, dict) else None
                if isinstance(img_or_url, (bytes, bytearray)):
                    print("[fal] got inline image", flush=True)
                    return bytes(img_or_url)
                if isinstance(img_or_url, str):
                    try:
                        print("[fal] fetching url", flush=True)
                        got = await client.get(img_or_url)
                        got.raise_for_status()
                        return got.content
                    except Exception:
                        print("[fal] fetch url failed", flush=True)
                        continue
            else:
                if body:
                    print("[fal] got body bytes", flush=True)
                    return body
    return None


async def fal_text_to_image(prompt: str, width: int, height: int) -> Optional[bytes]:
    api_key = os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY")
    if not api_key:
        print("[fal] missing api key", flush=True)
        return None
    url = "https://fal.run/fal-ai/nano-banana"
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "image_size": {"width": width, "height": height}}
    async with httpx.AsyncClient(timeout=120) as client:
        print("[fal-t2i] attempt", flush=True)
        resp = await client.post(url, json=payload, headers=headers)
        if not (200 <= resp.status_code < 300):
            print(f"[fal-t2i] bad status={resp.status_code}", flush=True)
            return None
        ctype = resp.headers.get("content-type", "")
        body = await resp.aread()
        if "application/json" in ctype:
            try:
                data = resp.json()
            except Exception:
                data = None
            img_or_url = _parse_resp_json(data) if isinstance(data, dict) else None
            if isinstance(img_or_url, (bytes, bytearray)):
                print("[fal-t2i] got inline image", flush=True)
                return bytes(img_or_url)
            if isinstance(img_or_url, str):
                try:
                    print("[fal-t2i] fetching url", flush=True)
                    got = await client.get(img_or_url)
                    got.raise_for_status()
                    return got.content
                except Exception:
                    print("[fal-t2i] fetch url failed", flush=True)
                    return None
        else:
            if body:
                print("[fal-t2i] got body bytes", flush=True)
                return body
    return None


@app.get("/maps/{map_id}/tiles/{z}/{x}/{y}.png")
def get_map_tile(map_id: str, z: int, x: int, y: int):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    p = _find_latest_tile_path(map_id, z, x, y)
    if p and p.exists():
        return FileResponse(str(p))
    raise HTTPException(status_code=404, detail="not found")


@app.post("/maps/{map_id}/generate")
async def generate_for_map(map_id: str, inp: GenerateIn):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    _ensure_map(map_id)
    try:
        print(f"[gen] map={map_id} z={inp.z} x={inp.x} y={inp.y}", flush=True)
        collage, info = _build_collage_image(map_id, inp.z, inp.x, inp.y)
        have_neighbors = sum(1 for _, _, ok in info if ok)
        print(f"[gen] neighbors_present={have_neighbors}", flush=True)
        buf = BytesIO()
        collage.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        base, tiles_root, up_root, down_root, _ = _map_paths(map_id)
        seq = _next_seq(map_id)
        (up_root / f"{seq:06d}.png").write_bytes(png_bytes)
        if have_neighbors == 0:
            full_prompt = f"{inp.prompt}. No text."
            result_bytes = await fal_text_to_image(full_prompt, TILE_SIZE, TILE_SIZE)
            mode = "t2i"
        else:
            full_prompt = f"{inp.prompt} Only fill the checkerboard center tile; keep all other areas exactly unchanged; match edges to neighbors; no text."
            result_bytes = await fal_generate_edit(png_bytes, full_prompt)
            mode = "i2i"
        if not result_bytes:
            print("[gen] model failed, using fallback", flush=True)
            fallback = True
            result_bytes = png_bytes
        else:
            fallback = False
            (down_root / f"{seq:06d}.png").write_bytes(result_bytes)
        with Image.open(BytesIO(result_bytes)) as im:
            w, h = im.size
            if mode == "i2i":
                seg_w = max(1, int(w / COLLAGE_TILES))
                seg_h = max(1, int(h / COLLAGE_TILES))
                tile = im.crop((seg_w, seg_h, seg_w * 2, seg_h * 2))
            else:
                tile = im
            if tile.size != (TILE_SIZE, TILE_SIZE):
                tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            out_dir = tiles_root / str(inp.z) / str(inp.x)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{seq:06d}_{inp.y}.png"
            tile.save(out_path)
        print(f"[gen] saved {out_path}", flush=True)
        return {"ok": True, "seq": seq, "fallback": fallback, "tile": f"/maps/{map_id}/tiles/{inp.z}/{inp.x}/{inp.y}.png"}
    except Exception as e:
        print(f"[gen] exception {e}", flush=True)
        return JSONResponse({"ok": False, "error": str(e)})
