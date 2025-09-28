import base64
import json
import math
import os
import shutil
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
MAX_SELECTION_TILES = 15
ANIMATIONS_FILENAME = "animations.json"
ANIMATIONS_DIRNAME = "animations"
DEFAULT_ANIMATION_FRAME_DURATION_MS = 120
DEFAULT_ANIMATION_FRAME_COUNT = 16

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


class GenerateIn(BaseModel):
    x: int
    y: int
    width: int = 1
    height: int = 1
    z: int
    prompt: str
    delete: bool = False


class MapCreateIn(BaseModel):
    map_id: str


class AnimationIn(BaseModel):
    x: int
    y: int
    z: int
    frame_count: int = DEFAULT_ANIMATION_FRAME_COUNT
    frame_duration_ms: Optional[int] = None
    sheet_data: Optional[str] = None
    prompt: Optional[str] = None
    delete: bool = False


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


def _scan_existing_tiles(map_id: str, z: int) -> list[dict]:
    base, tiles_root, _, _, _ = _map_paths(map_id)
    out: list[dict] = []
    z_dir = tiles_root / str(z)
    if not z_dir.exists():
        return out
    for x_dir in z_dir.iterdir():
        if not x_dir.is_dir():
            continue
        try:
            x_val = int(x_dir.name)
        except Exception:
            continue
        for p in x_dir.glob("*_*.png"):
            stem = p.stem
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                y_val = int(parts[1])
            except Exception:
                continue
            out.append({
                "x": x_val,
                "y": y_val,
                "url": f"/maps/{map_id}/tiles/{z}/{x_val}/{y_val}.png"
            })
    return out


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


def _animations_root(map_id: str) -> Path:
    base, _, _, _, _ = _map_paths(map_id)
    root = base / ANIMATIONS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _animations_manifest_path(map_id: str) -> Path:
    base, _, _, _, _ = _map_paths(map_id)
    return base / ANIMATIONS_FILENAME


def _load_animations_manifest(map_id: str) -> dict:
    path = _animations_manifest_path(map_id)
    if not path.exists():
        return {"tiles": {}}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {"tiles": {}}
    if not isinstance(data, dict):
        return {"tiles": {}}
    tiles = data.get("tiles")
    if not isinstance(tiles, dict):
        data["tiles"] = {}
    return data


def _save_animations_manifest(map_id: str, manifest: dict) -> None:
    path = _animations_manifest_path(map_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")


def _animation_key(z: int, x: int, y: int) -> str:
    return f"{z}/{x}/{y}"


def _decode_image_data(data: str) -> bytes:
    if not isinstance(data, str):
        raise ValueError("sheet data must be a string")
    text = data.strip()
    if text.startswith("data:"):
        _, _, encoded = text.partition(",")
        text = encoded.strip()
    try:
        return base64.b64decode(text)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("invalid base64 image data") from exc


def _spritesheet_layout(frame_count: int) -> tuple[int, int, int, int]:
    count = max(1, frame_count)
    frames_per_row = min(4, count)
    rows = max(1, math.ceil(count / frames_per_row))
    width = frames_per_row * TILE_SIZE
    height = rows * TILE_SIZE
    return frames_per_row, rows, width, height


def _build_repeated_sheet_bytes(tile_path: Optional[Path], frame_count: int) -> bytes:
    frames_per_row, rows, width, height = _spritesheet_layout(frame_count)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if tile_path and tile_path.exists():
        with Image.open(tile_path) as src:
            tile_img = src.convert("RGBA")
            if tile_img.size != (TILE_SIZE, TILE_SIZE):
                tile_img = tile_img.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
            base_tile = tile_img
    else:
        base_tile = _make_matte(TILE_SIZE)
    for index in range(frame_count):
        row = index // frames_per_row
        col = index % frames_per_row
        pos = (col * TILE_SIZE, row * TILE_SIZE)
        canvas.paste(base_tile, pos)
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _make_matte(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cell = max(4, size // 8)
    for j in range(0, size, cell):
        for i in range(0, size, cell):
            c = 220 if ((i // cell) + (j // cell)) % 2 == 0 else 200
            d.rectangle([i, j, i + cell, j + cell], fill=(c, c, c, 120))
    return img


def _build_selection_image(
    map_id: str, z: int, x: int, y: int, width: int, height: int
) -> tuple[Image.Image, list[tuple[int, int, bool]], bool]:
    width_px = max(1, width * TILE_SIZE)
    height_px = max(1, height * TILE_SIZE)
    canvas = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    matte = _make_matte(TILE_SIZE)
    info: list[tuple[int, int, bool]] = []
    have_tiles = False
    for dy in range(height):
        for dx in range(width):
            pos_x = dx * TILE_SIZE
            pos_y = dy * TILE_SIZE
            pos = (pos_x, pos_y)
            tile_x = x + dx
            tile_y = y + dy
            p = _find_latest_tile_path(map_id, z, tile_x, tile_y)
            exists = bool(p and p.exists())
            info.append((tile_x, tile_y, exists))
            if not exists:
                canvas.paste(matte, pos)
                continue
            have_tiles = True
            with Image.open(p) as im:
                tile_img = im.convert("RGBA")
                if tile_img.size != (TILE_SIZE, TILE_SIZE):
                    tile_img = tile_img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                canvas.paste(tile_img, pos)
    return canvas, info, have_tiles


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


async def fal_generate_edit(png_grid: bytes, prompt: str, width: int, height: int) -> Optional[bytes]:
    api_key = os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY")
    if not api_key:
        print("[fal] missing api key", flush=True)
        return None
    url = "https://fal.run/fal-ai/nano-banana/edit"
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    public_url = await _upload_public_image(png_grid)
    attempts: list[tuple[str, dict]] = []
    image_size = {"width": width, "height": height}
    if public_url:
        attempts.extend(
            [
                (
                    "shape_urls",
                    {
                        "prompt": prompt,
                        "image_urls": [public_url],
                        "image_size": image_size,
                    },
                ),
                (
                    "shape_url_single",
                    {
                        "prompt": prompt,
                        "image_url": public_url,
                        "image_size": image_size,
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
                        "image_size": image_size,
                    },
                ),
                (
                    "shape_image_base64",
                    {
                        "prompt": prompt,
                        "image": {"data": b64, "mime_type": "image/png"},
                        "image_size": image_size,
                    },
                ),
                (
                    "shape_images_base64",
                    {
                        "prompt": prompt,
                        "images": [{"data": b64, "mime_type": "image/png"}],
                        "image_size": image_size,
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


@app.get("/maps/{map_id}/tiles/{z}")
def list_existing_tiles(map_id: str, z: int):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    _ensure_map(map_id)
    tiles = _scan_existing_tiles(map_id, z)
    tiles.sort(key=lambda t: (t["x"], t["y"]))
    return {"tiles": tiles}


@app.get("/maps/{map_id}/animations")
def list_tile_animations(map_id: str):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    _ensure_map(map_id)
    manifest = _load_animations_manifest(map_id)
    tiles_section = manifest.get("tiles", {}) if isinstance(manifest, dict) else {}
    items: list[dict] = []
    for key_str, entry in tiles_section.items():
        if not isinstance(entry, dict):
            continue
        parts = str(key_str).split("/")
        if len(parts) != 3:
            continue
        try:
            z_val = int(parts[0])
            x_val = int(parts[1])
            y_val = int(parts[2])
        except Exception:
            continue
        animated = bool(entry.get("animated"))
        frame_urls = entry.get("frame_urls")
        if not isinstance(frame_urls, list):
            frame_urls = []
        frame_duration = entry.get("frame_duration_ms")
        try:
            frame_duration_ms = int(frame_duration)
            if frame_duration_ms <= 0:
                raise ValueError
        except Exception:
            frame_duration_ms = DEFAULT_ANIMATION_FRAME_DURATION_MS
        version = entry.get("version")
        try:
            version_val = int(version)
        except Exception:
            version_val = None
        frame_count = entry.get("frame_count")
        try:
            frame_count_val = int(frame_count)
        except Exception:
            frame_count_val = len(frame_urls)
        items.append(
            {
                "z": z_val,
                "x": x_val,
                "y": y_val,
                "animated": animated,
                "frame_count": frame_count_val,
                "frame_urls": frame_urls,
                "frame_duration_ms": frame_duration_ms,
                "version": version_val,
                "prompt": entry.get("prompt"),
            }
        )
    items.sort(key=lambda v: (v["z"], v["x"], v["y"]))
    return {"animations": items}


@app.get("/maps/{map_id}/animations/{z}/{x}/{y}/{frame_name}")
def get_animation_frame(map_id: str, z: int, x: int, y: int, frame_name: str):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    _ensure_map(map_id)
    if Path(frame_name).name != frame_name or "/" in frame_name or ".." in frame_name:
        raise HTTPException(status_code=404, detail="not found")
    if not frame_name.lower().endswith(".png"):
        raise HTTPException(status_code=404, detail="not found")
    frames_dir = _animations_root(map_id) / str(z) / str(x) / str(y)
    target = frames_dir / frame_name
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    raise HTTPException(status_code=404, detail="not found")


@app.post("/maps/{map_id}/animations")
async def set_tile_animation(map_id: str, inp: AnimationIn):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    _ensure_map(map_id)
    _, tiles_root, up_root, down_root, _ = _map_paths(map_id)
    frame_duration = inp.frame_duration_ms or DEFAULT_ANIMATION_FRAME_DURATION_MS
    try:
        frame_duration_ms = int(frame_duration)
    except Exception:
        frame_duration_ms = DEFAULT_ANIMATION_FRAME_DURATION_MS
    if frame_duration_ms <= 0:
        frame_duration_ms = DEFAULT_ANIMATION_FRAME_DURATION_MS

    key = _animation_key(inp.z, inp.x, inp.y)
    manifest = _load_animations_manifest(map_id)
    manifest_tiles = manifest.setdefault("tiles", {})
    animations_root = _animations_root(map_id)
    frames_dir = animations_root / str(inp.z) / str(inp.x) / str(inp.y)
    tile_url_base = f"/maps/{map_id}/tiles/{inp.z}/{inp.x}/{inp.y}.png"

    if inp.delete:
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        removed_version = int(time.time())
        entry = {
            "animated": False,
            "frame_count": 0,
            "frame_urls": [],
            "frame_duration_ms": frame_duration_ms,
            "updated_at": removed_version,
            "version": removed_version,
            "prompt": None,
        }
        manifest_tiles[key] = entry
        _save_animations_manifest(map_id, manifest)
        latest_tile = _find_latest_tile_path(map_id, inp.z, inp.x, inp.y)
        tile_payload = None
        if latest_tile and latest_tile.exists():
            tile_payload = {
                "x": inp.x,
                "y": inp.y,
                "z": inp.z,
                "url": f"{tile_url_base}?t={removed_version}",
            }
        response_entry = entry.copy()
        response_entry.update({"x": inp.x, "y": inp.y, "z": inp.z})
        return {"ok": True, "tile": tile_payload, "animation": response_entry}

    frame_count = inp.frame_count or DEFAULT_ANIMATION_FRAME_COUNT
    try:
        frame_count = int(frame_count)
    except Exception:
        frame_count = DEFAULT_ANIMATION_FRAME_COUNT
    if frame_count != DEFAULT_ANIMATION_FRAME_COUNT:
        raise HTTPException(status_code=400, detail="frame_count must be 16")

    frames_per_row, rows, sheet_width, sheet_height = _spritesheet_layout(frame_count)

    sheet_bytes: Optional[bytes] = None
    uploaded_sheet_bytes: Optional[bytes] = None
    prompt_used: Optional[str] = None

    if inp.sheet_data:
        try:
            sheet_bytes = _decode_image_data(inp.sheet_data)
            uploaded_sheet_bytes = sheet_bytes
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        user_prompt = (inp.prompt or "").strip()
        if not user_prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        prompt_used = user_prompt
        base_tile_path = _find_latest_tile_path(map_id, inp.z, inp.x, inp.y)
        base_sheet_bytes: Optional[bytes] = None
        try:
            base_sheet_bytes = _build_repeated_sheet_bytes(base_tile_path, frame_count)
        except Exception:
            base_sheet_bytes = None
        sheet_prompt = (
            f"{user_prompt}. 4x4 grid of sequential animation frames for a 64x64 pixel art tile. "
            "Each frame is exactly 64x64 pixels, pixel-perfect, crisp edges, consistent lighting, no text."
        )
        result_bytes: Optional[bytes] = None
        if base_sheet_bytes and base_tile_path:
            result_bytes = await fal_generate_edit(
                base_sheet_bytes,
                sheet_prompt,
                sheet_width,
                sheet_height,
            )
        if not result_bytes:
            result_bytes = await fal_text_to_image(
                sheet_prompt,
                sheet_width,
                sheet_height,
            )
        if not result_bytes:
            if base_sheet_bytes:
                result_bytes = base_sheet_bytes
            else:
                fallback_canvas = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
                buf = BytesIO()
                fallback_canvas.save(buf, format="PNG")
                result_bytes = buf.getvalue()
        sheet_bytes = result_bytes

    if not sheet_bytes:
        raise HTTPException(status_code=500, detail="failed to prepare animation spritesheet")

    seq = _next_seq(map_id)
    if uploaded_sheet_bytes:
        try:
            (up_root / f"{seq:06d}_anim.png").write_bytes(uploaded_sheet_bytes)
        except Exception:
            pass
    try:
        (down_root / f"{seq:06d}_anim.png").write_bytes(sheet_bytes)
    except Exception:
        pass

    with Image.open(BytesIO(sheet_bytes)) as src:
        sheet_img = src.convert("RGBA")

    if sheet_img.size != (sheet_width, sheet_height):
        sheet_img = sheet_img.resize((sheet_width, sheet_height), Image.NEAREST)

    frame_width = sheet_width // frames_per_row if frames_per_row else TILE_SIZE
    frame_height = sheet_height // rows if rows else TILE_SIZE
    if frame_width <= 0 or frame_height <= 0:
        raise HTTPException(status_code=400, detail="invalid spritesheet dimensions")

    frames: list[Image.Image] = []
    for index in range(frame_count):
        row = index // frames_per_row
        col = index % frames_per_row
        if row >= rows:
            raise HTTPException(status_code=400, detail="spritesheet layout insufficient for frames")
        box = (
            col * frame_width,
            row * frame_height,
            (col + 1) * frame_width,
            (row + 1) * frame_height,
        )
        frame_img = sheet_img.crop(box).convert("RGBA")
        if frame_img.size != (TILE_SIZE, TILE_SIZE):
            frame_img = frame_img.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
        frames.append(frame_img)

    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_urls: list[str] = []
    for idx, img in enumerate(frames):
        frame_path = frames_dir / f"frame_{idx:02d}.png"
        img.save(frame_path)
        frame_urls.append(
            f"/maps/{map_id}/animations/{inp.z}/{inp.x}/{inp.y}/{frame_path.name}"
        )

    tile_dir = tiles_root / str(inp.z) / str(inp.x)
    tile_dir.mkdir(parents=True, exist_ok=True)
    first_frame_path = tile_dir / f"{seq:06d}_{inp.y}.png"
    frames[0].save(first_frame_path)

    manifest_entry = {
        "animated": True,
        "frame_count": frame_count,
        "frame_urls": frame_urls,
        "frame_duration_ms": frame_duration_ms,
        "updated_at": int(time.time()),
        "version": seq,
    }
    if prompt_used is not None:
        manifest_entry["prompt"] = prompt_used
    manifest_tiles[key] = manifest_entry
    _save_animations_manifest(map_id, manifest)

    tile_payload = {
        "x": inp.x,
        "y": inp.y,
        "z": inp.z,
        "url": f"{tile_url_base}?t={seq}",
    }

    response_entry = manifest_entry.copy()
    response_entry["frame_urls"] = [f"{url}?t={seq}" for url in frame_urls]
    response_entry.update({"x": inp.x, "y": inp.y, "z": inp.z})

    return {"ok": True, "tile": tile_payload, "animation": response_entry}


@app.post("/maps/{map_id}/generate")
async def generate_for_map(map_id: str, inp: GenerateIn):
    if not _valid_map_id(map_id):
        raise HTTPException(status_code=404, detail="not found")
    _ensure_map(map_id)
    try:
        width = int(inp.width or 1)
        height = int(inp.height or 1)
        if width < 1 or height < 1:
            raise HTTPException(status_code=400, detail="invalid selection size")
        if width > MAX_SELECTION_TILES or height > MAX_SELECTION_TILES:
            raise HTTPException(status_code=400, detail="selection too large")

        print(
            f"[gen] map={map_id} z={inp.z} x={inp.x} y={inp.y} width={width} height={height}",
            flush=True,
        )
        selection_img, info, have_tiles = _build_selection_image(
            map_id, inp.z, inp.x, inp.y, width, height
        )
        existing_tiles = sum(1 for _, _, exists in info if exists)
        print(
            f"[gen] selection_contains={existing_tiles} existing tiles (have_tiles={have_tiles})",
            flush=True,
        )

        base, tiles_root, up_root, down_root, _ = _map_paths(map_id)
        seq = _next_seq(map_id)

        if inp.delete:
            base, tiles_root, _, _, _ = _map_paths(map_id)
            removed = []
            for dy in range(height):
                for dx in range(width):
                    tile_x = inp.x + dx
                    tile_y = inp.y + dy
                    x_dir = tiles_root / str(inp.z) / str(tile_x)
                    if x_dir.exists():
                        for p in x_dir.glob(f"*_{tile_y}.png"):
                            try:
                                p.unlink()
                            except Exception:
                                pass
                    removed.append({"x": tile_x, "y": tile_y})
            print(f"[gen] deleted {len(removed)} tiles", flush=True)
            return {
                "ok": True,
                "seq": None,
                "fallback": False,
                "mode": "delete",
                "tiles": [],
                "removed": removed,
                "width": width,
                "height": height,
            }

        png_bytes: Optional[bytes] = None
        if have_tiles:
            buf = BytesIO()
            selection_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            (up_root / f"{seq:06d}.png").write_bytes(png_bytes)
        else:
            print("[gen] no existing tiles in selection; skipping collage upload", flush=True)

        req_width = width * TILE_SIZE
        req_height = height * TILE_SIZE

        user_prompt = (inp.prompt or "").strip()
        if have_tiles and png_bytes:
            full_prompt = (
                f"{user_prompt}, pixel art style, top down view"
                if user_prompt
                else "pixel art style, top down view"
            )
            result_bytes = await fal_generate_edit(png_bytes, full_prompt, req_width, req_height)
            mode = "i2i"
        else:
            full_prompt = (
                f"{user_prompt}, pixel art style, top down view. No text."
                if user_prompt
                else "pixel art style, top down view. No text."
            )
            result_bytes = await fal_text_to_image(full_prompt, req_width, req_height)
            mode = "t2i"

        if not result_bytes:
            print("[gen] model failed, using fallback", flush=True)
            fallback = True
            if have_tiles and png_bytes:
                result_bytes = png_bytes
            else:
                empty_img = Image.new("RGBA", (req_width, req_height), (0, 0, 0, 0))
                buf = BytesIO()
                empty_img.save(buf, format="PNG")
                result_bytes = buf.getvalue()
        else:
            fallback = False
            (down_root / f"{seq:06d}.png").write_bytes(result_bytes)

        updated_tiles = []
        with Image.open(BytesIO(result_bytes)) as im:
            combo = im.convert("RGBA")
            if combo.size != (req_width, req_height):
                combo = combo.resize((req_width, req_height), Image.LANCZOS)
            for dy in range(height):
                for dx in range(width):
                    crop_box = (
                        dx * TILE_SIZE,
                        dy * TILE_SIZE,
                        (dx + 1) * TILE_SIZE,
                        (dy + 1) * TILE_SIZE,
                    )
                    tile_img = combo.crop(crop_box).convert("RGBA")
                    if tile_img.size != (TILE_SIZE, TILE_SIZE):
                        tile_img = tile_img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                    out_dir = tiles_root / str(inp.z) / str(inp.x + dx)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{seq:06d}_{inp.y + dy}.png"
                    tile_img.save(out_path)
                    updated_tiles.append(
                        {
                            "x": inp.x + dx,
                            "y": inp.y + dy,
                            "url": f"/maps/{map_id}/tiles/{inp.z}/{inp.x + dx}/{inp.y + dy}.png",
                        }
                    )
        print(
            f"[gen] saved {len(updated_tiles)} tiles seq={seq} fallback={fallback}",
            flush=True,
        )
        return {
            "ok": True,
            "seq": seq,
            "fallback": fallback,
            "mode": mode,
            "tiles": updated_tiles,
            "width": width,
            "height": height,
        }
    except Exception as e:
        print(f"[gen] exception {e}", flush=True)
        return JSONResponse({"ok": False, "error": str(e)})
