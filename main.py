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
from os import scandir

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAPS_DIR = BASE_DIR / "maps"
TILE_SIZE = 64
MAX_SELECTION_TILES = 15

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
