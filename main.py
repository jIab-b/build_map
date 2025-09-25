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
TILES_DIR = BASE_DIR / "tiles"
TILE_SIZE = 256
ZOOM_LEVEL = 8

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class GenerateIn(BaseModel):
	x: int
	y: int
	z: int
	prompt: str


@app.get("/")
def root():
	return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/tiles/{z}/{x}/{y}.png")
def get_tile(z: int, x: int, y: int):
	path = TILES_DIR / str(z) / str(x) / f"{y}.png"
	if path.exists():
		return FileResponse(str(path))
	raise HTTPException(status_code=404, detail="not found")


@app.delete("/tiles/{z}/{x}/{y}")
def delete_tile(z: int, x: int, y: int):
	path = TILES_DIR / str(z) / str(x) / f"{y}.png"
	try:
		path.unlink(missing_ok=True)
	except TypeError:
		if path.exists():
			path.unlink()
	return {"ok": True}


def make_checkerboard(size: int, cell: int = 32) -> Image.Image:
	img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
	d = ImageDraw.Draw(img)
	for j in range(0, size, cell):
		for i in range(0, size, cell):
			if ((i // cell) + (j // cell)) % 2 == 0:
				d.rectangle([i, j, i + cell, j + cell], fill=(180, 180, 180, 160))
			else:
				d.rectangle([i, j, i + cell, j + cell], fill=(140, 140, 140, 160))
	return img


def build_collage(z: int, x: int, y: int) -> Image.Image:
	canvas = Image.new("RGBA", (TILE_SIZE * 3, TILE_SIZE * 3), (0, 0, 0, 0))
	matte = make_checkerboard(TILE_SIZE)
	for dy in range(-1, 2):
		for dx in range(-1, 2):
			src = TILES_DIR / str(z) / str(x + dx) / f"{y + dy}.png"
			pos = ((dx + 1) * TILE_SIZE, (dy + 1) * TILE_SIZE)
			try:
				with Image.open(src) as im:
					canvas.paste(im.convert("RGBA"), pos)
			except Exception:
				canvas.paste(matte, pos)
	canvas.paste(matte, (TILE_SIZE, TILE_SIZE))
	return canvas




def _preset() -> str:
	# Choose model preset: "lightning" or "banana".
	# Default to banana; set FAL_PRESET=lightning to switch back.
	p = (os.environ.get("FAL_PRESET") or os.environ.get("USE_MODEL") or "banana").strip().lower()
	return "banana" if p.startswith("banan") else "lightning"


def _endpoints() -> Tuple[str, str]:
	# Returns (t2i_url, i2i_url)
	if _preset() == "banana":
		return (
			"https://fal.run/fal-ai/nano-banana",  # T2I
			"https://fal.run/fal-ai/nano-banana/edit",  # I2I
		)
	else:
		return (
			"https://fal.run/fal-ai/fast-lightning-sdxl",  # T2I
			"https://fal.run/fal-ai/fast-lightning-sdxl/image-to-image",  # I2I
		)


async def _upload_public_image(buf: bytes) -> Optional[str]:
	# Anonymous temp host; replace with your own (S3/CDN) in production.
	try:
		async with httpx.AsyncClient(timeout=60) as up:
			files = {"file": ("grid.png", buf, "image/png")}
			up_resp = await up.post("https://0x0.st", files=files)
			if 200 <= up_resp.status_code < 300:
				url_text = (await up_resp.aread()).decode(errors="ignore").strip()
				if url_text.startswith("http"):
					return url_text
			return None
	except Exception:
		return None


def _parse_resp_json(obj: dict) -> Optional[bytes | str]:
	# Try several expected shapes for fal endpoints
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
		print("[fal] missing FAL_API_KEY", flush=True)
		return None
	_, i2i_url = _endpoints()
	headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
	b64 = base64.b64encode(png_grid).decode()
	public_url = await _upload_public_image(png_grid)
	if not public_url:
		print("[fal-edit] could not obtain public URL for grid image")
		return None

	attempts = [
		("shape_urls", {"prompt": prompt, "image_urls": [public_url]}),
		("shape_url_single", {"prompt": prompt, "image_url": public_url}),
		("shape_images_base64", {"prompt": prompt, "images": [{"data": b64, "mime_type": "image/png"}]}),
	]

	async with httpx.AsyncClient(timeout=120) as client:
		for shape_name, payload in attempts:
			start = time.time()
			resp = await client.post(i2i_url, json=payload, headers=headers)
			elapsed_ms = int((time.time() - start) * 1000)
			ok = 200 <= resp.status_code < 300
			print(f"[fal-edit:{_preset()}:{shape_name}] status={resp.status_code} elapsed={elapsed_ms}ms payloadKB={len(png_grid)//1024}")
			if not ok:
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
					return bytes(img_or_url)
				if isinstance(img_or_url, str):
					try:
						got = await client.get(img_or_url)
						got.raise_for_status()
						return got.content
					except Exception:
						continue
			else:
				if body:
					return body
	return None


async def fal_text_to_image(prompt: str, width: int = 768, height: int = 768) -> Optional[bytes]:
	api_key = os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY")
	if not api_key:
		print("[fal] missing FAL_API_KEY", flush=True)
		return None
	t2i_url, _ = _endpoints()
	headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
	payload = {"prompt": prompt, "image_size": {"width": width, "height": height}}
	async with httpx.AsyncClient(timeout=120) as client:
		resp = await client.post(t2i_url, json=payload, headers=headers)
		if not (200 <= resp.status_code < 300):
			print(f"[fal-t2i:{_preset()}] status={resp.status_code}")
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
				return bytes(img_or_url)
			if isinstance(img_or_url, str):
				try:
					got = await client.get(img_or_url)
					got.raise_for_status()
					return got.content
				except Exception:
					return None
		else:
			if body:
				return body
	return None




@app.post("/generate")
async def generate(inp: GenerateIn):
	if inp.z != ZOOM_LEVEL:
		raise HTTPException(status_code=400, detail="z must be 8")
	try:
		collage = build_collage(inp.z, inp.x, inp.y)
		buf = BytesIO()
		collage.save(buf, format="PNG")
		png_bytes = buf.getvalue()
		full_prompt = f"{inp.prompt} Only fill the checkerboard center tile; keep all areas outside the checkerboard exactly unchanged; match edges to neighbors; no text."
		result_bytes = await fal_generate_edit(png_bytes, full_prompt)
		if not result_bytes:
			return JSONResponse({"ok": False, "error": "model failed"})
		with Image.open(BytesIO(result_bytes)) as im:
			w, h = im.size
			tile = im.crop((int(w / 3), int(h / 3), int(2 * w / 3), int(2 * h / 3)))
			if tile.size != (TILE_SIZE, TILE_SIZE):
				tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
			out_dir = TILES_DIR / str(inp.z) / str(inp.x)
			out_dir.mkdir(parents=True, exist_ok=True)
			out_path = out_dir / f"{inp.y}.png"
			tile.save(out_path)
		return {"ok": True}
	except Exception as e:
		return JSONResponse({"ok": False, "error": str(e)})
