import base64
import os
import time
from io import BytesIO
from typing import Optional, Tuple

import httpx


def _preset() -> str:
    # Choose model preset: "lightning" or "banana".
    # Default to banana; set FAL_PRESET=lightning to switch back.
    p = (os.environ.get("FAL_PRESET") or os.environ.get("USE_MODEL") or "banana").strip().lower()
    return "banana" if p.startswith("banan") else "lightning"


def _endpoints() -> Tuple[str, str]:
    # Returns (t2i_url, i2i_url)
    if _preset() == "banana":
        return (
            "https://fal.run/fal-ai/nano-banana",                   # T2I
            "https://fal.run/fal-ai/nano-banana/edit",              # I2I
        )
    else:
        return (
            "https://fal.run/fal-ai/fast-lightning-sdxl",           # T2I
            "https://fal.run/fal-ai/fast-lightning-sdxl/image-to-image",  # I2I
        )


async def _upload_public_image(buf: bytes) -> Optional[str]:
    # Anonymous temp host; replace with your own (S3/CDN) in production.
    try:
        async with httpx.AsyncClient(timeout=60) as up:
            files = {"file": ("grid.png", buf, "image/png")}
            up_resp = await up.post("https://tmpfiles.org/api/v1/upload", files=files)
            if 200 <= up_resp.status_code < 300:
                try:
                    payload = up_resp.json()
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    data = payload.get("data") or {}
                    url_text = data.get("url") if isinstance(data, dict) else None
                else:
                    body = (await up_resp.aread()).decode(errors="ignore").strip()
                    url_text = body
                if isinstance(url_text, str) and url_text.startswith("http"):
                    return url_text
            return None
    except Exception:
        return None


def _parse_resp_json(obj: dict) -> Optional[bytes | str]:
    # Try several expected shapes for fal endpoints
    if not isinstance(obj, dict):
        return None
    if obj.get("image") and isinstance(obj["image"], dict) and obj["image"].get("data"):
        return base64.b64decode(obj["image"]["data"])  # inline base64 image
    if obj.get("images") and isinstance(obj["images"], list) and obj["images"]:
        first = obj["images"][0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]  # return URL
    if obj.get("output") and isinstance(obj["output"], list) and obj["output"]:
        first = obj["output"][0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
    # Some models return {image: {url: ...}}
    if obj.get("image") and isinstance(obj["image"], dict) and obj["image"].get("url"):
        return obj["image"]["url"]
    return None


async def fal_generate_edit(png_grid: bytes, prompt: str) -> Optional[bytes]:
    api_key = os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY")
    if not api_key:
        print("[fal] missing FAL_API_KEY", flush=True)
        return None
    t2i_url, i2i_url = _endpoints()
    url = i2i_url
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    public_url = await _upload_public_image(png_grid)
    attempts = []
    if public_url:
        attempts.extend(
            [
                ("shape_urls", {"prompt": prompt, "image_urls": [public_url]}),
                ("shape_url_single", {"prompt": prompt, "image_url": public_url}),
            ]
        )
    else:
        print("[fal-edit] could not obtain public URL for grid image; falling back to inline payload")

    b64 = base64.b64encode(png_grid).decode()
    attempts.extend(
        [
            (
                "shape_data_url",
                {
                    "prompt": prompt,
                    "image_urls": [f"data:image/png;base64,{b64}"],
                },
            ),
            (
                "shape_image_base64",
                {
                    "prompt": prompt,
                    "image": {"data": b64, "mime_type": "image/png"},
                },
            ),
            (
                "shape_images_base64",
                {
                    "prompt": prompt,
                    "images": [{"data": b64, "mime_type": "image/png"}],
                },
            ),
        ]
    )

    async with httpx.AsyncClient(timeout=120) as client:
        for shape_name, payload in attempts:
            start = time.time()
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.time() - start) * 1000)
                print(f"[fal-edit:{_preset()}:{shape_name}] exception after {elapsed_ms}ms: {exc}")
                continue
            elapsed_ms = int((time.time() - start) * 1000)
            ok = 200 <= resp.status_code < 300
            print(f"[fal-edit:{_preset()}:{shape_name}] status={resp.status_code} elapsed={elapsed_ms}ms payloadKB={len(png_grid)//1024}")
            if not ok:
                try:
                    err_body = await resp.aread()
                except Exception:
                    err_body = b""
                if err_body:
                    snippet = err_body.decode(errors="ignore").strip().replace("\n", " ")
                    print(f"  ↳ error body: {snippet[:240]}")
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
