"""
视觉层内容安全过滤器（第二道防线）。

把同一关键词下的全部候选素材缩略图拼成一张带编号的网格图，用一次视觉模型
请求批量判断每个素材是否包含女性人物。相比逐个素材发请求，一次合成图调用
可以把请求数从 N 次降到 1 次，显著降低配额消耗和总延迟。

失败策略是“放行并告警”（fail-open）：下载缩略图、拼图或调用模型出现任何
异常时保留原候选，绝不让安全过滤阻断视频生成主流程；第一道防线（元数据
关键词过滤）此时仍然生效。
"""

import hashlib
import io
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from loguru import logger

from app.config import config

# 单张合成网格最多包含的素材数。超过该数量时分批发送，每批仍然只占一次
# 视觉请求；网格过大时缩略图过小会明显降低识别准确率。
MAX_ASSETS_PER_GRID = 12
# 缩略图并发下载线程数。缩略图都很小，少量线程即可打满带宽。
_THUMBNAIL_WORKERS = 8
_THUMBNAIL_TIMEOUT = (10, 30)
_THUMBNAIL_MAX_BYTES = 5 * 1024 * 1024

_GRID_COLUMNS = 4
_CELL_SIZE = 240
_LABEL_BAR_HEIGHT = 28

# 用户指定的审查提示词。输出格式固定为逐行 “ASSET_n: SAFE | WOMAN_PRESENT”。
SAFETY_PROMPT = """You are a content safety filter. Examine this grid of stock media thumbnails.
For each numbered asset, determine if it contains ANY visible woman, girl, or female-presenting person.
Even a partial face, silhouette, or body counts as a "woman present".
Respond ONLY in this exact format — no other text:

ASSET_1: SAFE | WOMAN_PRESENT
ASSET_2: SAFE | WOMAN_PRESENT
...

Use one line per asset, numbered exactly as labelled in the grid."""

# 兼容模型输出的宽松变体：“ASSET_3 - WOMAN_PRESENT”、“asset 3 : safe” 等。
_VERDICT_RE = re.compile(
    r"ASSET[_\-\s]*(\d+)\s*[:：\-]\s*(SAFE|WOMAN[\s_]?PRESENT)",
    re.IGNORECASE,
)

# 判定结果按缩略图内容哈希缓存，避免相同素材在多个任务里重复消耗视觉请求。
_verdict_cache: dict[str, bool] = {}
_verdict_cache_lock = threading.Lock()


def is_enabled(app_config=None) -> bool:
    """
    判断视觉安全过滤是否可用。

    开关打开且配置了 Gemini API Key 时才启用；缺 Key 时静默降级，
    由第一道防线继续兜底。
    """
    runtime_config = app_config if app_config is not None else config.app
    if not runtime_config.get("vision_safety_filter_enabled", True):
        return False
    return bool(str(runtime_config.get("gemini_api_key", "") or "").strip())


def _resolve_vision_model(runtime_config: dict) -> str:
    """优先使用专用覆盖模型，否则复用 Gemini 主模型的注册默认值。"""
    override = str(runtime_config.get("vision_safety_model", "") or "").strip()
    if override:
        return override
    from app.models.llm_provider import get_llm_provider

    configured = str(runtime_config.get("gemini_model_name", "") or "").strip()
    provider = get_llm_provider("gemini")
    if provider is not None:
        return provider.resolve_model_name(configured)
    return configured


def _grid_size(runtime_config: dict) -> int:
    try:
        size = int(runtime_config.get("vision_safety_max_per_grid", MAX_ASSETS_PER_GRID))
    except (TypeError, ValueError):
        return MAX_ASSETS_PER_GRID
    return max(1, min(size, MAX_ASSETS_PER_GRID))


def _verdict_cache_key(blob: bytes) -> str:
    return hashlib.md5(blob).hexdigest()


def _cached_verdicts(keys: Iterable[str]) -> tuple[set[str], set[str]]:
    safe, unsafe = set(), set()
    with _verdict_cache_lock:
        for key in keys:
            verdict = _verdict_cache.get(key)
            if verdict is True:
                safe.add(key)
            elif verdict is False:
                unsafe.add(key)
    return safe, unsafe


def _store_verdicts(verdict_by_key: dict[str, bool]) -> None:
    with _verdict_cache_lock:
        _verdict_cache.update(verdict_by_key)


def _download_blob(url: str) -> bytes | None:
    import requests

    response = requests.get(
        url,
        timeout=_THUMBNAIL_TIMEOUT,
        verify=_get_tls_verify(),
        proxies=config.proxy,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            )
        },
        stream=True,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}")
    data = b""
    for chunk in response.iter_content(chunk_size=64 * 1024):
        data += chunk
        if len(data) > _THUMBNAIL_MAX_BYTES:
            raise RuntimeError("thumbnail exceeds size limit")
    return data or None


def _get_tls_verify() -> bool:
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")
    return bool(tls_verify)


def _download_thumbnails(
    entries: list[tuple[str, str]],
) -> dict[str, bytes]:
    """并发下载缩略图；单个失败只跳过该素材，不影响其它下载。"""
    results: dict[str, bytes] = {}
    if not entries:
        return results

    def _fetch(entry_key: str, url: str) -> tuple[str, bytes]:
        blob = _download_blob(url)
        if not blob:
            raise RuntimeError("empty thumbnail body")
        return entry_key, blob

    workers = min(_THUMBNAIL_WORKERS, len(entries))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, key, url): key for key, url in entries}
        for future in as_completed(futures):
            entry_key = futures[future]
            try:
                _, blob = future.result()
                results[entry_key] = blob
            except Exception as exc:
                logger.debug(
                    f"vision filter skipped thumbnail: key={entry_key}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
    return results


def build_composite_grid(labeled_blobs: list[tuple[str, bytes]]) -> bytes:
    """
    把若干缩略图合成为一张带编号标签的网格 PNG。

    每个单元格上方是按比例裁剪成正方形的缩略图，下方黑底白字标注
    ``ASSET_n``，与提示词要求的输出格式一一对应。
    """
    from PIL import Image, ImageDraw, ImageFont

    columns = min(_GRID_COLUMNS, len(labeled_blobs))
    rows = math.ceil(len(labeled_blobs) / columns)
    cell_width = _CELL_SIZE
    cell_height = _CELL_SIZE + _LABEL_BAR_HEIGHT
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(18)
    except TypeError:
        # 旧版 Pillow 的 load_default 不支持字号参数，退回内置点阵字体。
        font = ImageFont.load_default()

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", None)

    for index, (_, blob) in enumerate(labeled_blobs):
        column = index % columns
        row = index // columns
        origin_x = column * cell_width
        origin_y = row * cell_height

        thumb = Image.open(io.BytesIO(blob)).convert("RGB")
        width, height = thumb.size
        scale = max(_CELL_SIZE / max(width, 1), _CELL_SIZE / max(height, 1))
        resized = thumb.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resampling,
        )
        left = (resized.width - _CELL_SIZE) // 2
        top = (resized.height - _CELL_SIZE) // 2
        cropped = resized.crop(
            (left, top, left + _CELL_SIZE, top + _CELL_SIZE)
        )
        canvas.paste(cropped, (origin_x, origin_y))

        label_top = origin_y + _CELL_SIZE
        draw.rectangle(
            [origin_x, label_top, origin_x + cell_width - 1, origin_y + cell_height - 1],
            fill="black",
        )
        draw.text(
            (origin_x + 10, label_top + 5),
            f"ASSET_{index + 1}",
            fill="white",
            font=font,
        )
        draw.rectangle(
            [origin_x, origin_y, origin_x + cell_width - 1, origin_y + cell_height - 1],
            outline="#555555",
            width=2,
        )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _call_vision_model(grid_png: bytes, runtime_config: dict) -> str:
    """发送一次视觉审查请求，返回模型的原始文本响应。"""
    from google import genai
    from google.genai import types

    api_key = str(runtime_config.get("gemini_api_key", "") or "").strip()
    base_url = str(runtime_config.get("gemini_base_url", "") or "").strip()
    model_name = _resolve_vision_model(runtime_config)
    http_options = types.HttpOptions(base_url=base_url) if base_url else None

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=1,
        top_k=1,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
    )

    with genai.Client(api_key=api_key, http_options=http_options) as client:
        response = client.models.generate_content(
            model=model_name,
            contents=[SAFETY_PROMPT, types.Part.from_bytes(data=grid_png, mime_type="image/png")],
            config=generation_config,
        )
    return getattr(response, "text", "") or ""


def parse_safety_verdicts(response_text: str, expected_count: int) -> dict[int, bool]:
    """
    解析模型响应中的逐资产判定。

    返回 ``{序号: 是否安全}``；缺失或无法解析的条目不包含在结果中，
    由调用方按“未知即放行”处理。
    """
    verdicts: dict[int, bool] = {}
    for match in _VERDICT_RE.finditer(response_text or ""):
        try:
            index = int(match.group(1))
        except ValueError:
            continue
        token = re.sub(r"[^A-Z]", "", match.group(2).upper())
        if 1 <= index <= expected_count:
            verdicts[index] = token == "SAFE"
    return verdicts


def screen_image_blobs(
    blobs_by_key: dict[str, bytes],
    *,
    context: str = "",
    app_config=None,
) -> tuple[set[str], set[str]]:
    """
    对一组图片字节做女性人物筛查。

    参数是 ``{业务键: 图片字节}``（视频素材用缩略图、生成素材用首帧）。
    返回 ``(safe_keys, unsafe_keys)``；请求失败的批次整体按安全处理并告警，
    保证任何视觉链路故障都不会中断素材下载。
    """
    runtime_config = app_config if app_config is not None else config.app
    if not blobs_by_key:
        return set(), set()

    all_safe: set[str] = set()
    all_unsafe: set[str] = set()

    pending: dict[str, bytes] = {}
    cached_safe, cached_unsafe = _cached_verdicts(
        _verdict_cache_key(blob) for blob in blobs_by_key.values()
    )
    for key, blob in blobs_by_key.items():
        digest = _verdict_cache_key(blob)
        if digest in cached_unsafe:
            all_unsafe.add(key)
        elif digest in cached_safe:
            all_safe.add(key)
        else:
            pending[key] = blob

    if cached_safe or cached_unsafe:
        logger.info(
            f"vision safety cache reused: term={context!r}, "
            f"safe={len(cached_safe)}, rejected={len(cached_unsafe)}"
        )
    if not pending:
        return all_safe, all_unsafe

    grid_size = _grid_size(runtime_config)
    items = list(pending.items())
    for chunk_start in range(0, len(items), grid_size):
        chunk = items[chunk_start : chunk_start + grid_size]
        try:
            labeled = [
                (f"ASSET_{position}", blob)
                for position, (_, blob) in enumerate(chunk, start=1)
            ]
            grid_png = build_composite_grid(labeled)
            raw_response = _call_vision_model(grid_png, runtime_config)
            verdicts = parse_safety_verdicts(raw_response, len(chunk))
            for position, (key, blob) in enumerate(chunk, start=1):
                verdict = verdicts.get(position)
                if verdict is None:
                    # 模型漏答的条目按放行处理，同时记录便于排查提示词效果。
                    logger.warning(
                        f"vision filter returned no verdict, asset kept: "
                        f"term={context!r}, key={key}"
                    )
                    _store_verdicts({_verdict_cache_key(blob): True})
                    all_safe.add(key)
                elif verdict:
                    _store_verdicts({_verdict_cache_key(blob): True})
                    all_safe.add(key)
                else:
                    _store_verdicts({_verdict_cache_key(blob): False})
                    all_unsafe.add(key)
        except Exception as exc:
            # fail-open：这一批全部放行，仅记录告警。第一道元数据防线仍在。
            logger.warning(
                "vision safety screening failed, keeping assets: "
                f"term={context!r}, error={type(exc).__name__}, detail={exc}"
            )
            all_safe.update(key for key, _ in chunk)

    return all_safe, all_unsafe


def _material_thumbnail_url(item) -> str:
    source_info = item.source_info if isinstance(item.source_info, dict) else {}
    value = source_info.get("thumbnail_url")
    return value.strip() if isinstance(value, str) else ""


def filter_material_items(
    items: list,
    *,
    context: str = "",
    app_config=None,
) -> list:
    """
    过滤候选素材列表，返回确认无女性人物的子集。

    这是素材搜索后的统一入口：
    1. 未启用视觉过滤时原样返回；
    2. 缺少可下载缩略图的素材直接放行（由第一道元数据防线兜底）；
    3. 同一关键词的全部候选合并成一张网格图，一次视觉请求完成判定。
    """
    if not items:
        return []

    runtime_config = app_config if app_config is not None else config.app
    if not is_enabled(runtime_config):
        logger.debug(
            f"vision safety filter disabled, keeping all candidates: "
            f"term={context!r}, count={len(items)}"
        )
        return list(items)

    thumbnail_entries = []
    for item in items:
        url = _material_thumbnail_url(item)
        if url.startswith(("http://", "https://")):
            thumbnail_entries.append((item.url, url))
    downloaded = _download_thumbnails(thumbnail_entries)

    blobs_by_key: dict[str, bytes] = {}
    for item in items:
        blob = downloaded.get(item.url)
        if blob:
            blobs_by_key[item.url] = blob

    missing_thumbnails = len(items) - len(blobs_by_key)
    if missing_thumbnails:
        logger.info(
            f"{missing_thumbnails} candidate(s) have no downloadable thumbnail "
            f"and bypass vision screening: term={context!r}"
        )

    safe_urls, unsafe_urls = screen_image_blobs(
        blobs_by_key,
        context=context,
        app_config=runtime_config,
    )
    if unsafe_urls:
        preview = ", ".join(sorted(unsafe_urls)[:5])
        logger.warning(
            f"🚫 vision filter rejected {len(unsafe_urls)} asset(s) showing women: "
            f"term={context!r}, sample={preview}"
        )

    kept = [
        item
        for item in items
        if item.url not in unsafe_urls
    ]
    logger.success(
        f"👁️ vision safety screening done: term={context!r}, "
        f"checked={len(blobs_by_key)}, rejected={len(unsafe_urls)}, kept={len(kept)}"
    )
    return kept
