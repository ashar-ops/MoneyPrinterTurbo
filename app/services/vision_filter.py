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
import os
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

SAFETY_PROMPT = """You are an accurate visual content classifier. The grid shows frames sampled from stock videos.
For EACH numbered asset, determine if any woman, girl, or female person is visible in the frame (face, body, or clear silhouette).
If the asset contains nature, objects, technology, scenery, animals, abstract visuals, or non-female footage, classify it as SAFE.
There are exactly two allowed answers per asset, and nothing else:

ASSET_n: SAFE            # no woman/girl/female person visible (scenery, objects, technology, animals, abstract, etc.)
ASSET_n: WOMAN_PRESENT   # a woman, girl, or female person is clearly visible

Output ONLY the lines above, one per asset, numbered exactly as labelled."""

# 兼容模型输出的宽松变体：“**ASSET_3**: WOMAN_PRESENT”、“ASSET_3 - WOMAN PRESENT” 等。
_VERDICT_RE = re.compile(
    r"(?:\*\*|\b)ASSET[_\-\s]*(\d+)(?:\*\*|\b)?\s*[:：\-]\s*(?:\*\*)?\s*(SAFE|WOMAN[\s_]?PRESENT|FEMALE[\s_]?PRESENT|UNSAFE)",
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


# 已下载视频文件（按内容哈希）的判定缓存，避免同一份缓存文件在一次运行内
# 被反复抽帧筛查。键是文件内容的 md5，值是是否安全（True=可用）。
_file_verdict_cache: dict[str, bool] = {}


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _store_file_verdict(file_digest: str, safe: bool) -> None:
    if not file_digest:
        return
    with _verdict_cache_lock:
        _file_verdict_cache[file_digest] = safe


def _try_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.warning(f"failed to remove rejected clip: {path}, error={exc}")


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
            logger.info(
                f"👁️ vision pre-filter: scanning {len(chunk)} thumbnail(s) "
                f"for women · term={context!r}"
            )
            raw_response = _call_vision_model(grid_png, runtime_config)
            logger.debug(
                f"vision model raw response (thumbnails, term={context!r}): {raw_response}"
            )
            verdicts = parse_safety_verdicts(raw_response, len(chunk))
            for position, (key, blob) in enumerate(chunk, start=1):
                verdict = verdicts.get(position)
                if verdict is None:
                    # 模型漏答的条目按放行处理，同时记录便于排查提示词效果。
                    logger.warning(
                        f"   🟡 ASSET_{position} → UNCLEAR (kept for frame recheck) · {key}"
                    )
                    _store_verdicts({_verdict_cache_key(blob): True})
                    all_safe.add(key)
                elif verdict:
                    logger.info(f"   ✅ ASSET_{position} → SAFE · {key}")
                    _store_verdicts({_verdict_cache_key(blob): True})
                    all_safe.add(key)
                else:
                    logger.warning(
                        f"   🚫 ASSET_{position} → WOMAN_PRESENT (rejected) · {key}"
                    )
                    _store_verdicts({_verdict_cache_key(blob): False})
                    all_unsafe.add(key)
        except Exception as exc:
            # fail-open：这一批全部放行，仅记录告警。第一道元数据防线仍在。
            logger.warning(
                "vision safety screening failed, keeping assets: "
                "term={!r}, error={}: {}",
                context,
                type(exc).__name__,
                exc,
                exc_info=False,
            )
            all_safe.update(key for key, _ in chunk)

    return all_safe, all_unsafe


# ---------------------------------------------------------------------------
# 第三道防线（权威闸门）· 对真实视频抽帧做视觉筛查
#
# 缩略图只是静态海报，很多女性人物出现在动态画面里，单看海报会被漏掉；而且
# 旧缓存文件、local_videos 目录里的素材根本没有缩略图可下载。所以真正的“是否
# 进成片”判定必须落在“真实视频帧”上。每个候选片段只抽 **一帧** 即可：素材片段
# 都很短、只有一个场景，任何一帧都足以判断是否出现女性，没必要把整段视频逐帧
# 发给模型。策略是 fail-closed（严格）：模型判为女性、答非所问、抽帧失败或接口
# 报错，就判定该片段不可用并删除，绝不让未经验证的内容进入成片。
# ---------------------------------------------------------------------------
def _extract_single_frame_png(video_path: str, max_dim: int = 480) -> bytes | None:
    """
    从视频里抽取一帧（取中段，避开片头黑场）编码为 PNG 字节。

    素材片段通常只有一个场景，单帧足以判断是否含女性人物。返回 ``None`` 表示
    抽帧失败，由调用方按“无法验证即不可用”处理。
    """
    from PIL import Image

    from moviepy.video.io.VideoFileClip import VideoFileClip

    clip = None
    try:
        clip = VideoFileClip(video_path)
        if not clip.duration or clip.duration <= 0:
            return None
        # 取中段帧：避开片头/片尾可能的黑场或淡入淡出。
        timestamp = min(clip.duration / 2, max(0.0, clip.duration - 0.01))
        frame = clip.get_frame(timestamp)
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((max_dim, max_dim))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:
        logger.warning(
            "failed to extract a frame for vision screening: "
            f"video={os.path.basename(video_path)}, error={type(exc).__name__}, detail={exc}"
        )
        return None
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def parse_safe_indices(response_text: str, expected_count: int) -> set[int]:
    """
    解析严格二分类响应，只返回“明确判定为 SAFE”的资产序号集合。

    没有出现的序号、判为 WOMAN_PRESENT 的、或任何无法解析的内容都不在集合内，
    由调用方统一视作“不可用”。
    """
    safe: set[int] = set()
    for match in _VERDICT_RE.finditer(response_text or ""):
        try:
            index = int(match.group(1))
        except ValueError:
            continue
        token = re.sub(r"[^A-Z]", "", match.group(2).upper())
        if 1 <= index <= expected_count and token == "SAFE":
            safe.add(index)
    return safe


def screen_video_file(
    video_path: str,
    context: str = "",
    *,
    app_config=None,
    delete_unsafe: bool = False,
) -> bool:
    """
    对单个已下载视频做严格视觉安全筛查，返回是否可安全使用。

    这是“是否进入成片”的权威闸门：必须启用视觉过滤且模型明确判为 SAFE 才返回
    True；模型未启用、文件缺失、抽帧失败、响应无法解析或接口报错都返回 False
    （fail-closed）。``delete_unsafe=True`` 时，不可用片段会被直接删除（用于清理
    我们自己的下载缓存；本地用户素材默认只剔除不删除）。
    """
    runtime_config = app_config if app_config is not None else config.app
    if not is_enabled(runtime_config):
        # 用户要求“没有 AI 检查就绝不把片段加进成片”，因此视觉过滤不可用时必须
        # 拒绝一切片段，而不是退回到无防护状态。
        logger.critical(
            "vision safety is REQUIRED but disabled (missing gemini_api_key); "
            f"refusing to use unverified clip: {video_path}"
        )
        if delete_unsafe:
            _try_remove(video_path)
        return False

    if not video_path or not os.path.exists(video_path):
        return False

    file_digest = _file_md5(video_path)
    with _verdict_cache_lock:
        cached = _file_verdict_cache.get(file_digest)
    if cached is not None:
        return cached

    frame = _extract_single_frame_png(video_path)
    if not frame:
        logger.warning(
            f"🚫 vision rejected clip (no frame extracted): context={context!r}, "
            f"file={os.path.basename(video_path)}"
        )
        if delete_unsafe:
            _try_remove(video_path)
        _store_file_verdict(file_digest, False)
        return False

    labeled = [("ASSET_1", frame)]
    file_name = os.path.basename(video_path)
    logger.info(
        f"🔍 vision gate: scanning 1 frame of clip for women · "
        f"context={context!r} · file={file_name}"
    )
    try:
        grid_png = build_composite_grid(labeled)
        raw_response = _call_vision_model(grid_png, runtime_config)
        logger.debug(
            f"vision model raw response (frames, context={context!r}): {raw_response}"
        )
        safe_indices = parse_safe_indices(raw_response, 1)
        if 1 in safe_indices:
            logger.info("   ✅ ASSET_1 → SAFE")
        else:
            logger.warning("   🚫 ASSET_1 → WOMAN_PRESENT / UNCLEAR")
        if 1 not in safe_indices:
            logger.warning(
                "🚫 clip REJECTED (woman present or unclear): "
                "context={!r}, file={}".format(context, file_name)
            )
            if delete_unsafe:
                _try_remove(video_path)
            _store_file_verdict(file_digest, False)
            return False
        logger.success(
            "✅ clip PASSED vision safety: context={!r}, file={}".format(context, file_name)
        )
        _store_file_verdict(file_digest, True)
        return True
    except Exception as exc:
        # 接口故障也按“不可用”处理，宁可少一段素材也不能把未验证内容放进成片。
        # 异常对象可能带有花括号（如 API 错误 JSON），必须用参数传入以避免
        # loguru 把消息里的花括号误当格式字段而崩溃。
        logger.critical(
            "vision screening errored, rejecting clip (fail-closed): "
            "context={!r}, file={}, error={}: {}",
            context,
            file_name,
            type(exc).__name__,
            exc,
            exc_info=False,
        )
        if delete_unsafe:
            _try_remove(video_path)
        _store_file_verdict(file_digest, False)
        return False


def screen_video_paths(
    paths: list[str],
    context: str = "",
    *,
    delete_unsafe: bool = False,
) -> list[str]:
    """对一批视频路径逐个严格筛查，返回通过（可安全使用）的路径列表。"""
    logger.info(
        f"🔍 AI VISION SAFETY GATE · batch screening {len(paths)} clip(s) "
        f"frame-by-frame for women · context={context!r}"
    )
    safe_paths: list[str] = []
    for path in paths:
        if screen_video_file(path, context, delete_unsafe=delete_unsafe):
            safe_paths.append(path)
        else:
            logger.info(f"clip excluded by vision safety: {path}")
    logger.success(
        f"✅ vision batch done: context={context!r}, "
        f"passed={len(safe_paths)}/{len(paths)}"
    )
    return safe_paths


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
        # 没有 AI 检查就绝不把片段加进成片：视觉过滤不可用时直接清空候选，
        # 由上层因“无可用素材”而失败，而不是退回无防护状态。
        logger.critical(
            "vision safety filter is REQUIRED but disabled (missing gemini_api_key); "
            f"refusing all {len(items)} candidates for {context!r}"
        )
        return []

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


# ---------------------------------------------------------------------------
# 运维工具：清理已落盘的“脏”缓存、以及对视觉模型做连通性自检
# ---------------------------------------------------------------------------
_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")


def sweep_directory(directory: str, context: str = "sweep") -> tuple[int, int]:
    """
    扫描目录里的视频文件并严格筛查，删除被判为女性的片段。

    用于清理历史遗留的 ``cache_videos`` / ``local_videos`` 目录——这些文件可能在
    加入防护前就已经存在，且会被反复复用。返回 ``(扫描数, 删除数)``。
    """
    if not directory or not os.path.isdir(directory):
        logger.warning(f"sweep skipped, directory not found: {directory}")
        return 0, 0
    checked = 0
    removed = 0
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(_VIDEO_SUFFIXES):
            continue
        path = os.path.join(directory, name)
        checked += 1
        if not screen_video_file(path, context, delete_unsafe=True):
            removed += 1
    logger.info(
        f"vision sweep finished: directory={directory}, "
        f"checked={checked}, removed={removed}"
    )
    return checked, removed


def self_check() -> bool:
    """用一张纯色图自检视觉模型是否可用且返回预期格式。"""
    from PIL import Image

    runtime_config = config.app
    if not is_enabled(runtime_config):
        logger.error("self-check FAILED: vision safety is disabled (missing gemini_api_key)")
        return False
    image = Image.new("RGB", (240, 240), (200, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    blob = buffer.getvalue()
    labeled = [("ASSET_1", blob)]
    try:
        grid_png = build_composite_grid(labeled)
        raw = _call_vision_model(grid_png, runtime_config)
        safe = parse_safe_indices(raw, 1)
        logger.info(f"self-check model response: {raw!r}")
        if 1 in safe:
            logger.success("self-check OK: vision model reachable and returned SAFE")
            return True
        logger.error("self-check FAILED: model did not return expected SAFE verdict")
        return False
    except Exception as exc:
        logger.error(
            "self-check FAILED: vision model error: {}: {}",
            type(exc).__name__,
            exc,
            exc_info=False,
        )
        return False


def _main(argv: list[str]) -> int:

    from app.utils import utils

    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "check":
        return 0 if self_check() else 1
    if cmd == "sweep":
        target = argv[2] if len(argv) > 2 else "cache"
        if target in ("cache", "cache_videos"):
            directory = utils.storage_dir("cache_videos")
        elif target in ("local", "local_videos"):
            directory = utils.storage_dir("local_videos")
        else:
            directory = target
        checked, removed = sweep_directory(directory)
        logger.info(f"sweep complete: checked={checked}, removed={removed}")
        return 0
    logger.error("usage: python -m app.services.vision_filter [check|sweep [cache|local|<dir>]]")
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
