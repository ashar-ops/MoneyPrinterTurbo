import os
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

# 终端日志采用三段式布局：时间 │ 级别徽章 │ 位置 › 消息。
# - 时间与位置使用暗色/青色弱化，突出消息本体；
# - 级别通过 <level> 标签自动按严重度着色（DEBUG 灰、INFO 蓝、SUCCESS 绿、
#   WARNING 黄、ERROR 红），并带图标前缀，一眼可分辨动作类型；
# - 消息同样套用 <level>，让整行保持级别颜色语义。
# WebUI 复用同一格式但关闭 colorize，标记会被自动剥离为纯文本。
LOG_RECORD_FORMAT = (
    "<dim>{time:%Y-%m-%d %H:%M:%S}</> <dim>│</> "
    "<level>{level.icon}</> <level>{level: <7}</><dim>│</> "
    "<cyan>{file.path}:{line}</> <dim>{function}()</> <blue>›</> "
    "<level>{message}</>\n"
)
# 流水线阶段横幅宽度。任务日志里每个阶段用醒目的分隔块标出，
# 方便在大量素材下载/合成日志中快速定位当前执行到哪一步。
_STAGE_BAR = "━" * 64

# Loguru 启动时默认终端 handler 的 ID 为 0。WebUI 重新加载时只能替换这个
# 基础终端输出，不能调用 logger.remove() 清空全部 handler，否则正在运行任务
# 用于收集 WebUI 日志的临时 sink 也会被删除。
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()


def format_log_record(record):
    """
    统一格式化终端与 WebUI 日志。

    Loguru 会把同一条记录交给多个 sink。第一个 sink 可能已经将绝对路径转换
    为项目相对路径，因此这里同时兼容绝对路径和 ``./`` 开头的已格式化路径。
    WebUI sink 会关闭颜色，但时间、级别、调用位置和消息内容与终端保持一致。
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
        record["file"].path = f"./{relative_path}"

    # 日志消息有时会包含任务文件的绝对路径。统一缩短为项目相对路径，可以
    # 避免 WebUI 和终端因初始化入口不同而展示两套内容。
    record["message"] = record["message"].replace(PROJECT_ROOT, ".")
    return LOG_RECORD_FORMAT


def log_stage(title: str, icon: str = "🎬") -> None:
    """
    输出流水线阶段横幅。

    任务编排层的各阶段（脚本、配音、字幕、素材、合成……）都用这个入口打头，
    让终端日志呈现清晰的分段结构；颜色标记在非彩色 sink 中会被剥离。
    """
    header = f"{icon + '  ' if icon else ''}{title}".strip()
    logger.opt(depth=1).info(
        f"\n<bold><cyan>{_STAGE_BAR}</></>\n"
        f"<bold><cyan>▶ {header}</></>\n"
        f"<bold><cyan>{_STAGE_BAR}</></>"
    )


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    安全替换进程级终端日志 handler，并保留任务专用 handler。

    Streamlit 在代码热重载或缓存失效时可能重新执行日志初始化。这里只按已记录
    的 handler ID 精确移除旧终端输出，因此不会中断后台任务正在写入的 WebUI
    日志。锁用于保护多个浏览器会话同时初始化时的 ID 更新。
    """
    global _terminal_handler_id

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # 测试或外部入口可能已经移除该 handler。继续创建新的终端输出，
                # 不需要影响其它仍有效的日志 sink。
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=level,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
