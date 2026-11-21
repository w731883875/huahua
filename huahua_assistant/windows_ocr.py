from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
import atexit
from collections import OrderedDict

import cv2
import numpy as np


_CACHE_LIMIT = 24
#: 子进程冷启动（加载 WinRT + OCR 语言包）的上限。实测 1~4s，超过就是没起来。
_READY_TIMEOUT = 6.0
# 编码格式：BMP 不压缩，5MB 的图编码只要 ~10ms；PNG 同样一张要 ~90ms，
# 而这段时间全卡在实时帧上（每帧最多两次 OCR）。Windows 的 BitmapDecoder
# 原生认 BMP，解码也不慢，所以统一走 BMP。
_IMAGE_EXT = ".bmp"
_line_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_cache_lock = threading.Lock()
_async_pending: set[str] = set()
_process_lock = threading.RLock()
_ocr_process: subprocess.Popen[str] | None = None
_warm_thread: threading.Thread | None = None
_AVAILABLE: bool | None = None
_probe_lock = threading.Lock()

_POWERSHELL_OCR = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
function Await($Operation, $ResultType) {
    $Method = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
    })[0]
    $Task = $Method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $Task.Wait()
    $Task.Result
}
$Engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $Engine) { exit 2 }
$ProgressPreference = 'SilentlyContinue'
[Console]::Out.WriteLine('__READY__')
[Console]::Out.Flush()
while ($true) {
    $ImagePath = [Console]::In.ReadLine()
    if ($null -eq $ImagePath -or $ImagePath -eq '__EXIT__') { break }
    try {
        $File = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
        $Stream = Await ($File.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $Decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($Stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $Bitmap = Await ($Decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $Result = Await ($Engine.RecognizeAsync($Bitmap)) ([Windows.Media.Ocr.OcrResult])
        $Lines = @($Result.Lines | ForEach-Object { $_.Text })
        [Console]::Out.WriteLine('__RESULT__' + ($Lines | ConvertTo-Json -Compress))
    } catch {
        [Console]::Out.WriteLine('__ERROR__')
    }
    [Console]::Out.Flush()
}
"""


def warm_windows_ocr_async() -> None:
    global _warm_thread
    if os.name != "nt":
        return
    if _ocr_process is not None and _ocr_process.poll() is None:
        return
    if _warm_thread is not None and _warm_thread.is_alive():
        return
    _warm_thread = threading.Thread(target=_warm_windows_ocr, name="huahua-ocr-warmup", daemon=True)
    _warm_thread.start()


def _warm_windows_ocr() -> None:
    with _process_lock:
        _ensure_ocr_process_locked()


def _readline_with_timeout(stream, timeout: float) -> str | None:
    """读一行，但**不许无限等**。

    这里原来是裸 ``stream.readline()``：``timeout`` 参数写进了签名却从没用上。
    一旦 PowerShell 那侧的 WinRT ``Await`` 卡住（冷启动加载 OCR 语言包、
    图太大、或者进程半死），这条 readline 就永远不返回，而它是在
    ``_process_lock`` 里调的 —— 后面的 ``read_panel`` / 笔触面板请求全部排队。

    实机表现就是「开局识别卡十秒」：一帧里唯一那次同步 OCR 挂住，
    采集泵 12 秒没出帧被判 stall 重启，重启后缓存热了才恢复。

    超时返回 ``None``，调用方据此杀掉子进程、这一帧当没读到。
    """

    box: dict[str, str] = {}

    def reader() -> None:
        try:
            box["line"] = stream.readline()
        except Exception:  # noqa: BLE001 - 进程被杀时 readline 会抛
            box["line"] = ""

    thread = threading.Thread(target=reader, name="huahua-ocr-read", daemon=True)
    thread.start()
    thread.join(max(0.05, float(timeout)))
    if thread.is_alive():
        return None
    return box.get("line")


def recognize_chinese_lines(image: np.ndarray, timeout: float = 2.5) -> tuple[str, ...]:
    """Use the built-in Windows OCR only for uncached, unfamiliar brush cards.

    ``timeout`` 是**单次往返**的上限（读不到就放弃这一帧），不是整帧预算：
    超时会让子进程被重建，避免一个坏进程拖死后面每一帧。
    """

    if os.name != "nt" or image.size == 0:
        return ()
    ok, encoded = cv2.imencode(_IMAGE_EXT, image)
    if not ok:
        return ()
    payload = encoded.tobytes()
    cache_key = hashlib.sha1(payload).hexdigest()
    with _cache_lock:
        cached = _line_cache.get(cache_key)
        if cached is not None:
            _line_cache.move_to_end(cache_key)
            return cached

    path = ""
    cacheable = True
    try:
        with tempfile.NamedTemporaryFile(
            prefix="huahua_ocr_", suffix=_IMAGE_EXT, delete=False
        ) as handle:
            handle.write(payload)
            path = handle.name
        with _process_lock:
            process = _ensure_ocr_process_locked()
            if process is None or process.stdin is None or process.stdout is None:
                lines = ()
            else:
                process.stdin.write(os.path.abspath(path) + "\n")
                process.stdin.flush()
                response = _readline_with_timeout(process.stdout, timeout)
                if response is None:
                    # 超时：进程状态未知，杀掉重建。这一帧当作没读到，
                    # 也**不能**把空结果写进缓存——否则同一张图永远读不出字。
                    _abort_ocr_process_locked()
                    lines = ()
                    cacheable = False
                    response = ""
                payload_text = response.strip().removeprefix("__RESULT__")
                if not response.strip().startswith("__RESULT__") or not payload_text:
                    # 一帧里没有任何文字时，PowerShell 的 ConvertTo-Json 会输出空串。
                    # 以前这里直接 json.loads("") 抛异常，把整帧识别带崩。
                    lines = ()
                else:
                    import json

                    try:
                        decoded = json.loads(payload_text)
                    except ValueError:
                        decoded = ()
                    if isinstance(decoded, str):
                        decoded = [decoded]
                    lines = tuple(str(line).strip() for line in decoded if str(line).strip())
    except (OSError, subprocess.SubprocessError):
        lines = ()
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    if cacheable:
        with _cache_lock:
            _line_cache[cache_key] = lines
            _line_cache.move_to_end(cache_key)
            while len(_line_cache) > _CACHE_LIMIT:
                _line_cache.popitem(last=False)
    return lines


def is_available() -> bool:
    """真跑一次 OCR 探针，判断这台机器/这个环境到底能不能用。

    WinRT 的 ``OcrEngine`` 在某些环境里恒失败（沙箱里 ``Await`` 直接抛异常，
    ``recognize_chinese_lines`` 于是静默返回 ``()``，和「这一帧没有文字」分不开）。
    依赖 OCR 真值的用例必须能区分这两种情况，否则只能长期挂红。

    探针自己画一行「调色效果」再去识别：读出任何一行就算可用。结果缓存。
    """

    global _AVAILABLE
    with _probe_lock:
        if _AVAILABLE is not None:
            return _AVAILABLE
        _AVAILABLE = False
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = ImageFont.truetype(_probe_font_path(), 28)
            canvas = Image.new("L", (220, 56), 255)
            ImageDraw.Draw(canvas).text((8, 8), "调色效果", fill=0, font=font)
            probe = cv2.cvtColor(np.array(canvas), cv2.COLOR_GRAY2BGR)
            _AVAILABLE = bool(recognize_chinese_lines(probe, timeout=8.0))
        except Exception:  # noqa: BLE001 - 探针失败只意味着不可用
            _AVAILABLE = False
        return _AVAILABLE


def _probe_font_path() -> str:
    from .keyword_match import _font_path

    return _font_path()


def recognize_chinese_lines_async(image: np.ndarray) -> tuple[str, ...] | None:
    """Return a cached result, or schedule OCR without blocking the frame pump."""

    if os.name != "nt" or image.size == 0:
        return ()
    ok, encoded = cv2.imencode(_IMAGE_EXT, image)
    if not ok:
        return ()
    cache_key = hashlib.sha1(encoded.tobytes()).hexdigest()
    with _cache_lock:
        cached = _line_cache.get(cache_key)
        if cached is not None:
            _line_cache.move_to_end(cache_key)
            return cached
        if cache_key in _async_pending:
            return None
        _async_pending.add(cache_key)
    copied = image.copy()
    threading.Thread(
        target=_complete_async_ocr,
        args=(cache_key, copied),
        name="huahua-ocr-request",
        daemon=True,
    ).start()
    return None


def _complete_async_ocr(cache_key: str, image: np.ndarray) -> None:
    try:
        recognize_chinese_lines(image)
    finally:
        with _cache_lock:
            _async_pending.discard(cache_key)


def _abort_ocr_process_locked() -> None:
    """把当前子进程判死并清空引用，下一次调用会重建。

    超时之后不能只把结果丢掉：那条请求还在管道的另一头，进程可能停在半路，
    继续用它会让后面每一次 OCR 都接在同一条坏管道上。
    """

    global _ocr_process
    process = _ocr_process
    _ocr_process = None
    if process is None:
        return
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=0.3)
    except (OSError, subprocess.SubprocessError):
        pass


def _ensure_ocr_process_locked() -> subprocess.Popen[str] | None:
    global _ocr_process
    if _ocr_process is not None and _ocr_process.poll() is None:
        return _ocr_process
    encoded_command = _POWERSHELL_OCR.encode("utf-16le")
    command = [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        __import__("base64").b64encode(encoded_command).decode("ascii"),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            encoding="utf-8",
            errors="replace",
        )
        # 冷启动要加载 WinRT + OCR 语言包，实测 1~4s；但**不能无限等**：
        # 加载失败时 powershell 会一直不回 __READY__，而这里持着全局锁。
        ready = (
            _readline_with_timeout(process.stdout, _READY_TIMEOUT)
            if process.stdout is not None
            else None
        )
        if ready is None or ready.strip() != "__READY__":
            try:
                process.kill()
            except OSError:
                pass
            return None
        _ocr_process = process
    except OSError:
        _ocr_process = None
    return _ocr_process


def reset_cache() -> None:
    """清空进程内的 OCR 结果缓存。

    实时链路靠这份缓存消除"首帧拿不到文案"的抖动，所以命中与否会改变
    ``analyze_scene_realtime`` 的返回值。测试要断言"冷状态下的第一帧行为"时，
    必须先把缓存清掉，否则结果取决于上一个用例后台 OCR 线程是否恰好已经跑完。
    """

    with _cache_lock:
        _line_cache.clear()
        _async_pending.clear()


def _shutdown_ocr_process() -> None:
    global _ocr_process
    with _process_lock:
        process = _ocr_process
        _ocr_process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write("__EXIT__\n")
                process.stdin.flush()
            process.wait(timeout=1.0)
        except (OSError, subprocess.SubprocessError):
            process.terminate()


atexit.register(_shutdown_ocr_process)
