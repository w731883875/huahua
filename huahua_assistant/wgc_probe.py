"""窗口内容抓取（WGC）失效诊断：把「为什么抓不到」变成可读的结论。

背景（2026-11-16 实测）：`windows_capture` 只抛一句
``Failed to convert item to `GraphicsCaptureItem` ``，没有 HRESULT、没有阶段，
看起来像「这台机器的 WGC 坏了」。用 ctypes 直接走官方链路后真相是：

* ``CreateForWindow`` 对**任何**句柄都返回 ``0x80070005 E_ACCESSDENIED``，
  连无效句柄 ``0x1234`` 和 ``CreateForMonitor`` 也一样 → 拒绝发生在**进程**侧，
  跟游戏窗口、DirectX、窗口样式都无关；
* 该进程的完整性级别是 **LOW**（``RID=0x1000``）。WorkBuddy 的工具调用会把子
  进程降到 LOW，于是用它 ``pythonw run_app.py`` 拉起来的软件必然抓不到窗口内容。

正常路径（用户双击 ``启动.bat``，由资源管理器拉起）是 MEDIUM，WGC 可用。
所以诊断的第一句话永远是「我这个进程是什么完整性级别」。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

#: 完整性级别的 RID。低于 MEDIUM 的进程无法创建 WGC 采集项。
INTEGRITY_NAMES = {
    0x0000: "UNTRUSTED",
    0x1000: "LOW",
    0x2000: "MEDIUM",
    0x2100: "MEDIUM_PLUS",
    0x3000: "HIGH",
    0x4000: "SYSTEM",
    0x5000: "PROTECTED",
}
INTEGRITY_MEDIUM = 0x2000

#: ``TOKEN_INFORMATION_CLASS`` 里 ``TokenIntegrityLevel`` 的序号在各 SDK 里
#: 写法不一致，这里两个候选都试，认第一个能读出合法 SID 的。
_INTEGRITY_CLASS_CANDIDATES = (25, 19)

GA_ROOT = 2

CLASS_GRAPHICS_CAPTURE_ITEM = "Windows.Graphics.Capture.GraphicsCaptureItem"


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __str__(self) -> str:
        tail = "".join(f"{byte:02X}" for byte in self.Data4)
        return f"{{{self.Data1:08X}-{self.Data2:04X}-{self.Data3:04X}-{tail[:4]}-{tail[4:]}}}"


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", ctypes.c_uint32)]


class _TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", _SID_AND_ATTRIBUTES)]


_IID_IActivationFactory = _GUID(0x00000035, 0x0000, 0x0000, (ctypes.c_ubyte * 8)(0xC0, 0, 0, 0, 0, 0, 0, 0x46))
_IID_IGraphicsCaptureItemInterop = _GUID(
    0x3628E81B, 0x3CAC, 0x4C60, (ctypes.c_ubyte * 8)(0xB7, 0xF4, 0x23, 0xCE, 0x0E, 0x0C, 0x33, 0x56)
)
_IID_IGraphicsCaptureItem = _GUID(
    0x79C3F95B, 0x31F7, 0x4EC2, (ctypes.c_ubyte * 8)(0xA4, 0x64, 0x63, 0x2E, 0xF5, 0xD3, 0x07, 0x60)
)

_HRESULT_TEXT = {
    0x00000000: ("S_OK", "成功"),
    0x80004005: ("E_FAIL", "未指明的失败"),
    0x80070005: ("E_ACCESSDENIED", "拒绝访问：调用进程权限不够（LOW 完整性最常见）"),
    0x80070057: ("E_INVALIDARG", "参数无效：句柄/目标不对"),
    0x80070490: ("ERROR_NOT_FOUND", "找不到元素：句柄不是可采集的窗口"),
    0x80040154: ("REGDB_E_CLASSNOTREG", "Windows 版本太老，没有这个 API"),
    0x80004002: ("E_NOINTERFACE", "接口不可用"),
    0x80070006: ("E_HANDLE", "句柄无效"),
    0x800700AA: ("ERROR_BUSY", "目标忙（独占全屏 / 受保护内容）"),
}


def hresult_text(hr: int) -> str:
    code = hr & 0xFFFFFFFF
    name, hint = _HRESULT_TEXT.get(code, ("", ""))
    return f"0x{code:08X} {name} {hint}".strip()


def process_integrity() -> dict[str, object]:
    """当前进程的完整性级别（WGC 的第一道门）。"""

    result: dict[str, object] = {"rid": None, "name": "UNKNOWN", "restricted": None, "elevation_type": None}
    kernel32 = ctypes.windll.kernel32
    advapi32 = ctypes.windll.advapi32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    advapi32.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
    advapi32.GetTokenInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
    advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    advapi32.GetSidSubAuthority.argtypes = (ctypes.c_void_p, wintypes.DWORD)
    advapi32.GetSidSubAuthority.restype = ctypes.POINTER(ctypes.c_uint32)
    advapi32.IsTokenRestricted.argtypes = (wintypes.HANDLE,)

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(wintypes.HANDLE(kernel32.GetCurrentProcess()), 0x0008, ctypes.byref(token)):
        return result
    try:
        try:
            result["restricted"] = bool(advapi32.IsTokenRestricted(token))
        except Exception:
            pass
        elevation = wintypes.DWORD()
        if advapi32.GetTokenInformation(token, 18, ctypes.byref(elevation), ctypes.sizeof(elevation), ctypes.byref(wintypes.DWORD())):
            # 1 = default, 2 = full（提权）, 3 = limited（UAC 过滤后的管理员）
            result["elevation_type"] = int(elevation.value)
        for index in _INTEGRITY_CLASS_CANDIDATES:
            buffer = ctypes.create_string_buffer(256)
            size = wintypes.DWORD(256)
            if not advapi32.GetTokenInformation(token, index, buffer, 256, ctypes.byref(size)):
                continue
            label = ctypes.cast(buffer, ctypes.POINTER(_TOKEN_MANDATORY_LABEL))[0]
            sid = label.Label.Sid
            if not sid:
                continue
            count = advapi32.GetSidSubAuthorityCount(sid)
            if not count or count[0] == 0:
                continue
            rid = int(advapi32.GetSidSubAuthority(sid, int(count[0]) - 1)[0])
            if rid not in INTEGRITY_NAMES and not (0x1000 <= rid <= 0x5000):
                continue
            result["rid"] = rid
            result["name"] = INTEGRITY_NAMES.get(rid, f"RID=0x{rid:x}")
            break
    finally:
        kernel32.CloseHandle(token)
    return result


def integrity_allows_window_capture() -> bool:
    rid = process_integrity().get("rid")
    if rid is None:
        # 读不到就别拦：宁可让 WGC 自己报错，也不要因为诊断失败而禁用采集。
        return True
    return int(rid) >= INTEGRITY_MEDIUM


def _com_method(ptr: int, index: int, *argtypes):
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _make_hstring(text: str):
    combase = ctypes.windll.combase
    combase.WindowsCreateString.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
    combase.WindowsCreateString.restype = ctypes.c_long
    handle = ctypes.c_void_p()
    if combase.WindowsCreateString(text, len(text), ctypes.byref(handle)) != 0:
        return None
    return handle


def _capture_item_interop():
    """拿到 ``IGraphicsCaptureItemInterop``；失败抛 OSError（带 HRESULT 文本）。"""

    combase = ctypes.windll.combase
    combase.RoInitialize.argtypes = (ctypes.c_int,)
    combase.RoInitialize.restype = ctypes.c_long
    # S_FALSE（已初始化）与 S_OK 都算成功；RPC_E_CHANGED_MODE 也不该拦。
    combase.RoInitialize(1)  # RO_INIT_MULTITHREADED

    combase.RoGetActivationFactory.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    combase.RoGetActivationFactory.restype = ctypes.c_long

    hstr = _make_hstring(CLASS_GRAPHICS_CAPTURE_ITEM)
    if hstr is None:
        raise OSError("WindowsCreateString failed")
    factory = ctypes.c_void_p()
    hr = combase.RoGetActivationFactory(hstr, ctypes.byref(_IID_IActivationFactory), ctypes.byref(factory))
    if hr != 0 or not factory:
        raise OSError(f"RoGetActivationFactory {hresult_text(hr)}")

    interop = ctypes.c_void_p()
    hr = _com_method(factory.value, 0, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))(
        factory, ctypes.byref(_IID_IGraphicsCaptureItemInterop), ctypes.byref(interop)
    )
    if hr != 0 or not interop:
        raise OSError(f"QueryInterface(IGraphicsCaptureItemInterop) {hresult_text(hr)}")
    return interop


def probe_create_for_window(hwnd: int) -> int | None:
    """直接问系统：这个句柄能不能建采集项。返回 HRESULT。"""

    try:
        interop = _capture_item_interop()
    except Exception:
        return None
    item = ctypes.c_void_p()
    try:
        hr = _com_method(
            interop.value,
            3,
            wintypes.HWND,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )(interop, wintypes.HWND(int(hwnd)), ctypes.byref(_IID_IGraphicsCaptureItem), ctypes.byref(item))
    except Exception:
        return None
    return int(hr)


def diagnose(hwnds: dict[str, int] | None = None) -> dict[str, object]:
    """一次跑完所有判据。``hwnds`` 是「标签 → 句柄」，用于逐个试 CreateForWindow。"""

    integrity = process_integrity()
    payload: dict[str, object] = {
        "pid": ctypes.windll.kernel32.GetCurrentProcessId(),
        "integrity": integrity,
        "wgc_integrity_ok": bool(integrity.get("rid") is None or int(integrity["rid"]) >= INTEGRITY_MEDIUM),
        "items": {},
    }
    targets = dict(hwnds or {})
    # 无效句柄是关键的对照：权限被拒时它对**任何**句柄都返回 E_ACCESSDENIED，
    # 而正常进程这里应当是 E_INVALIDARG / ERROR_NOT_FOUND。
    targets.setdefault("bogus-0x1234", 0x1234)
    results: dict[str, dict[str, object]] = {}
    for label, hwnd in targets.items():
        hr = probe_create_for_window(hwnd)
        results[label] = {
            "hwnd": hwnd,
            "hresult": None if hr is None else f"0x{hr & 0xFFFFFFFF:08X}",
            "text": None if hr is None else hresult_text(hr),
            "ok": hr == 0,
        }
    payload["items"] = results

    denied = [item for item in results.values() if item["hresult"] == "0x80070005"]
    if not payload["wgc_integrity_ok"]:
        verdict = "low_integrity"
        advice = "当前进程是 LOW 完整性（被自动化环境拉起），WGC 必然被拒；请双击 启动.bat 从资源管理器启动。"
    elif results and len(denied) == len(results):
        verdict = "access_denied"
        advice = "所有目标（含无效句柄）都被拒绝：进程级权限问题，不是游戏窗口的问题。"
    elif any(item["ok"] for item in results.values()):
        verdict = "ready"
        advice = "至少一个目标可以建采集项，WGC 可用。"
    else:
        verdict = "unavailable"
        advice = "系统拒绝建采集项，退回屏幕复制（工具窗口已对截屏免疫，不挡画面）。"
    payload["verdict"] = verdict
    payload["advice"] = advice
    return payload


def format_report(payload: dict[str, object]) -> str:
    lines = [f"进程 pid={payload.get('pid')}"]
    integrity = payload.get("integrity") or {}
    lines.append(
        "完整性级别: {name} (RID=0x{rid:x})  restricted={restricted}  elevation={elevation}".format(
            name=integrity.get("name"),
            rid=int(integrity.get("rid") or 0),
            restricted=integrity.get("restricted"),
            elevation=integrity.get("elevation_type"),
        )
    )
    lines.append(f"判定: {payload.get('verdict')} —— {payload.get('advice')}")
    for label, item in (payload.get("items") or {}).items():
        lines.append(f"  CreateForWindow[{label}] hwnd={item['hwnd']} -> {item['text']}")
    return "\n".join(lines)
