"""程序入口：先声明 DPI 感知，再装配桌宠扩展，最后进主循环。

崩溃取证：``pythonw.exe`` 没有控制台，启动阶段抛出的任何异常都无处可去，
用户看到的就是「双击了，任务栏里什么都没有」。所以这里把整条启动链包起来，
traceback 落盘到 ``runtime/crash.log``，出问题能直接看。

Tk 回调里的异常另有一处：``HuahuaApp.report_callback_exception``。
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

CRASH_LOG = Path("runtime/crash.log")


def log_crash(exc_type, exc_value, exc_tb) -> None:
    """把异常追加到 runtime/crash.log；日志写不进去也不能连累进程。"""

    try:
        CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CRASH_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            handle.write(f"python {sys.version.split()[0]}  exe={sys.executable}\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=handle)
            handle.write("\n")
    except OSError:
        pass


def boot() -> None:
    from huahua_assistant.capture import enable_process_dpi_awareness

    enable_process_dpi_awareness()

    from huahua_assistant.app import HuahuaApp, main
    from huahua_assistant.desktop_pet import install_desktop_pet_markers

    install_desktop_pet_markers(HuahuaApp)
    main()


sys.excepthook = log_crash

if __name__ == "__main__":
    try:
        boot()
    except BaseException:
        log_crash(*sys.exc_info())
        raise
