"""Keep one tracker alive while the game is resized; save geometry evidence.

Run from the repository: python -B scripts/verify_adaptive_layout.py --seconds 120
The script only captures and analyzes. It never clicks or changes game state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huahua_assistant.capture import (capture_target, close_capture_backends,
                                      enable_process_dpi_awareness, find_capture_window, write_image)
from huahua_assistant.frame_pump import FramePump
from huahua_assistant.vision import VisionTracker, make_debug_image, set_layout_autosave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("runtime/adaptive_verification/live"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    enable_process_dpi_awareness()
    set_layout_autosave(False)
    tracker = VisionTracker()
    pump = FramePump(capture_func=lambda:capture_target(fallback_to_screen=False, activate=False),
                     analyze_func=tracker.analyze_quick, needs_full_refresh=lambda:False,
                     full_analyze_func=tracker.analyze_realtime,
                     prepare_capture_func=tracker.set_capture_context, geometry_probe=find_capture_window)
    end = time.monotonic() + args.seconds
    last_size = None
    try:
        with (args.output / "measurements.jsonl").open("w", encoding="utf-8") as log:
            while time.monotonic() < end:
                result = pump._capture_and_analyze()
                scene = result.scene
                if scene is not None:
                    window = result.capture.window
                    key = (result.frame_shape, None if window is None else (window.left, window.top))
                    if key != last_size:
                        last_size = key
                        name = f"frame_{result.sequence:04d}"
                        write_image(args.output / (name + ".png"), result.capture.image)
                        write_image(args.output / (name + "_debug.png"), make_debug_image(result.capture.image, scene))
                        data = {"pid":os.getpid(), "sequence":result.sequence,
                                "window":None if window is None else asdict(window),
                                "shape":result.frame_shape, "seconds":round(result.elapsed,3),
                                "source":result.capture.source, "layout":scene.layout,
                                "board":asdict(scene.board.rect),
                                "cards":[asdict(r) for r in getattr(scene.brush_options,"card_rects",())],
                                "options":getattr(scene.brush_options,"stage1_options",()),
                                "image":name+"_debug.png"}
                        log.write(json.dumps(data,ensure_ascii=False)+"\n");log.flush()
                        print(json.dumps(data,ensure_ascii=False),flush=True)
                elif result.error:
                    print(result.error,flush=True)
                time.sleep(.25)
    finally:
        close_capture_backends()


if __name__ == "__main__":
    main()
