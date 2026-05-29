from __future__ import annotations

import argparse
from pathlib import Path

from .capture import DEFAULT_WINDOW_TITLE, capture_target, list_visible_windows, read_image, write_image
from .game_model import (
    BrushType,
    PuzzleState,
    ResourceState,
    current_scope_cells,
    normalize_brush_options,
    random_brush_choice_scenario,
    random_state,
    recommend_affixes,
    recommend_brush_types,
    recommend_install_plan,
    recommend_placements,
    recommend_special_actions,
)
from .models import SceneObservation
from .vision import analyze_and_save_debug


def main() -> None:
    parser = argparse.ArgumentParser(prog="huahua-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze an existing screenshot.")
    analyze.add_argument("image", type=Path)
    analyze.add_argument("--debug", type=Path, default=Path("runtime/debug.png"))

    capture = subparsers.add_parser("capture", help="Capture the primary screen and analyze it.")
    capture.add_argument("--image", type=Path, default=Path("runtime/capture.png"))
    capture.add_argument("--debug", type=Path, default=Path("runtime/debug.png"))
    capture.add_argument("--window", default=DEFAULT_WINDOW_TITLE)
    capture.add_argument("--no-fallback", action="store_true")
    capture.add_argument(
        "--fallback-uncaptureable",
        action="store_true",
        help="Fallback to full-screen capture when a matched game window is minimized or off-screen.",
    )

    subparsers.add_parser("windows", help="List visible windows for capture debugging.")

    simulate = subparsers.add_parser("simulate", help="Generate a random puzzle and print recommendations.")
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument("--brush", action="store_true", help="Also generate random brush choices and rank them.")
    simulate.add_argument("--refresh", action="store_true", help="Enable Lv5 pet affix refresh evaluation.")
    simulate.add_argument("--force-used", type=int, default=0, choices=[0, 1, 2, 3])

    brush_order = subparsers.add_parser("brush-order", help="Normalize stage-1 brush option order.")
    brush_order.add_argument("options", nargs="+", choices=[item.value for item in BrushType])

    args = parser.parse_args()
    if args.command == "analyze":
        frame = read_image(args.image)
        scene, debug_path = analyze_and_save_debug(frame, args.debug)
        _print_scene(scene, debug_path, source=f"file:{args.image}")
        return

    if args.command == "capture":
        capture_result = capture_target(
            args.window,
            fallback_to_screen=not args.no_fallback,
            fallback_if_target_uncaptureable=args.fallback_uncaptureable,
        )
        write_image(args.image, capture_result.image)
        scene, debug_path = analyze_and_save_debug(capture_result.image, args.debug)
        _print_scene(scene, debug_path, source=capture_result.source)
        return

    if args.command == "windows":
        for window in list_visible_windows():
            print(
                f"hwnd={window.hwnd} "
                f"pid={window.process_id} "
                f"process={_safe_text(window.process_name)} "
                f"x={window.left} y={window.top} "
                f"w={window.width} h={window.height} "
                f"minimized={window.minimized} "
                f"captureable={window.captureable} "
                f"title={_safe_text(window.title)}"
            )
        return

    if args.command == "simulate":
        state = random_state(seed=args.seed)
        _print_simulation(
            state,
            include_brush=args.brush,
            seed=args.seed,
            refresh_available=args.refresh,
            force_used=args.force_used,
        )
        return

    if args.command == "brush-order":
        options = tuple(BrushType(item) for item in args.options)
        normalized = normalize_brush_options(options)
        print("Input:", " ".join(item.value for item in options))
        print("Normalized:", " ".join(item.value for item in normalized))
        return

    raise SystemExit(f"Unknown command: {args.command}")


def _print_scene(scene: SceneObservation | None, debug_path: Path | None, source: str) -> None:
    print(f"Source: {source}")
    if scene is None:
        print("Board: not detected")
        if debug_path:
            print(f"Debug image: {debug_path}")
        return

    board = scene.board
    print(
        "Board:",
        f"x={board.rect.x}",
        f"y={board.rect.y}",
        f"w={board.rect.w}",
        f"h={board.rect.h}",
        f"score={board.score:.3f}",
    )
    symbol = {
        "blank": ".",
        "blocked": "B",
        "void": "V",
        "faded": "F",
        "paint": "P",
        "unknown": "?",
    }
    for row in range(6):
        print(" ".join(symbol[board.tile(row, col).state.value] for col in range(6)))
    print("Regions:")
    for region in scene.regions:
        print(
            f"- {region.kind.value}: "
            f"x={region.rect.x} y={region.rect.y} "
            f"w={region.rect.w} h={region.rect.h} "
            f"confidence={region.confidence:.3f}"
        )
    if scene.current_paint is not None:
        paint = scene.current_paint
        print("Current paint:")
        print(
            f"- category={paint.category.value} "
            f"confidence={paint.category_confidence:.3f}"
        )
        print(
            f"- level={paint.level} "
            f"confidence={paint.level_confidence:.3f}"
        )
        print(
            f"- subtype={paint.subtype} "
            f"is_treasure={paint.is_treasure}"
        )
        print(
            f"- target={paint.target_object.value} "
            f"direction={paint.direction.value} "
            f"upgrade={paint.upgrade_amount} "
            f"match={paint.match_current}/{paint.match_required}"
        )
        if paint.keyword_scores:
            top_scores = sorted(paint.keyword_scores.items(), key=lambda item: item[1], reverse=True)[:8]
            print("- keyword_scores:", ", ".join(f"{key}={value:.2f}" for key, value in top_scores))
        print(
            f"- icon: x={paint.icon_rect.x} y={paint.icon_rect.y} "
            f"w={paint.icon_rect.w} h={paint.icon_rect.h}"
        )
        print(
            f"- text: x={paint.text_rect.x} y={paint.text_rect.y} "
            f"w={paint.text_rect.w} h={paint.text_rect.h}"
        )
        print("- crops: runtime/current_paint_panel.png, runtime/current_paint_icon.png, runtime/current_paint_text.png")
    if debug_path:
        print(f"Debug image: {debug_path}")


def _safe_text(value: str) -> str:
    return value.encode("gbk", errors="replace").decode("gbk")


def _print_simulation(
    state: PuzzleState,
    *,
    include_brush: bool = False,
    seed: int | None = None,
    refresh_available: bool = False,
    force_used: int = 0,
) -> None:
    symbol = {
        "blank": ".",
        "blocked": "B",
        "void": "V",
        "faded": "F",
        "paint": "P",
        "unknown": "?",
    }
    print("Simulated board:")
    for row in range(6):
        print(" ".join("I" if state.current_anchor == (row, col) else symbol[state.tile(row, col).state.value] for col in range(6)))

    if state.current_rule is not None:
        rule = state.current_rule
        print("Current paint:")
        print(
            f"- category={rule.paint.category.value} "
            f"subtype={_safe_text(str(rule.paint.subtype))} "
            f"treasure={rule.paint.is_treasure} "
            f"level={rule.paint.level}"
        )
        print(
            f"- target={rule.target_object.value} "
            f"direction={rule.direction.value} "
            f"upgrade={rule.upgrade_amount} "
            f"match={rule.match_current}/{rule.match_required}"
        )
        print(f"- anchor={state.current_anchor} gold_scope={len(current_scope_cells(state, include_anchor=True))} cells")

    print("Backpack:")
    if not state.backpack:
        print("- empty / waiting recognition")
    else:
        for index, paint in enumerate(state.backpack, start=1):
            print(f"- {index}. {_safe_text(_paint_summary(paint))}")

    plan = recommend_install_plan(state)
    print("Install plan:")
    if plan is None:
        print("- waiting for current paint + backpack")
    else:
        print(f"- {'complete' if plan.completes_match else 'incomplete'}: {_safe_text(plan.reason)}")
        for index, step in enumerate(plan.steps, start=1):
            print(f"- {index}. row={step.row + 1} col={step.col + 1} <- {_safe_text(_paint_summary(step.paint))}")

    recommendations = recommend_placements(state)
    print("Single-step preview:")
    if not recommendations:
        print("- none")
        return
    for index, rec in enumerate(recommendations, start=1):
        print(
            f"- {index}. row={rec.row + 1} col={rec.col + 1} "
            f"score={rec.score:.1f} completes={rec.completes_match} "
            f"reason={_safe_text(rec.reason)}"
        )

    scenario = random_brush_choice_scenario(seed=None if seed is None else seed + 1000) if include_brush else None
    if scenario is not None:
        print("Brush stage 1:")
        for rec in recommend_brush_types(state, scenario.stage1_options):
            print(f"- {rec.brush_type.value} score={rec.score:.1f} reason={_safe_text(rec.reason)}")
        print(f"Brush stage 2 ({scenario.chosen_stage1.value}):")
        for rec in recommend_affixes(state, scenario.stage2_options):
            print(
                f"- {rec.affix.brush_type.value} { _safe_text(rec.affix.name) } "
                f"score={rec.score:.1f} reason={_safe_text(rec.reason)}"
            )

    resources = ResourceState(
        force_remaining=max(0, 3 - force_used),
        force_used=force_used,
        refresh_available=refresh_available,
        refresh_remaining=2 if refresh_available else 0,
    )
    print("Special actions:")
    actions = recommend_special_actions(state, resources, scenario)
    if not actions:
        print("- none")
        return
    for action in actions:
        mark = "USE" if action.recommended else "HOLD"
        print(
            f"- {mark} {action.action.value} "
            f"score={action.score:.1f} reason={_safe_text(action.reason)}"
        )


def _paint_summary(paint) -> str:
    if paint.is_treasure:
        return f"宝箱 Lv{paint.level}"
    category = "怪物" if paint.category.value == "monster" else "玩法" if paint.category.value == "gameplay" else "未知"
    subtype = f"-{paint.subtype}" if paint.subtype else ""
    return f"{category}{subtype} Lv{paint.level}"
