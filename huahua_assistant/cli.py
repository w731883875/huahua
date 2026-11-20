from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch_evaluation import build_batch_evaluation_report
from .capture import DEFAULT_WINDOW_TITLE, capture_target, list_visible_windows, read_image, write_image
from .game_model import (
    DEFAULT_PET_LEVEL,
    MAX_FORCE_TIMES,
    PET_REFRESH_BY_LEVEL,
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
from .vision import analyze_and_save_debug, analyze_scene


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
    capture.add_argument(
        "--backend",
        default=None,
        choices=["auto", "legacy", "dxcam", "highspeed"],
        help="Capture backend. Default follows runtime config.",
    )
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

    batch = subparsers.add_parser(
        "batch-evaluate",
        help="Run deterministic policy and paired resource proxy evaluation.",
    )
    batch.add_argument("--start-seed", type=int, default=0)
    batch.add_argument("--runs", type=int, default=100)
    batch.add_argument("--output", type=Path, default=None)
    batch.add_argument("--code-revision", default=None)

    brush_order = subparsers.add_parser("brush-order", help="Normalize stage-1 brush option order.")
    brush_order.add_argument("options", nargs="+", choices=[item.value for item in BrushType])

    practice = subparsers.add_parser(
        "practice",
        help="雪原练习模式：只练放置颜料（不刷副本、不算技艺），按官方几率生成对局。",
    )
    practice.add_argument("--seed", type=int, default=None)
    practice.add_argument("--pet-level", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    practice.add_argument("--rules", type=int, default=2, help="这局最多结算几条规则（0 = 不限）。")
    practice.add_argument("--runs", type=int, default=0, help=">0 时改为批量自动对局并输出分布。")
    practice.add_argument("--base-seed", type=int, default=1, help="批量模式起始 seed。")
    practice.add_argument(
        "--auto",
        action="store_true",
        help="用内置策略把这一局自动打完（--runs=1 时可用）。",
    )
    practice.add_argument(
        "--greedy",
        action="store_true",
        help="自动对局只吃即时收益（快，适合刷大量局）。",
    )
    practice.add_argument(
        "--anchor",
        default=None,
        metavar="ROW,COL",
        help="手动指定初始颜料锚点，例如 --anchor 2,2。不给则停在等待放锚点。",
    )
    practice.add_argument(
        "--keep-pending",
        action="store_true",
        help="保留随机生成时已放好的锚点（默认会退回等待放锚点）。",
    )
    practice.add_argument("--json", action="store_true", help="以 JSON 输出，便于落盘归档。")
    practice.add_argument(
        "--record",
        action="store_true",
        help="把这一局（或这批自动对局）存进练习高分库，之后可完整还原。",
    )
    practice.add_argument(
        "--note",
        default="",
        help="给这次记录加一句备注，方便以后检索（例如「宝箱流」）。",
    )
    practice.add_argument(
        "--library",
        type=Path,
        default=None,
        help="练习库路径，默认 runtime/practice_library.jsonl。",
    )

    practice_lib = subparsers.add_parser(
        "practice-library",
        help="查看练习高分库：最高分、技巧统计、以及逐局还原校验。",
    )
    practice_lib.add_argument("--library", type=Path, default=None)
    practice_lib.add_argument("--top", type=int, default=10, help="列出前 N 高分。")
    practice_lib.add_argument(
        "--verify",
        action="store_true",
        help="把库里每一局都重放一遍，确认能完整还原。",
    )
    practice_lib.add_argument(
        "--compare",
        action="store_true",
        help="对比各技巧命中局的均分与全库均分，验证技巧建议是否站得住。",
    )
    practice_lib.add_argument(
        "--recheck",
        action="store_true",
        help="把全部记录重新校验并回写 replay 字段（规则改动后必须跑一次）。",
    )
    practice_lib.add_argument("--json", action="store_true")

    practice_play = subparsers.add_parser(
        "practice-play",
        help="手动练习：一步一步下，可存档、撤销、随时复盘为什么高分。",
    )
    practice_play.add_argument("--seed", type=int, default=None)
    practice_play.add_argument("--pet-level", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    practice_play.add_argument("--rules", type=int, default=3, help="本局最多结算几条词缀。")
    practice_play.add_argument("--max-steps", type=int, default=60)
    practice_play.add_argument("--load", default=None, help="从存档继续（存档名，不含 .json）。")
    practice_play.add_argument("--library", type=Path, default=None)

    practice_sim = subparsers.add_parser(
        "practice-simulate",
        help="按官方几率批量模拟，并把观测分布与 pak 官方数字对账。",
    )
    practice_sim.add_argument("--runs", type=int, default=30)
    practice_sim.add_argument("--base-seed", type=int, default=1)
    practice_sim.add_argument("--pet-level", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    practice_sim.add_argument("--rules", type=int, default=2)
    practice_sim.add_argument("--prefer", choices=("greedy", "lookahead"), default="greedy")
    practice_sim.add_argument("--record", action="store_true", help="同时写进高分库。")
    practice_sim.add_argument("--library", type=Path, default=None)
    practice_sim.add_argument("--json", default=None, help="把结果另存为 JSON 文件。")

    practice_scan = subparsers.add_parser(
        "practice-scan",
        help="批量跑局，挑出终局有 7 级宝箱的复盘（模拟复盘，非实机）。",
    )
    practice_scan.add_argument("--runs", type=int, default=10000)
    practice_scan.add_argument("--base-seed", type=int, default=1)
    practice_scan.add_argument("--pet-level", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    practice_scan.add_argument("--rules", type=int, default=2)
    practice_scan.add_argument("--prefer", choices=("greedy", "lookahead"), default="greedy")
    practice_scan.add_argument("--workers", type=int, default=None)
    practice_scan.add_argument("--no-save", action="store_true", help="只统计，不落盘复盘。")
    practice_scan.add_argument("--no-record", action="store_true", help="不入库。")
    practice_scan.add_argument("--library", type=Path, default=None)
    practice_scan.add_argument("--out-dir", default=None)
    practice_scan.add_argument("--json", default=None)
    practice_scan.add_argument("--progress", action="store_true")

    llm = subparsers.add_parser(
        "llm",
        help="Ask the vision LLM what to do. Give it a screenshot, or it captures one.",
    )
    llm.add_argument("image", type=Path, nargs="?", default=None)
    llm.add_argument("--provider", default=None, choices=["openai_compat", "ollama"])
    llm.add_argument("--base-url", default=None)
    llm.add_argument("--api-key", default=None)
    llm.add_argument("--model", default=None)
    llm.add_argument("--context", default="", help="Extra text context for the model.")
    llm.add_argument(
        "--no-context",
        action="store_true",
        help="Do not attach the local recognition result as context.",
    )

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
            backend=args.backend,
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

    if args.command == "practice":
        _run_practice(args)
        return

    if args.command == "practice-library":
        _run_practice_library(args)
        return

    if args.command == "practice-play":
        from . import practice_play as play_mod

        return play_mod.main([
            *([] if args.seed is None else ["--seed", str(args.seed)]),
            "--pet", str(args.pet_level),
            "--rules", str(args.rules),
            "--max-steps", str(args.max_steps),
            *([] if args.load is None else ["--load", args.load]),
            *([] if args.library is None else ["--library", str(args.library)]),
        ])

    if args.command == "practice-simulate":
        from . import practice_simulate as simulate_mod

        return simulate_mod.main([
            "--runs", str(args.runs),
            "--base-seed", str(args.base_seed),
            "--pet", str(args.pet_level),
            "--rules", str(args.rules),
            "--prefer", args.prefer,
            *(["--record"] if args.record else []),
            *([] if args.library is None else ["--library", str(args.library)]),
            *([] if args.json is None else ["--json", args.json]),
        ])

    if args.command == "practice-scan":
        from . import practice_batch as batch_mod

        return batch_mod.main([
            "--runs", str(args.runs),
            "--base-seed", str(args.base_seed),
            "--pet", str(args.pet_level),
            "--rules", str(args.rules),
            "--prefer", args.prefer,
            *([] if args.workers is None else ["--workers", str(args.workers)]),
            *(["--no-save"] if args.no_save else []),
            *(["--no-record"] if args.no_record else []),
            *([] if args.library is None else ["--library", str(args.library)]),
            *([] if args.out_dir is None else ["--out-dir", args.out_dir]),
            *([] if args.json is None else ["--json", args.json]),
            *(["--progress"] if args.progress else []),
        ])

    if args.command == "batch-evaluate":
        if args.runs <= 0:
            parser.error("--runs must be greater than zero")
        report = build_batch_evaluation_report(
            range(args.start_seed, args.start_seed + args.runs),
            code_revision=args.code_revision,
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output is None:
            print(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Batch evaluation: {args.output}")
            print(f"Report SHA256: {report['report_sha256']}")
        return

    if args.command == "brush-order":
        options = tuple(BrushType(item) for item in args.options)
        normalized = normalize_brush_options(options)
        print("Input:", " ".join(item.value for item in options))
        print("Normalized:", " ".join(item.value for item in normalized))
        return

    if args.command == "llm":
        _run_llm_advice(args)
        return

    raise SystemExit(f"Unknown command: {args.command}")


def _run_llm_advice(args: argparse.Namespace) -> None:
    """把一张画面丢给视觉大模型，看它说出「点哪里」。"""
    from dataclasses import replace

    from .llm import Advisor
    from .llm.advisor import load_config

    overrides = {
        key: value
        for key, value in (
            ("provider", args.provider),
            ("base_url", args.base_url),
            ("api_key", args.api_key),
            ("model", args.model),
        )
        if value
    }
    config = replace(load_config(), enabled=True, **overrides)

    if args.image is not None:
        frame = read_image(args.image)
        source = f"file:{args.image}"
    else:
        captured = capture_target()
        frame = captured.image
        source = captured.source

    print(f"来源 {source}  尺寸 {frame.shape[1]}x{frame.shape[0]}")
    print(f"模型 {config.provider} / {config.model} @ {config.base_url}")

    context = args.context
    if not args.no_context:
        # 和程序里走同一条路：先本地识别，再把已知信息当文字上下文附上去。
        # 单图验证要能代表实机，否则「命令行好好的、实机不对」会很难查。
        from .llm.context import describe_scene

        try:
            scene = analyze_scene(frame)
        except Exception:  # noqa: BLE001 - 本地识别失败不该挡着测模型
            scene = None
        local = describe_scene(scene) if scene is not None else ""
        if local:
            print("---- 本地上下文 ----")
            print(local)
        context = "\n".join(part for part in (context, local) if part.strip())

    advice = Advisor(config).advise(frame, context, force=True)
    print(f"耗时 {advice.elapsed:.2f}s")
    if advice.ok:
        print(f"建议 {advice.summary()}")
    else:
        print(f"失败 {advice.error}")
    if advice.raw:
        print("---- 原始输出 ----")
        print(advice.raw)


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


def _run_practice(args) -> None:
    """练习模式入口：单局 / 手动锚点 / 自动对局 / 批量分布。"""
    import json as _json

    from . import practice as practice_mod
    from . import practice_library as library_mod

    rules = None if args.rules == 0 else max(1, args.rules)
    prefer = "greedy" if args.greedy else "lookahead"
    library_path = getattr(args, "library", None)

    if args.runs and args.runs > 0:
        library = library_mod.PracticeLibrary(library_path)
        explanations = []
        written = 0
        for index in range(args.runs):
            session = practice_mod.start_practice(
                args.base_seed + index,
                pet_level=args.pet_level,
                rules=rules,
            )
            practice_mod.auto_play(session, prefer=prefer)
            explanation = practice_mod.explain_high_score(
                session.state,
                steps=session.steps,
                session=session,
                weights=session.weights,
            )
            explanations.append(explanation)
            if args.record:
                entry = library.record(
                    explanation, session.snapshot(), source="simulation",
                    note=args.note,
                )
                if entry is not None:
                    written += 1
        summary = practice_mod.summarize_runs(explanations)
        data = summary.as_dict()
        if args.json:
            print(_json.dumps(data, ensure_ascii=False, indent=1))
            return
        print(f"批量练习 {data['runs']} 局（seed {args.base_seed}..{args.base_seed + args.runs - 1}）")
        print(f"  平均得分   {data['mean_score']:.4f}")
        print(f"  最高得分   {data['best_score']:.4f}")
        print(f"  平均等级   {data['mean_level']:.2f}")
        print(f"  7 级宝箱   共 {data['level7_treasure_total']} 个")
        if data["tactic_hits"]:
            print("  技巧触发：")
            for key, count in data["tactic_hits"].items():
                title = practice_mod.TACTIC_LIBRARY.get(key, (key, "", ""))[0]
                print(f"    {count:4d} 次  {title}")
        _print_paint_level_source()
        if args.record:
            print(f"  已入库 {written} 局 → {library.path}")
            if written < args.runs:
                print(f"  （{args.runs - written} 局与库中已有记录重复，已跳过）")
        return

    session = practice_mod.start_practice(
        args.seed,
        pet_level=args.pet_level,
        rules=rules,
        wait_for_anchor=not args.keep_pending,
    )

    if args.anchor:
        try:
            row_text, col_text = args.anchor.split(",")
            row, col = int(row_text), int(col_text)
        except ValueError:
            raise SystemExit("--anchor 需要形如 2,2 的行列号")
        if not session.place_anchor(row, col):
            raise SystemExit(f"锚点 ({row},{col}) 放不下（可能不是空白格）")

    if args.auto:
        practice_mod.auto_play(session, prefer=prefer)

    explanation = practice_mod.explain_high_score(
        session.state,
        steps=session.steps,
        session=session,
        weights=session.weights,
    )

    entry = None
    if args.record:
        library = library_mod.PracticeLibrary(library_path)
        entry = library.record(
            explanation, session.snapshot(), source="practice", note=args.note
        )

    if args.json:
        payload = explanation.as_dict()
        if args.record:
            payload["recorded"] = entry is not None
            payload["signature"] = entry["signature"] if entry else None
        print(_json.dumps(payload, ensure_ascii=False, indent=1))
        return

    print(f"练习对局  seed={session.seed}  宠物Lv{session.pet_level}  "
          f"强制剩余={session.force_remaining}/{MAX_FORCE_TIMES}  "
          f"刷新剩余={session.refresh_remaining}/"
          f"{PET_REFRESH_BY_LEVEL.get(session.pet_level, 0)}")
    for row_text in practice_mod.board_rows(session.state):
        print("  " + row_text)
    rule = session.state.current_rule
    if rule is not None:
        print(
            f"当前规则：{rule.paint.category.value} Lv{rule.paint.level} "
            f"→ {rule.target_object.value} / {rule.direction.value}，"
            f"匹配 {rule.match_current}/{rule.match_required}"
        )
    if session.steps:
        print(f"已走 {len(session.steps)} 步：")
        for step in session.steps:
            where = "-" if step.row < 0 else f"({step.row},{step.col})"
            print(
                f"  {step.index:2d}. {step.kind:<14s} {where:<9s} "
                f"Lv{step.level_before}→Lv{step.level_after}  "
                f"分 {step.board_value_before:.3f}→{step.board_value_after:.3f}  {step.note}"
            )
    print()
    print(explanation.format_text())
    if args.record:
        if entry is None:
            print("\n（这局与库中已有记录完全重复，未重复入库）")
        else:
            print(f"\n已入库：{library_mod.PracticeLibrary(library_path).path}")
            print(f"  签名 {entry['signature'][:16]}…  可用 `practice-library --verify` 校验可还原")


def _print_paint_level_source() -> None:
    """把「颜料出现几率」的出处打出来。

    模拟输出必须能自证数据来源，否则「按官方几率」只是句口号。
    """
    from . import s8_official_data as official

    weights = official.PAINT_LEVEL_WEIGHTS
    if not weights:
        return
    total = sum(weights.values())
    parts = "  ".join(
        f"Lv{level}={count / total:.2%}" for level, count in sorted(weights.items())
    )
    print(f"  颜料等级权重（官方）：{parts}")
    print(f"    期望等级 {official.paint_level_expectation():.2f}；来源 {official.PAINT_LEVEL_WEIGHT_SOURCE}")


def _run_practice_library(args) -> None:
    """练习高分库：统计 / 前 N 高分 / 逐局还原校验 / 技巧有效性对比。

    四个开关相互独立，可以任意组合（例如 ``--verify --compare`` 同时看
    还原结果与技巧有效性）。所以这里不用「命中一个就 return」的写法，
    而是把每块内容收集起来最后统一输出。
    """
    import json as _json

    from . import practice_library as library_mod

    library = library_mod.PracticeLibrary(args.library)
    # --recheck 要在读统计之前跑：它会把 replay 字段就地刷新，
    # 统计里的一致性数字才有意义。
    recheck_result = library.recheck_consistency() if args.recheck else None
    stats = library.stats()
    top = library.top(args.top)
    verify_rows: list[tuple[dict, object]] | None = (
        library.verify_all() if args.verify else None
    )
    compare_rows = library.compare_tactics() if args.compare else None

    if args.json:
        payload: dict = {"stats": stats, "top": top}
        if recheck_result is not None:
            payload["recheck"] = recheck_result
        if verify_rows is not None:
            failed = [(entry, check) for entry, check in verify_rows if not check.ok]
            payload["verify"] = {
                "verified": len(verify_rows),
                "failed": len(failed),
                "failures": [
                    {
                        "seed": entry.get("seed"),
                        "signature": entry.get("signature"),
                        "mismatch": check.mismatch,
                        "paint_mismatch": check.paint_mismatch,
                    }
                    for entry, check in failed
                ],
            }
        if compare_rows is not None:
            payload["compare"] = compare_rows
        print(_json.dumps(payload, ensure_ascii=False, indent=1))
        return

    print(library_mod.format_library_report(stats))

    if recheck_result is not None:
        print()
        print(
            f"重新校验：{recheck_result['checked']} 局"
            f"（修好 {recheck_result['fixed']} 局，"
            f"新增不一致 {recheck_result['broken']} 局，"
            f"未变 {recheck_result['unchanged']} 局）"
        )

    if top:
        print()
        print(f"最高分前 {len(top)} 局：")
        for entry in top:
            print(library_mod.LibraryEntryView.from_entry(entry).format_text())

    if verify_rows is not None:
        failed = [(entry, check) for entry, check in verify_rows if not check.ok]
        print()
        print(f"还原校验：{len(verify_rows)} 局中 {len(verify_rows) - len(failed)} 局完全一致")
        for entry, check in failed[:10]:
            print(f"  ✘ seed={entry.get('seed')}  {check.mismatch}")
            if check.paint_mismatch:
                index, recorded, actual = check.paint_mismatch
                print(f"      第 {index} 步：记录 {recorded}，回放 {actual}")

    if compare_rows:
        print()
        print("技巧有效性（命中该技巧的局均分 vs 全库均分）：")
        for row in compare_rows:
            sign = "+" if row["delta"] >= 0 else ""
            print(
                f"  {sign}{row['delta']:.4f}  {row['hits']:4d} 次  {row['title']}"
            )
        print("  注：这是相关性，不是因果——命中某技巧的局本身可能就来自更好的局面。")



def _print_simulation(
    state: PuzzleState,
    *,
    include_brush: bool = False,
    seed: int | None = None,
    refresh_available: bool = False,
    force_used: int = 0,
    pet_level: int = DEFAULT_PET_LEVEL,
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

    refresh_total = PET_REFRESH_BY_LEVEL.get(int(pet_level), 0)
    resources = ResourceState(
        force_remaining=max(0, MAX_FORCE_TIMES - force_used),
        force_used=force_used,
        refresh_available=refresh_available and refresh_total > 0,
        refresh_remaining=refresh_total if refresh_available else 0,
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
