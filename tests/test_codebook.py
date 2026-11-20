"""编码字典：棋盘/颜料/词缀编号。

这层的意义是让「一步记录」从「重写整个棋盘」变成「写一个编号」。实测
926 步只有 283 种局面（重复率 69%），编号后重复的 69% 只花一个 ``B12``。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant.codebook import (
    CODEBOOK_VERSION,
    Codebook,
    affix_id,
    board_signature,
    paint_id,
)


def _tile(code: str, bonus: int = 0, aura: int = 0) -> dict:
    return {"code": code, "b": bonus, "au": aura}


class BoardSignatureTests(unittest.TestCase):
    def test_same_board_same_signature(self) -> None:
        states = "PPVBBK" * 6
        tiles = [_tile("M1"), None, None]
        self.assertEqual(
            board_signature(states, tiles, "I:M1:V:A:4:1"),
            board_signature(states, list(tiles), "I:M1:V:A:4:1"),
        )

    def test_any_paint_change_breaks_the_signature(self) -> None:
        """颜料从 M1 变成 M2，就是另一个局面。"""

        states = "PPP" * 12
        a = board_signature(states, [_tile("M1")], "I:M1:V:A:4:1")
        b = board_signature(states, [_tile("M2")], "I:M1:V:A:4:1")
        self.assertNotEqual(a, b)

    def test_bonus_level_breaks_the_signature(self) -> None:
        """同一格吃到不同加成，是不同局面——这正是「还原时地块自带 buff」的关键。"""

        states = "PPP" * 12
        plain = board_signature(states, [_tile("M1", bonus=0)], "I:M1:V:A:4:1")
        boosted = board_signature(states, [_tile("M1", bonus=2)], "I:M1:V:A:4:1")
        self.assertNotEqual(plain, boosted)

    def test_aura_amount_breaks_the_signature(self) -> None:
        states = "PPP" * 12
        no_aura = board_signature(states, [_tile("M1", aura=0)], "I:M1:V:A:4:1")
        with_aura = board_signature(states, [_tile("M1", aura=1)], "I:M1:V:A:4:1")
        self.assertNotEqual(no_aura, with_aura)

    def test_target_change_breaks_the_signature(self) -> None:
        states = "PPP" * 12
        tiles = [_tile("M1")]
        self.assertNotEqual(
            board_signature(states, tiles, "I:M1:V:A:4:1"),
            board_signature(states, tiles, "I:M1:H:A:4:1"),
        )


class PaintIdTests(unittest.TestCase):
    def test_known_subtype_gets_its_initial(self) -> None:
        self.assertEqual("sr1", paint_id("monster", 1, "鼠人"))
        self.assertEqual("hb3", paint_id("monster", 3, "黑帮"))

    def test_unknown_subtype_falls_back_to_category(self) -> None:
        """识别不出种类时退回老写法，两套编码共存。"""

        self.assertEqual("M1", paint_id("monster", 1, None))
        self.assertEqual("G4", paint_id("gameplay", 4, None))

    def test_treasure_ignores_subtype(self) -> None:
        """宝箱本身就是唯一一类，不需要前缀。"""

        self.assertEqual("T7", paint_id("gameplay", 7, "宝箱", is_treasure=True))
        self.assertEqual("T7", paint_id("gameplay", 7, None, is_treasure=True))


class AffixIdTests(unittest.TestCase):
    def test_zhuanfeng_carries_direction(self) -> None:
        """转锋词缀会改写方向，编号里带上方向码才能不查表就知道改成了什么。"""

        self.assertEqual("ZF-H1", affix_id("1", "H"))
        self.assertEqual("ZF-V2", affix_id("1", "V", index=2))

    def test_other_brushes_still_get_a_prefix(self) -> None:
        self.assertTrue(affix_id("2", None).startswith("DQ"))
        self.assertTrue(affix_id("3", None).startswith("KS"))
        self.assertTrue(affix_id(None, None).startswith("QT"))


class CodebookTests(unittest.TestCase):
    def test_repeated_board_gets_the_same_code(self) -> None:
        book = Codebook()
        first = book.intern_board("sig-a", {"states": "PPP"})
        second = book.intern_board("sig-a", {"states": "PPP"})
        third = book.intern_board("sig-b", {"states": "BBB"})

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(2, len(book.boards))

    def test_codes_are_never_recycled(self) -> None:
        """编号只增不减，否则旧记录会指向错的内容。"""

        book = Codebook()
        codes = [book.intern_board(f"sig-{i}", {"i": i}) for i in range(5)]
        self.assertEqual(["B1", "B2", "B3", "B4", "B5"], codes)
        self.assertEqual(0, book.resolve_board("B1")["i"])

    def test_paint_and_affix_intern_are_idempotent(self) -> None:
        book = Codebook()
        book.intern_paint("sr1", {"c": "monster", "l": 1, "s": "鼠人"})
        book.intern_paint("sr1", {"c": "monster", "l": 1, "s": "鼠人"})
        self.assertEqual(1, len(book.paints))

        book.intern_affix("ZF-H1", {"brush": "1", "direction": "H"})
        self.assertEqual(1, len(book.affixes))

    def test_round_trip_through_disk(self) -> None:
        book = Codebook()
        book.intern_board("sig-a", {"states": "PPP"})
        book.intern_paint("sr1", {"c": "monster", "l": 1, "s": "鼠人"})
        book.intern_affix("ZF-H1", {"brush": "1"})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "codebook.json"
            book.save(path)
            loaded = Codebook.load(path)

        self.assertEqual(book.boards, loaded.boards)
        self.assertEqual(book.paints, loaded.paints)
        self.assertEqual(book.affixes, loaded.affixes)
        self.assertEqual("B1", loaded.board_index["sig-a"])

    def test_missing_or_corrupt_file_yields_empty_book(self) -> None:
        """丢一次索引不该让工具起不来。"""

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            self.assertEqual(0, len(Codebook.load(missing).boards))

            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            self.assertEqual(0, len(Codebook.load(broken).boards))

            wrong_version = Path(tmp) / "old.json"
            wrong_version.write_text(
                json.dumps({"version": CODEBOOK_VERSION + 1, "boards": {"B1": {}}}),
                encoding="utf-8",
            )
            self.assertEqual(0, len(Codebook.load(wrong_version).boards))

    def test_dirty_flag_tracks_mutations(self) -> None:
        book = Codebook()
        self.assertFalse(book.dirty)
        book.intern_board("sig-a", {"states": "PPP"})
        self.assertTrue(book.dirty)

        with tempfile.TemporaryDirectory() as tmp:
            book.save(Path(tmp) / "cb.json")
        self.assertFalse(book.dirty)


if __name__ == "__main__":
    unittest.main()
