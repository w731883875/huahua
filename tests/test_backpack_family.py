"""背包格认家族：只靠「框形状 + 框色」是认不出来的，得对着官方图标认人。

## 修的是什么

界面上背包那一排以前只画得出**等级框**（彩色菱形），图案永远是空的——因为
``BackpackPaintObservation.subtype`` 从头到尾没被赋过值，而 ``compose_icon``
拿不到家族名就贴不上图案。实机上表现为「背包颜料认不出来」，只能看颜色猜。

根因是背包格走的是 ``_classify_backpack_frame_shape``：靠 HoughLinesP 量菱形/
圆形边框的线段长度，在 90px 的小格上加一圈 ``LvN`` 角标，判据基本失控——
``jietu/初始和背包宝箱.png`` 那张 15 格**清一色**判成 ``gameplay 0.64``，
连 4 个圆形框（征战之神）也算成菱形。

现在改成和棋盘格同一套顺序：**模板库 → 官方图标**，类别由家族反推。

## 为什么还要裁中心

官方图案只占品质框的 ``paint_frame.ICON_SCALE``（128/192），而整格 crop 里
还塞着外框和角标。整格直接比：15 格里只有 7 格凑够 40 个关键点；按 ICON_SCALE
裁中心后 **15 格全过**（good 55~93）。所以阈值没有放宽——是无关像素被去掉了。
:meth:`BackpackFamilyTest.test_center_crop_is_what_makes_it_readable` 就是这条的负例自证。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from huahua_assistant import paint_frame, paint_icon, paint_kind, vision
from huahua_assistant.backpack import order_violations, paint_from_observation
from huahua_assistant.capture import read_image
from huahua_assistant.game_model import Paint, PaintCategory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_ROOT = PROJECT_ROOT / "jietu"
BACKPACK_SHOT = SCREENSHOT_ROOT / "初始和背包宝箱.png"

#: 等后台把 15 格认完的上限（秒）。认一格 40~60ms，15 格串行不到 1 秒；
#: 留 20 秒是给慢机器和 ORB 冷启动（0.7 秒）的余量。
FAMILY_WAIT_SECONDS = 20.0

#: 人工读图得到的真值（3 行 × 5 列，行优先）。
#: 这张图 15 格全是**玩法**颜料，所以它同时也证明「类别不能靠框形状猜」——
#: 旧代码把 4 个圆形框判成菱形，恰好因为全图都是玩法才没露馅。
EXPECTED_FAMILIES = (
    "赛季",
    "存活",
    "机械之神",
    "巨力之神",
    "宝箱",
    "存活",
    "存活",
    "机械之神",
    "巨力之神",
    "狩猎之神",
    "狩猎之神",
    "狩猎之神",
    "狩猎之神",
    "征战之神",
    "征战之神",
)


def _shot():
    if not BACKPACK_SHOT.is_file():
        return None
    return read_image(BACKPACK_SHOT)


class BackpackFamilyTest(unittest.TestCase):
    """真实截图上认全 15 格的家族。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image = _shot()
        if cls.image is None:
            raise unittest.SkipTest(f"缺少实机截图: {BACKPACK_SHOT}")
        # ORB 的懒初始化要 0.7 秒，先付掉，别算进断言里。
        paint_icon.gallery()
        # 认家族在后台线程做（每格 40~60ms，15 格放帧里就是 0.9 秒）。
        # 这里把第一帧跑出来、等后台认完，再取第二帧——真实软件就是这个节奏：
        # 面板出现的那一帧先出空框，结果出来后的下一次取场景就有图案。
        vision.reset_vision_caches()
        scene = vision.analyze_scene_realtime(cls.image)
        if not vision.wait_for_family_identification(FAMILY_WAIT_SECONDS):
            raise AssertionError("后台 20 秒内没认完 15 格")
        cls.first_frame = scene
        cls.scene = vision.analyze_scene_realtime(cls.image, previous_scene=scene)

    def _scene_after_settling(self):
        return self.scene

    def test_first_frame_does_not_block_on_identification(self) -> None:
        """面板出现的第一帧必须立刻返回：那一刻一格都还没认。

        这条同时锁住「别把识别偷偷搬回帧里」——搬回去的话这里就有家族了，
        而 15 格 × 40~60ms 会把 0.75 秒的交互预算吃穿。
        """
        paints = self.first_frame.backpack.paints
        self.assertEqual(len(paints), 15)
        self.assertTrue(
            all(paint.subtype is None for paint in paints),
            "首帧不该有家族——识别在后台线程，帧内只查缓存",
        )

    def test_second_frame_picks_up_background_results(self) -> None:
        """后台结果出来后，下一帧就该带上图案（不用等第三次取场景）。"""
        second = vision.analyze_scene_realtime(self.image, previous_scene=self.scene)
        self.assertEqual(
            [paint.subtype for paint in second.backpack.paints],
            list(EXPECTED_FAMILIES),
        )

    def test_real_backpack_identifies_every_family(self) -> None:
        scene = self._scene_after_settling()
        self.assertIsNotNone(scene.backpack)
        paints = scene.backpack.paints
        self.assertEqual(len(paints), 15, "这张图是 3×5 满页")

        families = tuple(paint.subtype for paint in paints)
        self.assertEqual(families, EXPECTED_FAMILIES)

    def test_identified_category_comes_from_family(self) -> None:
        scene = self._scene_after_settling()
        for paint in scene.backpack.paints:
            known = paint_kind.category_of(paint.subtype)
            self.assertIsNotNone(known, f"{paint.subtype} 应在官方目录里")
            expected = PaintCategory.MONSTER if known == "monster" else PaintCategory.GAMEPLAY
            self.assertEqual(paint.category, expected, f"{paint.subtype} 的类别由家族决定")

    def test_backpack_reading_satisfies_sorting_rule(self) -> None:
        """实机读数必须经得起「目标类别优先 + 等级降序」这条硬约束。"""
        scene = self._scene_after_settling()
        paints = tuple(paint_from_observation(item) for item in scene.backpack.paints)
        self.assertEqual(order_violations(paints), (), "真值读数不该被判可疑")

    def test_center_crop_is_what_makes_it_readable(self) -> None:
        """负例自证：不裁中心就有一半格子凑不够关键点。

        这条用来挡住「把阈值调低换召回」的做法——阈值没动过，
        变的只是喂进去的像素。
        """
        scene = self._scene_after_settling()
        gallery = paint_icon.gallery()
        whole = 0
        cropped = 0
        for paint in scene.backpack.paints:
            rect = paint.slot_rect
            crop = self.image[rect.y : rect.bottom, rect.x : rect.right]
            if gallery.identify(crop) is not None:
                whole += 1
            if gallery.identify(crop, center_ratio=paint_frame.ICON_SCALE) is not None:
                cropped += 1

        self.assertEqual(cropped, 15, "裁到官方图案比例后应当全部认出来")
        self.assertLess(whole, cropped, "不裁中心时匹配点数不够，正是这次要修的坑")
        self.assertLessEqual(whole, 9, "整格比对时至少 6 格认不出（实测 8 格）")


class BackpackOrderTest(unittest.TestCase):
    """排序规则是硬约束：违反它说明这一帧有格子认错了。"""

    @staticmethod
    def _paint(category: PaintCategory, level: int, subtype: str) -> Paint:
        return Paint(category=category, level=level, subtype=subtype)

    def test_flags_category_switching_back(self) -> None:
        # 玩法 → 怪物 → 玩法：第二个玩法以前必有认错的格子。
        paints = [
            self._paint(PaintCategory.GAMEPLAY, 5, "存活"),
            self._paint(PaintCategory.MONSTER, 4, "鼠人"),
            self._paint(PaintCategory.GAMEPLAY, 3, "存活"),
        ]
        self.assertIn(3, order_violations(paints))

    def test_flags_level_going_up(self) -> None:
        paints = [
            self._paint(PaintCategory.GAMEPLAY, 3, "存活"),
            self._paint(PaintCategory.GAMEPLAY, 5, "巨力之神"),
        ]
        self.assertEqual(order_violations(paints), (2,))

    def test_accepts_monotonic_reading(self) -> None:
        paints = [
            self._paint(PaintCategory.GAMEPLAY, 3, "存活"),
            self._paint(PaintCategory.GAMEPLAY, 3, "机械之神"),
            self._paint(PaintCategory.GAMEPLAY, 1, "宝箱"),
        ]
        self.assertEqual(order_violations(paints), ())

    def test_empty_reading_is_not_an_error(self) -> None:
        self.assertEqual(order_violations(()), ())


class BackpackTreasureGuardTest(unittest.TestCase):
    """家族认不出时才看颜色，而且必须过棋盘那条同样的闸。"""

    def test_silver_cell_is_not_a_treasure_without_family(self) -> None:
        image = _shot()
        if image is None:
            self.skipTest(f"缺少实机截图: {BACKPACK_SHOT}")
        # 用最后一格（征战之神，紫黑图案）当「金色误判」的替身：
        # 家族没认出来 + 等级读不出来 → 颜色说明不了任何事。
        crop = image[572:666, 1297:1391]
        is_treasure, _confidence = vision._resolve_backpack_treasure(
            crop,
            None,
            5,  # Lv5 橙框本身就够金，颜色没有分辨力
            PaintCategory.GAMEPLAY,
            0.70,
        )
        self.assertFalse(is_treasure, "Lv5 橙框不能光凭颜色翻成宝箱")

    def test_named_family_decides_treasure(self) -> None:
        image = _shot()
        if image is None:
            self.skipTest(f"缺少实机截图: {BACKPACK_SHOT}")
        crop = image[376:470, 1297:1391]
        is_treasure, _confidence = vision._resolve_backpack_treasure(
            crop,
            "宝箱",
            1,
            PaintCategory.GAMEPLAY,
            0.90,
        )
        self.assertTrue(is_treasure)
        is_treasure, _confidence = vision._resolve_backpack_treasure(
            crop,
            "存活",
            1,
            PaintCategory.GAMEPLAY,
            0.90,
        )
        self.assertFalse(is_treasure, "家族认出来了就只信家族")


class FamilyBudgetTest(unittest.TestCase):
    """认家族不能把交互帧吃穿：后台线程 + 指纹缓存。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image = _shot()
        if cls.image is None:
            raise unittest.SkipTest(f"缺少实机截图: {BACKPACK_SHOT}")
        paint_icon.gallery()
        vision.reset_vision_caches()
        scene = vision.analyze_scene_realtime(cls.image)
        if not vision.wait_for_family_identification(FAMILY_WAIT_SECONDS):
            raise AssertionError("后台 20 秒内没认完")
        cls.scene = vision.analyze_scene_realtime(cls.image, previous_scene=scene)

    def test_cached_read_never_calls_orb_or_templates(self) -> None:
        """同一屏内容再读一遍不该再碰 ORB/模板库——靠指纹缓存直接命中。

        缓存失效时这两个函数会被调用，于是这里的 patch 会抛 AssertionError，
        等于用负例证明缓存真的接在链路上，而不是写了没用。
        """
        with (
            patch.object(vision.paint_icon, "identify", side_effect=AssertionError("缓存没命中：又去认 ORB 了")),
            patch.object(vision.paint_kind, "read_kind", side_effect=AssertionError("缓存没命中：又去跑模板库了")),
        ):
            observation = vision._analyze_backpack_fast(self.image, self.scene.regions, None)

        self.assertIsNotNone(observation)
        self.assertEqual(
            [paint.subtype for paint in observation.paints],
            list(EXPECTED_FAMILIES),
            "缓存命中后应当立刻给全 15 格",
        )

    def test_cold_read_returns_cells_without_families(self) -> None:
        """缓存空的时候：照样给出 15 个格子（界面显示待识别），只是先没有图案。"""
        vision.reset_vision_caches()
        observation = vision._analyze_backpack_fast(self.image, self.scene.regions, None)
        self.assertIsNotNone(observation, "认不出家族也要给出格子，不能整块消失")
        self.assertEqual(len(observation.paints), 15)
        self.assertTrue(all(paint.subtype is None for paint in observation.paints))
        # 收尾：把后台队列排空，别影响后面的用例
        vision.wait_for_family_identification(FAMILY_WAIT_SECONDS)

    def test_queue_is_bounded(self) -> None:
        """同一格内容只入队一次，队列不会因为每帧重复投递而涨。"""
        vision.reset_vision_caches()
        vision._analyze_backpack_fast(self.image, self.scene.regions, None)
        with vision._FAMILY_PENDING_LOCK:
            after_first = len(vision._FAMILY_PENDING)
        vision._analyze_backpack_fast(self.image, self.scene.regions, None)
        with vision._FAMILY_PENDING_LOCK:
            after_second = len(vision._FAMILY_PENDING)
        self.assertGreater(after_first, 0, "首帧应该有格子入队")
        self.assertLessEqual(after_second, after_first, "重复帧不该继续往队列里堆")
        self.assertLessEqual(after_first, vision.FAMILY_QUEUE_MAX)
        vision.wait_for_family_identification(FAMILY_WAIT_SECONDS)


class FamilyCacheMatchTest(unittest.TestCase):
    """缓存必须按**距离**查，不能按字符串相等查。

    实机捕获到的画面每帧都有 1~3 像素的位移和亮度微变，同一个 slot 矩形里的
    读数「精确相同」只占 68.6%，但「距离 ≤0.14」占 99.9%。旧代码用
    ``fingerprint in _FAMILY_CACHE`` 精确查，于是每帧都判「没认出」：
    ``observations.jsonl`` 里 3402 次背包读数，``subtype`` 非空 **0 次**。
    """

    def setUp(self) -> None:
        vision.reset_family_cache()

    def test_one_byte_drift_still_hits(self) -> None:
        """64 字节里漂 1~8 个（距离 ≤0.125）必须命中。"""
        base = "08" * 64
        vision._remember_family(base, "存活", PaintCategory.GAMEPLAY, 0.72)
        for changed in (1, 4, 8):
            drifted = "09" * changed + "08" * (64 - changed)
            hit = vision._family_lookup(drifted)
            self.assertIsNotNone(hit, f"漂 {changed} 个字节就查不到了，等于每帧重认")
            self.assertEqual(hit[0], "存活")

    def test_far_apart_is_not_matched(self) -> None:
        """距离 0.5 是另一款颜料，绝不能被最近邻硬凑上去（那会贴错图案）。"""
        vision._remember_family("08" * 64, "存活", PaintCategory.GAMEPLAY, 0.72)
        self.assertIsNone(vision._family_lookup("0f" * 32 + "08" * 32))

    def test_exact_hit_is_cheapest_path(self) -> None:
        """精确命中不必建索引：同一个指纹重复查时索引应当根本没被创建。"""
        vision._remember_family("08" * 64, "存活", PaintCategory.GAMEPLAY, 0.72)
        self.assertIsNone(vision._FAMILY_INDEX, "还没人查过，不该有索引")
        self.assertIsNotNone(vision._family_lookup("08" * 64))

    def test_index_survives_cache_reset(self) -> None:
        """清缓存后索引必须一起失效，否则最近邻会越界（曾经的 IndexError）。"""
        vision._remember_family("08" * 64, "存活", PaintCategory.GAMEPLAY, 0.72)
        vision._family_lookup("09" * 64)  # 触发建索引
        vision.reset_family_cache()
        self.assertIsNone(vision._family_lookup("09" * 64))
        vision._remember_family("0a" * 64, "赛季", PaintCategory.GAMEPLAY, 0.72)
        self.assertEqual(vision._family_lookup("0a" * 64)[0], "赛季")


class BackpackFamilyDriftTest(unittest.TestCase):
    """帧间漂移下的真实链路：第一帧空框、后台认完、第二帧（画面已漂 1~2 像素）补上。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image = _shot()
        if cls.image is None:
            raise unittest.SkipTest(f"缺少实机截图: {BACKPACK_SHOT}")
        paint_icon.gallery()

    def _drifted(self, dy: int, dx: int, beta: int):
        import cv2
        import numpy as np

        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(
            self.image,
            matrix,
            (self.image.shape[1], self.image.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        if beta:
            moved = cv2.convertScaleAbs(moved, alpha=1.0, beta=beta)
        return moved

    def test_second_frame_survives_one_pixel_drift(self) -> None:
        """画面轻微漂移时，第二帧就该把大部分格子补上。

        实测（``_diag_drift_distance``）：同一格的指纹在**完全不漂**时距离为 0、
        只改亮度时 ≤0.016，漂 1 像素 mean 0.11 / 4 格超 0.14，漂 2 像素 9 格超。
        所以「漂 1 像素」是这套指纹还能用的边界：允许少数细节多的图案（宝箱、
        机械之神）晚一帧由后台补认，但不允许整体回到「永远空框」。
        """
        vision.reset_vision_caches()
        first = vision.analyze_scene_realtime(self.image)
        self.assertIsNotNone(first.backpack)
        self.assertTrue(
            all(paint.subtype is None for paint in first.backpack.paints),
            "首帧仍然不该有家族",
        )
        self.assertTrue(vision.wait_for_family_identification(FAMILY_WAIT_SECONDS))

        second = vision.analyze_scene_realtime(
            self._drifted(1, 1, 3), previous_scene=first
        )
        self.assertIsNotNone(second.backpack)
        names = [paint.subtype for paint in second.backpack.paints]
        known = sum(1 for name in names if name)
        wrong = [
            (index, name)
            for index, name in enumerate(names)
            if name and name != EXPECTED_FAMILIES[index]
        ]
        self.assertEqual(wrong, [], f"漂移后贴错了图案：{wrong}")
        self.assertGreaterEqual(
            known,
            10,
            f"画面漂 1 像素后只认出 {known}/15，说明缓存又是按精确键在查：{names}",
        )


#: 两张实拍图 × 人工读图得到的等级真值（3 行 × 5 列，行优先）。
#:
#: 真值来源：卡片右下角 "LvN" 角标。第二张图第三行被面板底边裁掉了角标，
#: 所以那 5 格的等级是按「色相 = 品质色」（银白 1 / 绿 2 / 青 3 / 紫 4）反推的，
#: 与用户报的「总共 1 个 4 级、1 个 3 级」一致。
LEVEL_SHOTS = (
    ("初始和背包宝箱.png", (3, 2, 2, 2) + (1,) * 11),
    ("背包第一页.png", (2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 2)),
)


class BackpackLevelTest(unittest.TestCase):
    """背包卡片等级：靠品质框的**色相**，不靠 ``category`` 猜框型。

    旧口径把卡片底纹的半透明深青当成了框色，矮人绿框（Lv2）被读成青框（Lv3）；
    而且它按 ``category`` 选框型——而 ``category`` 是从家族反推的，实拍里就有
    「玩法颜料配圆形框」的格子，按菱形采样会把银白框读成 Lv3。
    """

    def _read(self, filename: str) -> list[int]:
        image = read_image(SCREENSHOT_ROOT / filename)
        vision.reset_vision_caches()
        vision.analyze_scene_realtime(image)
        vision.wait_for_family_identification(FAMILY_WAIT_SECONDS)
        scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene.backpack, f"{filename} 里应该有背包面板")
        return [paint.level for paint in scene.backpack.paints]

    def test_levels_match_what_the_corner_labels_say(self) -> None:
        for filename, truth in LEVEL_SHOTS:
            with self.subTest(filename=filename):
                self.assertEqual(self._read(filename), list(truth), filename)

    def test_frame_color_reading_ignores_the_category_guess(self) -> None:
        """同一格，``category`` 传什么都要读出同一个等级。

        这才是「色相=等级」该有的性质：等级写在框上，跟「这是怪物还是玩法」无关。
        旧实现把两者绑死（gameplay 只量菱形环带），category 一错等级就跟着错。
        """
        image = read_image(SCREENSHOT_ROOT / "背包第一页.png")
        vision.reset_vision_caches()
        scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene.backpack)
        for index, paint in enumerate(scene.backpack.paints):
            slot = paint.slot_rect
            crop = image[slot.y : slot.bottom, slot.x : slot.right]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            readings = {
                category: vision._detect_backpack_level_by_frame_color(hsv, category)[0]
                for category in (
                    PaintCategory.MONSTER,
                    PaintCategory.GAMEPLAY,
                    PaintCategory.UNKNOWN,
                )
            }
            self.assertEqual(
                len(set(readings.values())),
                1,
                f"第 {index} 格读出了不同等级：{readings}",
            )


class BackpackLowTextureFamilyTest(unittest.TestCase):
    """低纹理图标（``觉醒者``）靠**几何核验**认出来，不是靠放宽点数门槛。

    实拍 ``jietu/背包第一页.png`` 第 5、6 格是两张 ``觉醒者``：图案只有一坨白
    色椭圆 + 橙色碎片，官方标准照也只提出 143 个描述子（全库最少），在
    ``ICON_MATCH_SIZE=160`` 下只拿到 33 / 39 个好点，卡在 ``ICON_MIN_GOOD=40``
    门外——界面上就只剩一圈等级框、看不见图案（用户 2026-11-18 报的「未识别」）。

    修法不是把 40 改成 30（那是拿假阳换召回），而是换判据：真匹配的 30 来个点
    能被**同一个平移+等比缩放**解释（内点比 0.97、缩放 0.95、旋转 ≈0°），
    假匹配（``邪影``）估出来的变换是乱的（缩放 0.57、旋转 169°）。
    """

    #: 3 行 × 5 列，行优先，人工读图得到的家族真值。
    FAMILIES = (
        "矮人",
        "沙怪",
        "沙怪",
        "鼠人",
        "觉醒者",
        "觉醒者",
        "黑帮",
        "黑帮",
        "伊吉",
        "伊吉",
        "沙怪",
        "沙怪",
        "赛季",
        "赛季",
        "存活",
    )

    def test_every_family_is_named_on_this_page(self) -> None:
        image = read_image(SCREENSHOT_ROOT / "背包第一页.png")
        vision.reset_vision_caches()
        vision.analyze_scene_realtime(image)
        self.assertTrue(vision.wait_for_family_identification(FAMILY_WAIT_SECONDS))
        scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene.backpack)
        self.assertEqual(
            tuple(paint.subtype for paint in scene.backpack.paints),
            self.FAMILIES,
        )

    def test_awakened_slots_pass_by_geometry_not_point_count(self) -> None:
        """负例自证：这两格的点数**本来就没到线**，是几何判据放过的。

        同时锁住「别偷偷把 ``ICON_MIN_GOOD`` 调低」——调低的话点数就过线、
        ``verified`` 会是 ``False``，这条会红。
        """
        image = read_image(SCREENSHOT_ROOT / "背包第一页.png")
        gallery = paint_icon.gallery()
        vision.reset_vision_caches()
        vision.analyze_scene_realtime(image)
        self.assertTrue(vision.wait_for_family_identification(FAMILY_WAIT_SECONDS))
        scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene.backpack)

        checked = 0
        for paint in scene.backpack.paints:
            if paint.subtype != "觉醒者":
                continue
            rect = paint.slot_rect
            crop = image[rect.y : rect.bottom, rect.x : rect.right]
            match = gallery.identify(crop, center_ratio=paint_frame.ICON_SCALE)
            self.assertIsNotNone(match)
            self.assertEqual(match.family, "觉醒者")
            self.assertTrue(match.verified, "低纹理图标只能靠几何判据认出")
            self.assertLess(
                match.good,
                paint_icon.ICON_MIN_GOOD,
                "这两格的点数本来就不到线，别把门槛调低来「修」",
            )

            # 负例：亚军 邪影 的变换是散的，几何核验必须拒绝。
            detail = gallery.verify(crop, "邪影", center_ratio=paint_frame.ICON_SCALE)
            self.assertIsNotNone(detail, "亚军也要能估出变换，否则这条负例是假的")
            _total, _inliers, scale, rotation = detail
            low, high = paint_icon.ICON_VERIFY_SCALE_RANGE
            plausible = low <= scale <= high and abs(rotation) <= paint_icon.ICON_VERIFY_MAX_ROTATION_DEG
            self.assertFalse(
                plausible,
                f"邪影 的变换不该像真的：(scale={scale:.2f}, rot={rotation:.1f}°)",
            )
            checked += 1
        self.assertEqual(checked, 2, "这张实拍里就两格 觉醒者")


if __name__ == "__main__":
    unittest.main()
