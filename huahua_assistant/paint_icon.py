"""用官方图标直接认家族——27 个家族一张不缺，不再等截图攒模板。

## 为什么需要这一层

:mod:`paint_kind` 走的是「模板库」路子：从截图里抠一块图案存起来，下次拿
NCC + 颜色直方图比。这条路准，但**每个家族都得先见过一次**才能建模板，
而解包产物里没有美术资源（只有图标名 ``ColorMonster9_128``），所以一直在
「等实机截图」。剩下 11 个家族（黑帮、沙怪、幽灵、龙骨……）就这么卡着。

现在把 S8 的 PAK 解开、用 Oodle 解压 + BC7 解码拿到了 **30 张 256×256 图标原图**
（见 :data:`ICON_DIR`），其中 27 张是家族标准照，等于给每个家族
发了一张**标准照**。棋盘中央画的本来就是这张图，于是可以直接对着标准照认人，
不用再靠「先见过」：

============  ==========  ==========
指标          命中家族    其它家族
============  ==========  ==========
好匹配点数    148~185     12~32
好匹配占比    0.22~0.26   0.02~0.05
============  ==========  ==========

差了一个数量级，所以判据可以卡得很硬：占比 ≥0.08、好点数 ≥40、
领先亚军 ≥2 倍。三条同时满足才认，宁可不认也不认错。

## 为什么用 ORB 而不是继续用 NCC

模板库的 NCC 要求「同一块图案、同一个取景比例」；图标原图是满幅绘制、
没有菱形外框、还带渐变底，跟棋盘上的 34%~44% 小图根本对不上（实测最高
0.30，比随机好不了多少）。ORB 比的是**局部纹理的关键点**，天生对缩放、
裁剪、底色不敏感——棋盘上那点小图缩到 256 见方，仍然能跟上百个关键点对上。

## 用在哪

1. :func:`identify` —— 模板库认不出时兜底（``vision`` 的慢路径里接的）；
2. ``python -m huahua_assistant.paint_icon identify <图>`` —— 整盘体检，
   把每一格认成哪个家族、分数多少、亚军是谁全打出来；
3. 命名未知簇时当证据用：未知簇跟哪个图标能对上 150 个关键点，
   就是哪个家族，不用再猜。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import paint_kind

#: 官方图标目录。30 张 256×256 RGBA PNG，全部从 S8 的 PAK 里解出来
#: （``Content/Art/Season/S8/UI/Textures/Common/Icon/GamePlay/``），
#: 与 tlidb 网页图标逐像素比对过：27 张可比的 MAE 全部 <6，属 BC7 压缩噪声。
#:
#: 其中 27 张是**棋盘家族**的标准照；另外 3 张不画在棋盘上，见 :data:`ROLES_PATH`：
#: ``ColorPlay2`` = 放入虚空后的形态、``ColorPlay16`` = 通关后宠物特性、
#: ``ColorMonster4`` = 用途待确认。它们不进识别库，避免把状态认成家族。
ICON_DIR = Path(__file__).resolve().parent / "assets" / "paint_icons"

#: 非棋盘图标的角色表。``icon_names()`` 只收 ``role=family`` 的那些。
ROLES_PATH = ICON_DIR / "roles.json"

#: 匹配前把两边都缩到这个边长再提关键点。
ICON_MATCH_SIZE = 160
#: ORB 特征点上限。128 的图标放大到 256 后 600 个点足够，再多只是变慢。
ICON_FEATURES = 320
#: ORB 的 FAST 阈值。默认 20 会漏掉深色图案里的低对比关键点（黑帮、邪影都偏暗）。
ICON_FAST_THRESHOLD = 8
#: Lowe 比值判据：最近邻距离 < 次近邻 × 该系数才算「好匹配」。
ICON_RATIO_TEST = 0.78
#: 好匹配点数下限。
ICON_MIN_GOOD = 40
#: 好匹配点数 / 本图关键点数 的下限。
#:
#: 命中家族实测 0.15~0.26，但「机械之神」那类图案细节少、对比低，只到 0.08
#: （43 个好点、亚军 21），所以下限压到 0.08；真正挡住误判的是下面那条倍数。
ICON_MIN_SHARE = 0.08
#: 领先亚军的最小倍数。实测命中 3.6~8.4 倍，最弱的一次（机械之神 43 : 21）约 2.05。
ICON_MIN_MARGIN = 2.0

# -- 第二道判据：几何核验 ---------------------------------------------------
#
# 上面三条都是「数点数」，对**低纹理图标**不成立。``觉醒者`` 只有 143 个描述子
# （全库最少），图案是一坨白色椭圆 + 橙色碎片，在 ``ICON_MATCH_SIZE=160`` 下只
# 拿到 33 个好点，卡在 :data:`ICON_MIN_GOOD` 门外 → 实拍 ``背包第一页.png`` 的
# 第 5、6 格一直报「认不出」，界面上就只剩一圈等级框。纹理丰富的亚军
# （``邪影`` 深蓝翼盔）反而能靠背景渐变凑到 11~12 个。
#
# 换成**几何**口径就分得干干净净：真匹配是「同一张图」，40 来个好点能落进同一个
# 平移+等比缩放变换；假匹配是噪声零星撞上的，估出来的变换是乱的。实测
# （同一张实拍，``center_ratio=paint_frame.ICON_SCALE``）：
#
# ========  ==========  ============  ======  ==========  ==========
# 家族       好匹配点    RANSAC 内点    缩放      旋转        判定
# ========  ==========  ============  ======  ==========  ==========
# 觉醒者     33          32            0.95    -0.0°       接受
# 邪影       11           7            0.69    125.8°      拒绝
# ========  ==========  ============  ======  ==========  ==========
#
# 全部 15 格真命中的内点比都在 0.94 以上、缩放 0.94~0.96、|旋转| ≤0.5°；
# 全部亚军要么缩放 0.14~0.80，要么旋转几十度。两者完全不重叠，阈值取中间。
#
# 只在**主判据没过**时才跑这一道，命中路径一个像素都不多算。
ICON_VERIFY_MIN_INLIERS = 12
ICON_VERIFY_MIN_INLIER_RATIO = 0.6
ICON_VERIFY_SCALE_RANGE = (0.75, 1.25)
ICON_VERIFY_MAX_ROTATION_DEG = 20.0
#: 进几何核验之前，好点还得领先亚军这么多倍——否则谁都能撞进核验。
ICON_VERIFY_MIN_RUNNER_MARGIN = 1.5


@dataclass(frozen=True)
class IconMatch:
    """一次「按图标认家族」的结果。"""

    #: 认出的家族名（``黑帮``）。
    family: str
    #: 好匹配点数 / 本图关键点数（0~1）。
    share: float
    #: 好匹配点数。
    good: int
    #: 亚军家族名。
    runner_up: str | None
    #: 亚军的好匹配点数。
    runner_good: int
    #: RANSAC 内点数（只有过了几何核验才有值，否则 0）。
    inliers: int = 0
    #: 是不是靠第二道判据（几何核验）认出来的。
    #:
    #: ``True`` 表示「点数没到线，但 40 来个好点能被同一个变换解释」——
    #: 见 :data:`ICON_VERIFY_MIN_INLIERS`。界面上不必区分，排查时要看得见。
    verified: bool = False

    @property
    def margin(self) -> float:
        """领先亚军倍数。亚军为 0 时给一个大数，避免除零。"""
        return float(self.good) / max(float(self.runner_good), 1.0)


def icon_names() -> dict[str, str]:
    """图标文件名（不含扩展名）→ 家族名。名字统一从 ``paint_catalog.json`` 取，
    避免这张对照表在两处各写一遍、改一处漏一处。

    只收**棋盘家族**。``roles.json`` 里登记过的状态图标（虚空、宠物特性）
    和用途待确认的那张不进这个表——它们不是家族，混进来会让识别把状态报成家族。
    """
    special = special_roles()
    result: dict[str, str] = {}
    for entry in paint_kind.catalog().values():
        icon = entry.get("icon")
        name = entry.get("name")
        if icon and name and icon not in special:
            result[icon] = name
    return result


def special_roles() -> dict[str, dict]:
    """``roles.json``：图标名 → {role, name, note}。

    ``role`` 取值：``voided``（放入虚空后的形态）、``pet_trait``（通关后宠物特性）、
    ``unconfirmed``（用途待确认）。这三类都**不是棋盘家族**。
    """
    try:
        payload = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def icon_files(directory: Path | None = None) -> list[Path]:
    """目录里所有图标文件（``.png`` / ``.webp`` 都收，按文件名排序）。"""
    root = Path(directory) if directory is not None else ICON_DIR
    return sorted(root.glob("*.webp")) + sorted(root.glob("*.png"))


def _read_icon(path: Path) -> np.ndarray | None:
    """Windows 上 ``cv2.imread`` 读不了中文路径，统一走 ``imdecode``。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def _center_crop(image: np.ndarray, ratio: float) -> np.ndarray:
    """裁出中心一块正方形区域，``ratio`` 是边长占比。

    **为什么需要它**：官方图标是满幅绘制的，而游戏格子里图案只占品质框的
    ``paint_frame.ICON_SCALE``（128/192 ≈ 0.667），外面还套着一圈彩色等级框、
    右下角压着 ``LvN`` 角标。整格直接比对时这些无关内容把图案稀释掉——

    ==============  ================  ==========================
    背包 15 格        good 点数          过硬判据（≥40 点）
    ==============  ================  ==========================
    整格比对          24 ~ 70           7 / 15
    裁到 0.667        55 ~ 93          15 / 15
    ==============  ================  ==========================

    裁完领先亚军的倍数也从 2.5~8.8 提到 5.4~11.1，所以这不是「放宽阈值」换来的
    召回，是把无关像素去掉换来的信噪比。棋盘格不必裁（图案本来占得更多，
    且模板库是主力），所以默认 ``None`` 保持原行为。
    """
    if image is None or image.size == 0:
        return image
    ratio = float(min(max(ratio, 0.2), 1.0))
    if ratio >= 1.0:
        return image
    height, width = image.shape[:2]
    side_h = max(1, int(round(height * ratio)))
    side_w = max(1, int(round(width * ratio)))
    top = (height - side_h) // 2
    left = (width - side_w) // 2
    return image[top : top + side_h, left : left + side_w]


class IconGallery:
    """官方图标库：载入 27 张图标，预先算好 ORB 描述子。"""

    def __init__(self, icon_dir: Path | None = None, *, use_cache: bool = True) -> None:
        self.icon_dir = Path(icon_dir) if icon_dir is not None else ICON_DIR
        self.orb = cv2.ORB_create(nfeatures=ICON_FEATURES, fastThreshold=ICON_FAST_THRESHOLD)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.families: dict[str, np.ndarray] = {}
        #: 家族 → 模板关键点坐标 ``(N, 2) float32``。几何核验要把描述子索引换回
        #: 坐标才能估变换，光有描述子做不到——所以缓存里要一起存。
        self.points: dict[str, np.ndarray] = {}
        if use_cache and self._load_cache():
            self._warm_up()
            return
        self._build()
        if use_cache:
            self._save_cache()
        self._warm_up()

    def _warm_up(self) -> None:
        """先把 ORB 的懒初始化做掉。

        第一次 ``detectAndCompute`` 要 0.7 秒（内部建金字塔、分配缓冲都在头一次），
        这笔钱如果落在帧里，第一帧就会莫名超预算——``斜对角.png`` 实测就是这么
        被顶到 1.14 秒的。之后每格只要 ~10ms。

        **必须用一张真实图标来预热**：拿纯色空图跑，ORB 找不到任何角点，内部那条
        懒初始化路径根本不会走完，下一帧照样付 0.72 秒。这一点是实测出来的。
        """
        try:
            sample = None
            for path in icon_files(self.icon_dir)[:1]:
                sample = _read_icon(path)
            if sample is None:
                sample = np.zeros((ICON_MATCH_SIZE, ICON_MATCH_SIZE), dtype=np.uint8)
            else:
                sample = cv2.resize(sample, (ICON_MATCH_SIZE, ICON_MATCH_SIZE))
            self.orb.detectAndCompute(sample, None)
        except cv2.error:
            pass

    # -- 描述子缓存 ---------------------------------------------------
    #
    # 27 张图标现场提一遍 ORB 要 **0.95 秒**（fastThreshold 压到 8 是为了认得出
    # 深色图案，代价是候选角点多），冷启动时这笔时间直接吃掉了交互延迟预算
    # （``test_realtime_analysis_meets_interactive_latency_budget`` 因此从 0.75s
    # 掉到 1.34s）。描述子只跟图标文件和 OpenCV 版本有关，算一次存成 npz，
    # 之后 ``np.load`` 不到 10ms。

    def _cache_path(self) -> Path:
        return self.icon_dir / f"descriptors_{ICON_MATCH_SIZE}_{ICON_FEATURES}_{ICON_FAST_THRESHOLD}.npz"

    def _cache_key(self) -> str:
        """缓存失效判据：OpenCV 版本变了、图标文件变了，描述子就得重算。"""
        stamps = []
        for path in icon_files(self.icon_dir):
            try:
                stamps.append(f"{path.stem}:{path.stat().st_size}:{int(path.stat().st_mtime)}")
            except OSError:
                stamps.append(path.stem)
        # ``kp1``：缓存里开始存模板关键点坐标（几何核验要用）。旧缓存没有这一项，
        # 靠这个标记让它自然失效、重建一次，而不是在核验时抛 KeyError。
        return "|".join(
            ["kp1", cv2.__version__, str(ICON_MATCH_SIZE), str(ICON_FEATURES), *stamps]
        )

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"].item() if data["key"].shape == () else "") != self._cache_key():
                    return False
                names = [str(name) for name in data["names"]]
                loaded = {name: data[f"d{index}"] for index, name in enumerate(names)}
                points = {name: data[f"p{index}"] for index, name in enumerate(names)}
        except (OSError, KeyError, ValueError):
            return False
        if not loaded:
            return False
        self.families = loaded
        self.points = points
        return True

    def _save_cache(self) -> None:
        if not self.families:
            return
        names = list(self.families)
        payload: dict[str, np.ndarray] = {
            "key": np.array(self._cache_key()),
            "names": np.array(names),
        }
        for index, name in enumerate(names):
            payload[f"d{index}"] = self.families[name]
            points = self.points.get(name)
            if points is not None:
                payload[f"p{index}"] = points
        try:
            np.savez_compressed(self._cache_path(), **payload)
        except OSError:
            pass  # 只读安装目录：缓存存不下就算了，下次再算一遍

    def _build(self) -> None:
        names = icon_names()
        for path in icon_files(self.icon_dir):
            family = names.get(path.stem)
            if not family:
                continue
            gray = _read_icon(path)
            if gray is None:
                continue
            gray = cv2.resize(gray, (ICON_MATCH_SIZE, ICON_MATCH_SIZE), interpolation=cv2.INTER_CUBIC)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors):
                self.families[family] = descriptors
                self.points[family] = np.float32([kp.pt for kp in keypoints])

    def __len__(self) -> int:
        return len(self.families)

    def scores(
        self,
        tile_bgr: np.ndarray,
        *,
        center_ratio: float | None = None,
    ) -> dict[str, tuple[float, int]]:
        """家族 → ``(好匹配占比, 好匹配点数)``。认不出（没关键点）时返回空字典。

        ``center_ratio`` 给定就先按这个比例裁中心再匹配，见 :func:`_center_crop`。
        """
        if not self.families or tile_bgr is None or tile_bgr.size == 0:
            return {}
        if center_ratio is not None:
            tile_bgr = _center_crop(tile_bgr, center_ratio)
            if tile_bgr.size == 0:
                return {}
        if tile_bgr.ndim == 3:
            gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_bgr
        gray = cv2.resize(gray, (ICON_MATCH_SIZE, ICON_MATCH_SIZE), interpolation=cv2.INTER_CUBIC)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 8:
            return {}
        result: dict[str, tuple[float, int]] = {}
        for family, icon_descriptors in self.families.items():
            pairs = self.matcher.knnMatch(descriptors, icon_descriptors, k=2)
            good = sum(
                1
                for pair in pairs
                if len(pair) == 2 and pair[0].distance < ICON_RATIO_TEST * pair[1].distance
            )
            result[family] = (good / float(len(keypoints)), good)
        return result

    def identify(
        self,
        tile_bgr: np.ndarray,
        *,
        center_ratio: float | None = None,
    ) -> IconMatch | None:
        """认一个格子。

        判定两道：

        1. **点数判据**（快）：好匹配点 ≥ :data:`ICON_MIN_GOOD`、占比 ≥
           :data:`ICON_MIN_SHARE`、领先亚军 ≥ :data:`ICON_MIN_MARGIN` 倍。
        2. **几何判据**（只在第 1 道没过时跑）：好点能不能被同一个「平移+等比
           缩放」变换解释。低纹理图标（``觉醒者``）点数永远到不了 40，但内点比
           0.94、缩放 0.95、旋转 0° —— 和假匹配的散乱变换不重叠，见
           :data:`ICON_VERIFY_MIN_INLIERS`。
        """
        scores = self.scores(tile_bgr, center_ratio=center_ratio)
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
        family, (share, good) = ranked[0]
        runner_family, (_, runner_good) = ranked[1] if len(ranked) > 1 else (None, (0.0, 0))
        if (
            good >= ICON_MIN_GOOD
            and share >= ICON_MIN_SHARE
            and good >= ICON_MIN_MARGIN * max(runner_good, 1)
        ):
            return IconMatch(
                family=family,
                share=share,
                good=good,
                runner_up=runner_family,
                runner_good=runner_good,
            )
        # 第二道：只在「差一点点」的候选上做几何核验。先过这几条廉价的闸，
        # 免得每格都白跑一次 ORB+RANSAC（那笔钱比点数判据本身还贵）。
        if (
            good >= ICON_VERIFY_MIN_INLIERS
            and share >= ICON_MIN_SHARE
            and good >= ICON_VERIFY_MIN_RUNNER_MARGIN * max(runner_good, 1)
        ):
            detail = self.verify(tile_bgr, family, center_ratio=center_ratio)
            if detail is not None:
                total, inliers, scale, rotation = detail
                low, high = ICON_VERIFY_SCALE_RANGE
                if (
                    inliers >= ICON_VERIFY_MIN_INLIERS
                    and inliers >= ICON_VERIFY_MIN_INLIER_RATIO * total
                    and low <= scale <= high
                    and abs(rotation) <= ICON_VERIFY_MAX_ROTATION_DEG
                ):
                    return IconMatch(
                        family=family,
                        share=share,
                        good=good,
                        runner_up=runner_family,
                        runner_good=runner_good,
                        inliers=inliers,
                        verified=True,
                    )
        return None

    def verify(
        self,
        tile_bgr: np.ndarray,
        family: str,
        *,
        center_ratio: float | None = None,
    ) -> tuple[int, int, float, float] | None:
        """几何核验：这个家族的好匹配点能不能被同一个相似变换解释。

        返回 ``(好匹配点数, RANSAC 内点数, 缩放, 旋转角度)``；点太少或估不出变换
        时返回 ``None``。缩放按单位矩阵的列范数取（等比缩放假设下即缩放系数），
        旋转取变换的旋转角（度）。

        为什么不是「放宽 :data:`ICON_MIN_GOOD`」：那是拿真阳的召回换假阳。这里
        换的是判据种类——「像不像同一张图」本来就不该用点数多少来衡量。
        """
        points = self.points.get(family)
        descriptors = self.families.get(family)
        if points is None or descriptors is None or tile_bgr is None or tile_bgr.size == 0:
            return None
        if center_ratio is not None:
            tile_bgr = _center_crop(tile_bgr, center_ratio)
            if tile_bgr.size == 0:
                return None
        gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY) if tile_bgr.ndim == 3 else tile_bgr
        gray = cv2.resize(gray, (ICON_MATCH_SIZE, ICON_MATCH_SIZE), interpolation=cv2.INTER_CUBIC)
        keypoints, tile_descriptors = self.orb.detectAndCompute(gray, None)
        if tile_descriptors is None or len(keypoints) < 8:
            return None
        pairs = self.matcher.knnMatch(tile_descriptors, descriptors, k=2)
        good = []
        for first, second in (pair for pair in pairs if len(pair) == 2):
            if first.distance < ICON_RATIO_TEST * second.distance:
                good.append(first)
        if len(good) < 4:
            # 4 个点是 ``estimateAffinePartial2D`` 的最低要求；「够不够格信」由
            # 调用方按 :data:`ICON_VERIFY_MIN_INLIERS` 判——这里只负责把几何量出来。
            return None
        source = np.float32([keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
        target = points[np.array([match.trainIdx for match in good], dtype=np.int64)]
        target = np.asarray(target, dtype=np.float32).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
        )
        if matrix is None or inliers is None:
            return None
        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        return len(good), int(inliers.sum()), scale, rotation


#: 一帧里最多给多少个格子跑图标兜底。
#:
#: 单格 26ms：提关键点 10ms + 跟 27 张图标比 16ms。整盘 36 格全跑就是 0.9s，
#: 直接把实时链路的 0.75s 预算吃穿（``其他.png`` 实测 0.30s → 1.24s）。
#: 模板库命中时根本不会走到这里，走的都是「没见过的新家族」，一帧认清 6 格
#: 已经足够——剩下的下一帧接着认，而模板一旦入过库就再也不花这笔钱。
ICON_BUDGET_CELLS = 6
#: 一帧里图标兜底的时间上限（秒）。格数上限之外的第二道闸，防止机器慢时爆表。
ICON_BUDGET_SECONDS = 0.10


class IconBudget:
    """一帧一次的配额：最多认 ``cells`` 格、最多花 ``seconds`` 秒。

    计时从**第一次真正认图标**才开始，不是从重置开始。原先 ``reset()`` 就把
    deadline 定死 0.1 秒，可它前面还要跑 ``paint_kind.read_kind``（模板库 NCC，
    冷启动要几十毫秒）——机器一慢，等到该认图标时预算已经过期，图标兜底被饿死，
    得到的 ``subtype`` 是 ``None``。``test_mechanical_god_identified_by_icon_not_color``
    就是这么偶发变红的（同一个格子在单独跑时认得出、整批跑时认不出）。
    """

    def __init__(self, cells: int = ICON_BUDGET_CELLS, seconds: float = ICON_BUDGET_SECONDS) -> None:
        self.cells = cells
        self.seconds = seconds
        self._left = cells
        self._deadline = 0.0

    def reset(self) -> None:
        self._left = self.cells
        # 0 表示「还没开始计时」，第一格落地时才起算。
        self._deadline = 0.0

    def take(self) -> bool:
        if self._left <= 0:
            return False
        now = time.monotonic()
        if self._deadline <= 0.0:
            self._deadline = now + self.seconds
        elif now > self._deadline:
            return False
        self._left -= 1
        return True


_BUDGET = IconBudget()


def reset_budget(*, allow: bool = True) -> None:
    """每分析一帧调用一次（``vision`` 在分类棋盘前重置）。

    ``allow=False`` 给增量更新路径用：``analyze_quick`` 每帧只比上一帧变了几格，
    预算再宽也架不住帧率高，而且那条路径上的家族本来就该由模板库负责。
    """
    _BUDGET.reset()
    if not allow:
        _BUDGET._left = 0


_GALLERY: IconGallery | None = None
_GALLERY_LOCK = threading.Lock()


def gallery() -> IconGallery:
    """全局单例。载入 27 张图标约 30ms，只在第一次调用时付。

    加了锁是因为界面启动时会用后台线程预热（:func:`warm_async`），
    而第一帧也可能同时在主线程里碰它。
    """
    global _GALLERY
    if _GALLERY is None:
        with _GALLERY_LOCK:
            if _GALLERY is None:
                _GALLERY = IconGallery()
    return _GALLERY


def warm_async() -> threading.Thread:
    """后台把图标库载入好。

    第一次 ``detectAndCompute`` 里 OpenCV 要建金字塔、分配缓冲，实测冷启一次
    1.3~3 秒（背包一帧要认 15 格，这笔钱落在帧里足够把首帧顶到 4 秒）。界面
    启动时先花在后台线程上，用户拖好窗口时它已经好了。
    """
    thread = threading.Thread(target=gallery, name="paint-icon-warm", daemon=True)
    thread.start()
    return thread


def identify(
    tile_bgr: np.ndarray,
    *,
    budgeted: bool = False,
    center_ratio: float | None = None,
) -> IconMatch | None:
    """认一个格子。``budgeted=True`` 时受一帧配额限制（给实时链路用）。

    ``center_ratio`` 透传给 :meth:`IconGallery.identify`——背包格要按
    ``paint_frame.ICON_SCALE`` 裁中心，理由见 :func:`_center_crop`。
    """
    if budgeted and not _BUDGET.take():
        return None
    return gallery().identify(tile_bgr, center_ratio=center_ratio)


# ---------------------------------------------------------------- 命令行


def _read_image(path: Path) -> np.ndarray | None:
    return paint_kind._read_image(path)


def _scan(path: Path) -> list[tuple[int, int, str | None, int | None, IconMatch | None, str | None]]:
    """跑一遍场景分析，再逐格用图标认家族。"""
    from . import vision

    image = _read_image(path)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    scene = vision.analyze_scene(image)
    if scene is None or scene.board is None:
        raise SystemExit("没找到棋盘")
    rows = []
    for tile in scene.board.tiles:
        rect = tile.rect
        cell = image[rect.y : rect.bottom, rect.x : rect.right]
        match = identify(cell)
        rows.append((tile.row, tile.col, tile.paint_subtype, tile.paint_level, match, tile.state.value))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    path = Path(args[0])
    print(f"图标库: {len(gallery())} 个家族")
    for row, col, subtype, level, match, state in _scan(path):
        if state != "paint":
            continue
        template = subtype or "—"
        icon = f"{match.family} {match.good}点/{match.share:.2f}" if match else "—"
        agree = "✓" if match and subtype == match.family else " "
        print(f"  r{row}c{col} Lv{level or '?'}  模板:{template:<6} 图标:{icon:<16}{agree}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
