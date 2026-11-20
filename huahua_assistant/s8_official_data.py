"""S8 赛季「画中雪原 · 作画」官方配置数据（**自动生成，请勿手改**）。

生成脚本：.workbuddy/tmp/gen_s8_official_data.py
中间证据：.workbuddy/refs/official_S8_full.json

数据来源：客户端 PAK 的 Lua 配置表常量池 + 官方多语言表，二者按记录 ID 对齐。
"""

from __future__ import annotations

from typing import NamedTuple


class S8Affix(NamedTuple):
    """一条可选笔触效果（S8_Gameplay_SelectableEffect）。"""

    id: int                 # 官方记录 ID
    brush: str              # TRANSFORM / STACK / SPREAD
    library_name: str       # 官方大类别名：转锋 / 堆砌 / 扩散
    text: str               # 官方 EffectDes 纯文案
    effect_type: int | None # 该记录首次出现时可见的 EffectType（去重会遮蔽重复值）
    params: tuple[int | str, ...]  # 记录内可见的其它参数常量
    weights: tuple[tuple[int, int], ...]  # 随机档位权重 (档位, 权重)


class S8PlotItem(NamedTuple):
    """一种颜料（S8_Gameplay_PlotItem）。ID = 家族4位 + 等级2位。"""

    id: int
    name: str               # 怪物颜料-鼠人 / 玩法颜料-宝箱
    family: str             # 家族名（不含等级）
    level: int              # 颜料等级 1..7
    category: str           # MONSTER / GAMEPLAY
    gameplay_des: str       # 该等级在雪原上生成什么


S8_SELECTABLE_EFFECTS: tuple[S8Affix, ...] = (
    S8Affix(1001, 'TRANSFORM', '转锋',
            '随机转变自身的作用方向',
            None, (),
            ()),
    S8Affix(1002, 'TRANSFORM', '转锋',
            '转变自身的作用方向为横向',
            None, (),
            ()),
    S8Affix(1003, 'TRANSFORM', '转锋',
            '转变自身的作用方向为纵向',
            None, (),
            ()),
    S8Affix(1004, 'TRANSFORM', '转锋',
            '转变自身的作用方向为相邻',
            None, (),
            ()),
    S8Affix(1005, 'TRANSFORM', '转锋',
            '转变自身的作用方向为对角斜向',
            None, (),
            ()),
    S8Affix(1006, 'TRANSFORM', '转锋',
            '转变自身的作用对象为另一类型',
            None, ('1:1000;2:1000;3:100',),
            ((1, 1000), (2, 1000), (3, 100))),
    S8Affix(1007, 'TRANSFORM', '转锋',
            '随机转变自身的内容，等级不变',
            None, (),
            ()),
    S8Affix(1008, 'TRANSFORM', '转锋',
            '随机转变自身的等级，内容不变',
            None, (),
            ()),
    S8Affix(1009, 'TRANSFORM', '转锋',
            '重新获得1个随机调色颜料',
            None, (),
            ()),
    S8Affix(1010, 'TRANSFORM', '转锋',
            '调色目标的进度要求减少1',
            103, (),
            ()),
    S8Affix(1011, 'TRANSFORM', '转锋',
            '调色目标的进度要求减少2',
            None, (),
            ()),
    S8Affix(1012, 'TRANSFORM', '转锋',
            '调色目标的进度要求减少3',
            None, (),
            ()),
    S8Affix(2001, 'STACK', '堆砌',
            '使自身等级提升1级',
            None, (),
            ()),
    S8Affix(2002, 'STACK', '堆砌',
            '使自身等级提升2级',
            None, (),
            ()),
    S8Affix(2003, 'STACK', '堆砌',
            '使自身等级提升3级',
            None, (),
            ()),
    S8Affix(2004, 'STACK', '堆砌',
            '对作用范围内的随机1个颜料额外提升1级',
            None, (),
            ()),
    S8Affix(2005, 'STACK', '堆砌',
            '对作用范围内的随机2个颜料额外提升1级',
            None, (),
            ()),
    S8Affix(2006, 'STACK', '堆砌',
            '对作用范围内的所有颜料额外提升1级',
            None, (),
            ()),
    S8Affix(2007, 'STACK', '堆砌',
            '对作用范围内的所有颜料额外提升2级',
            None, (),
            ()),
    S8Affix(2008, 'STACK', '堆砌',
            '对作用范围内的所有颜料额外提升3级',
            None, (),
            ()),
    S8Affix(2009, 'STACK', '堆砌',
            '将作用范围内的颜料转变为同等级的随机颜料',
            None, (),
            ()),
    S8Affix(2010, 'STACK', '堆砌',
            '将作用范围内的颜料转变为同等级的随机怪物颜料',
            204, ('0',),
            ()),
    S8Affix(2011, 'STACK', '堆砌',
            '将作用范围内的颜料转变为同等级的随机玩法颜料',
            None, (),
            ()),
    S8Affix(2012, 'STACK', '堆砌',
            '将作用范围内所有颜料的等级和内容替换为自身',
            None, (),
            ()),
    S8Affix(2013, 'STACK', '堆砌',
            '将全图中所有颜料的内容替换为自身',
            202, (),
            ()),
    S8Affix(2014, 'STACK', '堆砌',
            '对全图中与自身内容相同的颜料额外提升1级',
            None, (),
            ()),
    S8Affix(2015, 'STACK', '堆砌',
            '对全图中与自身内容相同的颜料额外提升2级',
            None, ('5',),
            ()),
    S8Affix(2016, 'STACK', '堆砌',
            '对全图中与自身内容相同的颜料额外提升3级',
            None, (),
            ()),
    S8Affix(2017, 'STACK', '堆砌',
            '随机增加一种作用范围，调色进度要求增加2',
            None, (),
            ()),
    S8Affix(2018, 'STACK', '堆砌',
            '随机增加一种作用范围，调色进度要求增加1',
            None, (),
            ()),
    S8Affix(2019, 'STACK', '堆砌',
            '随机增加一种作用范围',
            205, ('1:1000;2:1000;3:1000;4:1000;5:100',),
            ((1, 1000), (2, 1000), (3, 1000), (4, 1000), (5, 100))),
    S8Affix(2020, 'STACK', '堆砌',
            '使全图中怪物颜料的等级提升1级',
            None, (),
            ()),
    S8Affix(2021, 'STACK', '堆砌',
            '使全图中玩法颜料的等级提升1级',
            None, (),
            ()),
    S8Affix(2022, 'STACK', '堆砌',
            '使全图中所有颜料的等级提升1级',
            None, ('3',),
            ()),
    S8Affix(3001, 'SPREAD', '扩散',
            '将相邻范围中随机1个阻挡地块转变为空白地块',
            None, (4,),
            ()),
    S8Affix(3002, 'SPREAD', '扩散',
            '将横向范围中最近1个阻挡地块转变为空白地块',
            None, (),
            ()),
    S8Affix(3003, 'SPREAD', '扩散',
            '将纵向范围中最近1个阻挡地块转变为空白地块',
            None, (6,),
            ()),
    S8Affix(3004, 'SPREAD', '扩散',
            '将对角斜向范围中最近1个阻挡地块转变为空白地块',
            None, (7,),
            ()),
    S8Affix(3005, 'SPREAD', '扩散',
            '将距离最近的1个阻挡地块转变为空白地块',
            301, ('1', 2),
            ()),
    S8Affix(3006, 'SPREAD', '扩散',
            '将作用范围内的阻挡地块转变为空白地块',
            None, (),
            ()),
    S8Affix(3007, 'SPREAD', '扩散',
            '将全图随机1个阻挡地块转变为空白地块',
            None, (8,),
            ()),
    S8Affix(3008, 'SPREAD', '扩散',
            '将全图随机2个阻挡地块转变为空白地块',
            None, ('2',),
            ()),
    S8Affix(3009, 'SPREAD', '扩散',
            '将全图所有阻挡地块转变为空白地块',
            None, (),
            ()),
    S8Affix(3010, 'SPREAD', '扩散',
            '可放置于阻挡地块中',
            None, (),
            ()),
    S8Affix(3011, 'SPREAD', '扩散',
            '随机转变全图所有阻挡地块的位置至其他空白地块',
            304, (),
            ()),
    S8Affix(3012, 'SPREAD', '扩散',
            '可放置于虚空地块中，但会失去自身内容',
            None, (),
            ()),
    S8Affix(3013, 'SPREAD', '扩散',
            '可放置于已放置颜料的空白地块中，但会失去自身内容',
            302, (),
            ()),
    S8Affix(3014, 'SPREAD', '扩散',
            '将全图随机1个虚空地块转变为空白地块',
            None, (),
            ()),
    S8Affix(3015, 'SPREAD', '扩散',
            '将全图随机2个虚空地块转变为空白地块',
            303, (),
            ()),
    S8Affix(3016, 'SPREAD', '扩散',
            '将作用范围内的失色地块转变为空白地块',
            None, (),
            ()),
    S8Affix(3017, 'SPREAD', '扩散',
            '可放置于失色地块中',
            None, (),
            ()),
    S8Affix(3018, 'SPREAD', '扩散',
            '将距离最近的1个失色地块转变为空白地块',
            305, (),
            ()),
    S8Affix(4001, 'TRANSFORM', '转锋',
            '转变自身的作用方向为横向',
            None, (),
            ()),
    S8Affix(4002, 'TRANSFORM', '转锋',
            '转变自身的作用方向为相邻',
            None, (),
            ()),
    S8Affix(4003, 'TRANSFORM', '转锋',
            '转变自身的作用方向为对角斜向',
            105, (3,),
            ()),
    S8Affix(4004, 'TRANSFORM', '转锋',
            '重新获得1个随机调色颜料',
            102, (0,),
            ()),
    S8Affix(4005, 'TRANSFORM', '转锋',
            '随机转变自身的等级，内容不变',
            106, ('1:700;2:1800;3:2000;4:3000;5:2000;6:400;7:100',),
            ((1, 700), (2, 1800), (3, 2000), (4, 3000), (5, 2000), (6, 400), (7, 100))),
    S8Affix(4006, 'STACK', '堆砌',
            '使自身等级提升1级',
            None, (),
            ()),
    S8Affix(4007, 'STACK', '堆砌',
            '对作用范围内的随机1个颜料额外提升1级',
            203, (),
            ()),
)


S8_PLOT_ITEMS: tuple[S8PlotItem, ...] = (
    S8PlotItem(100101, '怪物颜料-鼠人', '怪物颜料', 1, 'MONSTER', '雪原中出现鼠人怪物'),
    S8PlotItem(100102, '怪物颜料-鼠人', '怪物颜料', 2, 'MONSTER', '雪原中出现鼠人怪物'),
    S8PlotItem(100103, '怪物颜料-鼠人', '怪物颜料', 3, 'MONSTER', '雪原中出现鼠人怪物；\n出现稀有怪物'),
    S8PlotItem(100104, '怪物颜料-鼠人', '怪物颜料', 4, 'MONSTER', '雪原中出现鼠人怪物；\n出现稀有怪物'),
    S8PlotItem(100105, '怪物颜料-鼠人', '怪物颜料', 5, 'MONSTER', '雪原中出现鼠人怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100106, '怪物颜料-鼠人', '怪物颜料', 6, 'MONSTER', '雪原中出现鼠人怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100107, '怪物颜料-鼠人', '怪物颜料', 7, 'MONSTER', '雪原中出现鼠人怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100201, '怪物颜料-矮人', '怪物颜料', 1, 'MONSTER', '雪原中出现矮人怪物'),
    S8PlotItem(100202, '怪物颜料-矮人', '怪物颜料', 2, 'MONSTER', '雪原中出现矮人怪物'),
    S8PlotItem(100203, '怪物颜料-矮人', '怪物颜料', 3, 'MONSTER', '雪原中出现矮人怪物；\n出现稀有怪物'),
    S8PlotItem(100204, '怪物颜料-矮人', '怪物颜料', 4, 'MONSTER', '雪原中出现矮人怪物；\n出现稀有怪物'),
    S8PlotItem(100205, '怪物颜料-矮人', '怪物颜料', 5, 'MONSTER', '雪原中出现矮人怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100206, '怪物颜料-矮人', '怪物颜料', 6, 'MONSTER', '雪原中出现矮人怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100207, '怪物颜料-矮人', '怪物颜料', 7, 'MONSTER', '雪原中出现矮人怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100301, '怪物颜料-觉醒者', '怪物颜料', 1, 'MONSTER', '雪原中出现觉醒者怪物'),
    S8PlotItem(100302, '怪物颜料-觉醒者', '怪物颜料', 2, 'MONSTER', '雪原中出现觉醒者怪物'),
    S8PlotItem(100303, '怪物颜料-觉醒者', '怪物颜料', 3, 'MONSTER', '雪原中出现觉醒者怪物；\n出现稀有怪物'),
    S8PlotItem(100304, '怪物颜料-觉醒者', '怪物颜料', 4, 'MONSTER', '雪原中出现觉醒者怪物；\n出现稀有怪物'),
    S8PlotItem(100305, '怪物颜料-觉醒者', '怪物颜料', 5, 'MONSTER', '雪原中出现觉醒者怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100306, '怪物颜料-觉醒者', '怪物颜料', 6, 'MONSTER', '雪原中出现觉醒者怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100307, '怪物颜料-觉醒者', '怪物颜料', 7, 'MONSTER', '雪原中出现觉醒者怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100501, '怪物颜料-蛊虫', '怪物颜料', 1, 'MONSTER', '雪原中出现蛊虫怪物'),
    S8PlotItem(100502, '怪物颜料-蛊虫', '怪物颜料', 2, 'MONSTER', '雪原中出现蛊虫怪物'),
    S8PlotItem(100503, '怪物颜料-蛊虫', '怪物颜料', 3, 'MONSTER', '雪原中出现蛊虫怪物；\n出现稀有怪物'),
    S8PlotItem(100504, '怪物颜料-蛊虫', '怪物颜料', 4, 'MONSTER', '雪原中出现蛊虫怪物；\n出现稀有怪物'),
    S8PlotItem(100505, '怪物颜料-蛊虫', '怪物颜料', 5, 'MONSTER', '雪原中出现蛊虫怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100506, '怪物颜料-蛊虫', '怪物颜料', 6, 'MONSTER', '雪原中出现蛊虫怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100507, '怪物颜料-蛊虫', '怪物颜料', 7, 'MONSTER', '雪原中出现蛊虫怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100601, '怪物颜料-龙骨', '怪物颜料', 1, 'MONSTER', '雪原中出现龙骨怪物'),
    S8PlotItem(100602, '怪物颜料-龙骨', '怪物颜料', 2, 'MONSTER', '雪原中出现龙骨怪物'),
    S8PlotItem(100603, '怪物颜料-龙骨', '怪物颜料', 3, 'MONSTER', '雪原中出现龙骨怪物；\n出现稀有怪物'),
    S8PlotItem(100604, '怪物颜料-龙骨', '怪物颜料', 4, 'MONSTER', '雪原中出现龙骨怪物；\n出现稀有怪物'),
    S8PlotItem(100605, '怪物颜料-龙骨', '怪物颜料', 5, 'MONSTER', '雪原中出现龙骨怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100606, '怪物颜料-龙骨', '怪物颜料', 6, 'MONSTER', '雪原中出现龙骨怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100607, '怪物颜料-龙骨', '怪物颜料', 7, 'MONSTER', '雪原中出现龙骨怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100701, '怪物颜料-灾烬', '怪物颜料', 1, 'MONSTER', '雪原中出现灾烬怪物'),
    S8PlotItem(100702, '怪物颜料-灾烬', '怪物颜料', 2, 'MONSTER', '雪原中出现灾烬怪物'),
    S8PlotItem(100703, '怪物颜料-灾烬', '怪物颜料', 3, 'MONSTER', '雪原中出现灾烬怪物；\n出现稀有怪物'),
    S8PlotItem(100704, '怪物颜料-灾烬', '怪物颜料', 4, 'MONSTER', '雪原中出现灾烬怪物；\n出现稀有怪物'),
    S8PlotItem(100705, '怪物颜料-灾烬', '怪物颜料', 5, 'MONSTER', '雪原中出现灾烬怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100706, '怪物颜料-灾烬', '怪物颜料', 6, 'MONSTER', '雪原中出现灾烬怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100707, '怪物颜料-灾烬', '怪物颜料', 7, 'MONSTER', '雪原中出现灾烬怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100801, '怪物颜料-实验体', '怪物颜料', 1, 'MONSTER', '雪原中出现实验体怪物'),
    S8PlotItem(100802, '怪物颜料-实验体', '怪物颜料', 2, 'MONSTER', '雪原中出现实验体怪物'),
    S8PlotItem(100803, '怪物颜料-实验体', '怪物颜料', 3, 'MONSTER', '雪原中出现实验体怪物；\n出现稀有怪物'),
    S8PlotItem(100804, '怪物颜料-实验体', '怪物颜料', 4, 'MONSTER', '雪原中出现实验体怪物；\n出现稀有怪物'),
    S8PlotItem(100805, '怪物颜料-实验体', '怪物颜料', 5, 'MONSTER', '雪原中出现实验体怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100806, '怪物颜料-实验体', '怪物颜料', 6, 'MONSTER', '雪原中出现实验体怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100807, '怪物颜料-实验体', '怪物颜料', 7, 'MONSTER', '雪原中出现实验体怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(100901, '怪物颜料-黑帮', '怪物颜料', 1, 'MONSTER', '雪原中出现黑帮怪物'),
    S8PlotItem(100902, '怪物颜料-黑帮', '怪物颜料', 2, 'MONSTER', '雪原中出现黑帮怪物'),
    S8PlotItem(100903, '怪物颜料-黑帮', '怪物颜料', 3, 'MONSTER', '雪原中出现黑帮怪物；\n出现稀有怪物'),
    S8PlotItem(100904, '怪物颜料-黑帮', '怪物颜料', 4, 'MONSTER', '雪原中出现黑帮怪物；\n出现稀有怪物'),
    S8PlotItem(100905, '怪物颜料-黑帮', '怪物颜料', 5, 'MONSTER', '雪原中出现黑帮怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100906, '怪物颜料-黑帮', '怪物颜料', 6, 'MONSTER', '雪原中出现黑帮怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(100907, '怪物颜料-黑帮', '怪物颜料', 7, 'MONSTER', '雪原中出现黑帮怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(101001, '怪物颜料-伊吉', '怪物颜料', 1, 'MONSTER', '雪原中出现伊吉怪物'),
    S8PlotItem(101002, '怪物颜料-伊吉', '怪物颜料', 2, 'MONSTER', '雪原中出现伊吉怪物'),
    S8PlotItem(101003, '怪物颜料-伊吉', '怪物颜料', 3, 'MONSTER', '雪原中出现伊吉怪物；\n出现稀有怪物'),
    S8PlotItem(101004, '怪物颜料-伊吉', '怪物颜料', 4, 'MONSTER', '雪原中出现伊吉怪物；\n出现稀有怪物'),
    S8PlotItem(101005, '怪物颜料-伊吉', '怪物颜料', 5, 'MONSTER', '雪原中出现伊吉怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101006, '怪物颜料-伊吉', '怪物颜料', 6, 'MONSTER', '雪原中出现伊吉怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101007, '怪物颜料-伊吉', '怪物颜料', 7, 'MONSTER', '雪原中出现伊吉怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(101101, '怪物颜料-沙怪', '怪物颜料', 1, 'MONSTER', '雪原中出现沙怪怪物'),
    S8PlotItem(101102, '怪物颜料-沙怪', '怪物颜料', 2, 'MONSTER', '雪原中出现沙怪怪物'),
    S8PlotItem(101103, '怪物颜料-沙怪', '怪物颜料', 3, 'MONSTER', '雪原中出现沙怪怪物；\n出现稀有怪物'),
    S8PlotItem(101104, '怪物颜料-沙怪', '怪物颜料', 4, 'MONSTER', '雪原中出现沙怪怪物；\n出现稀有怪物'),
    S8PlotItem(101105, '怪物颜料-沙怪', '怪物颜料', 5, 'MONSTER', '雪原中出现沙怪怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101106, '怪物颜料-沙怪', '怪物颜料', 6, 'MONSTER', '雪原中出现沙怪怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101107, '怪物颜料-沙怪', '怪物颜料', 7, 'MONSTER', '雪原中出现沙怪怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(101201, '怪物颜料-邪影', '怪物颜料', 1, 'MONSTER', '雪原中出现邪影怪物'),
    S8PlotItem(101202, '怪物颜料-邪影', '怪物颜料', 2, 'MONSTER', '雪原中出现邪影怪物'),
    S8PlotItem(101203, '怪物颜料-邪影', '怪物颜料', 3, 'MONSTER', '雪原中出现邪影怪物；\n出现稀有怪物'),
    S8PlotItem(101204, '怪物颜料-邪影', '怪物颜料', 4, 'MONSTER', '雪原中出现邪影怪物；\n出现稀有怪物'),
    S8PlotItem(101205, '怪物颜料-邪影', '怪物颜料', 5, 'MONSTER', '雪原中出现邪影怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101206, '怪物颜料-邪影', '怪物颜料', 6, 'MONSTER', '雪原中出现邪影怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101207, '怪物颜料-邪影', '怪物颜料', 7, 'MONSTER', '雪原中出现邪影怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(101301, '怪物颜料-英灵', '怪物颜料', 1, 'MONSTER', '雪原中出现英灵怪物'),
    S8PlotItem(101302, '怪物颜料-英灵', '怪物颜料', 2, 'MONSTER', '雪原中出现英灵怪物'),
    S8PlotItem(101303, '怪物颜料-英灵', '怪物颜料', 3, 'MONSTER', '雪原中出现英灵怪物；\n出现稀有怪物'),
    S8PlotItem(101304, '怪物颜料-英灵', '怪物颜料', 4, 'MONSTER', '雪原中出现英灵怪物；\n出现稀有怪物'),
    S8PlotItem(101305, '怪物颜料-英灵', '怪物颜料', 5, 'MONSTER', '雪原中出现英灵怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101306, '怪物颜料-英灵', '怪物颜料', 6, 'MONSTER', '雪原中出现英灵怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101307, '怪物颜料-英灵', '怪物颜料', 7, 'MONSTER', '雪原中出现英灵怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(101401, '怪物颜料-幽灵', '怪物颜料', 1, 'MONSTER', '雪原中出现幽灵怪物'),
    S8PlotItem(101402, '怪物颜料-幽灵', '怪物颜料', 2, 'MONSTER', '雪原中出现幽灵怪物'),
    S8PlotItem(101403, '怪物颜料-幽灵', '怪物颜料', 3, 'MONSTER', '雪原中出现幽灵怪物；\n出现稀有怪物'),
    S8PlotItem(101404, '怪物颜料-幽灵', '怪物颜料', 4, 'MONSTER', '雪原中出现幽灵怪物；\n出现稀有怪物'),
    S8PlotItem(101405, '怪物颜料-幽灵', '怪物颜料', 5, 'MONSTER', '雪原中出现幽灵怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101406, '怪物颜料-幽灵', '怪物颜料', 6, 'MONSTER', '雪原中出现幽灵怪物；\n出现稀有怪物和传奇首领'),
    S8PlotItem(101407, '怪物颜料-幽灵', '怪物颜料', 7, 'MONSTER', '雪原中出现幽灵怪物；\n出现稀有怪物和传奇首领；\n怪物拥有专属机制'),
    S8PlotItem(200101, '玩法颜料-宝箱', '玩法颜料', 1, 'GAMEPLAY', '雪原中出现宝箱玩法'),
    S8PlotItem(200102, '玩法颜料-宝箱', '玩法颜料', 2, 'GAMEPLAY', '雪原中出现宝箱玩法'),
    S8PlotItem(200103, '玩法颜料-宝箱', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现宝箱玩法'),
    S8PlotItem(200104, '玩法颜料-宝箱', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现宝箱玩法'),
    S8PlotItem(200105, '玩法颜料-宝箱', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现宝箱玩法'),
    S8PlotItem(200106, '玩法颜料-宝箱', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现宝箱玩法'),
    S8PlotItem(200107, '玩法颜料-宝箱', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现宝箱玩法；\n玩法拥有专属机制'),
    S8PlotItem(200301, '玩法颜料-存活', '玩法颜料', 1, 'GAMEPLAY', '雪原中出现存活玩法'),
    S8PlotItem(200302, '玩法颜料-存活', '玩法颜料', 2, 'GAMEPLAY', '雪原中出现存活玩法'),
    S8PlotItem(200303, '玩法颜料-存活', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现存活玩法'),
    S8PlotItem(200304, '玩法颜料-存活', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现存活玩法'),
    S8PlotItem(200305, '玩法颜料-存活', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现存活玩法'),
    S8PlotItem(200306, '玩法颜料-存活', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现存活玩法'),
    S8PlotItem(200307, '玩法颜料-存活', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现存活玩法；\n玩法拥有专属机制'),
    S8PlotItem(200401, '玩法颜料-机械之神', '玩法颜料', 1, 'GAMEPLAY', '雪原中出现机械之神玩法'),
    S8PlotItem(200402, '玩法颜料-机械之神', '玩法颜料', 2, 'GAMEPLAY', '雪原中出现机械之神玩法'),
    S8PlotItem(200403, '玩法颜料-机械之神', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现机械之神玩法'),
    S8PlotItem(200404, '玩法颜料-机械之神', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现机械之神玩法'),
    S8PlotItem(200405, '玩法颜料-机械之神', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现机械之神玩法'),
    S8PlotItem(200406, '玩法颜料-机械之神', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现机械之神玩法'),
    S8PlotItem(200407, '玩法颜料-机械之神', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现机械之神玩法；\n玩法拥有专属机制'),
    S8PlotItem(200501, '玩法颜料-巨力之神', '玩法颜料', 1, 'GAMEPLAY', '雪原中出现巨力之神玩法'),
    S8PlotItem(200502, '玩法颜料-巨力之神', '玩法颜料', 2, 'GAMEPLAY', '雪原中出现巨力之神玩法'),
    S8PlotItem(200503, '玩法颜料-巨力之神', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现巨力之神玩法'),
    S8PlotItem(200504, '玩法颜料-巨力之神', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现巨力之神玩法'),
    S8PlotItem(200505, '玩法颜料-巨力之神', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现巨力之神玩法'),
    S8PlotItem(200506, '玩法颜料-巨力之神', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现巨力之神玩法'),
    S8PlotItem(200507, '玩法颜料-巨力之神', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现巨力之神玩法；\n玩法拥有专属机制'),
    S8PlotItem(200601, '玩法颜料-狩猎之神', '玩法颜料', 1, 'GAMEPLAY', '雪原中出现狩猎之神玩法'),
    S8PlotItem(200602, '玩法颜料-狩猎之神', '玩法颜料', 2, 'GAMEPLAY', '雪原中出现狩猎之神玩法'),
    S8PlotItem(200603, '玩法颜料-狩猎之神', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现狩猎之神玩法'),
    S8PlotItem(200604, '玩法颜料-狩猎之神', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现狩猎之神玩法'),
    S8PlotItem(200605, '玩法颜料-狩猎之神', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现狩猎之神玩法'),
    S8PlotItem(200606, '玩法颜料-狩猎之神', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现狩猎之神玩法'),
    S8PlotItem(200607, '玩法颜料-狩猎之神', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现狩猎之神玩法；\n玩法拥有专属机制'),
    S8PlotItem(200701, '玩法颜料-征战之神', '玩法颜料', 1, 'GAMEPLAY', '雪原中出现征战之神玩法'),
    S8PlotItem(200702, '玩法颜料-征战之神', '玩法颜料', 2, 'GAMEPLAY', '雪原中出现征战之神玩法'),
    S8PlotItem(200703, '玩法颜料-征战之神', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现征战之神玩法'),
    S8PlotItem(200704, '玩法颜料-征战之神', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现征战之神玩法'),
    S8PlotItem(200705, '玩法颜料-征战之神', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现征战之神玩法'),
    S8PlotItem(200706, '玩法颜料-征战之神', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现征战之神玩法'),
    S8PlotItem(200707, '玩法颜料-征战之神', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现征战之神玩法；\n玩法拥有专属机制'),
    S8PlotItem(200803, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(200804, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(200805, '玩法颜料-黑潮', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现黑潮玩法'),
    S8PlotItem(200806, '玩法颜料-黑潮', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现黑潮玩法'),
    S8PlotItem(200807, '玩法颜料-黑潮', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现黑潮玩法；\n玩法拥有专属机制'),
    S8PlotItem(200903, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(200904, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(200905, '玩法颜料-黑帆', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现黑帆玩法'),
    S8PlotItem(200906, '玩法颜料-黑帆', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现黑帆玩法'),
    S8PlotItem(200907, '玩法颜料-黑帆', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现黑帆玩法；\n玩法拥有专属机制'),
    S8PlotItem(201003, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201004, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201005, '玩法颜料-魔方', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现魔方玩法'),
    S8PlotItem(201006, '玩法颜料-魔方', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现魔方玩法'),
    S8PlotItem(201007, '玩法颜料-魔方', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现魔方玩法；\n玩法拥有专属机制'),
    S8PlotItem(201103, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201104, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201105, '玩法颜料-迷城', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现迷城玩法'),
    S8PlotItem(201106, '玩法颜料-迷城', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现迷城玩法'),
    S8PlotItem(201107, '玩法颜料-迷城', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现迷城玩法；\n玩法拥有专属机制'),
    S8PlotItem(201203, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201204, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201205, '玩法颜料-梦魇', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现梦魇玩法'),
    S8PlotItem(201206, '玩法颜料-梦魇', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现梦魇玩法'),
    S8PlotItem(201207, '玩法颜料-梦魇', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现梦魇玩法；\n玩法拥有专属机制'),
    S8PlotItem(201303, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201304, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201305, '玩法颜料-怪谈', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现怪谈玩法'),
    S8PlotItem(201306, '玩法颜料-怪谈', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现怪谈玩法'),
    S8PlotItem(201307, '玩法颜料-怪谈', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现怪谈玩法；\n玩法拥有专属机制'),
    S8PlotItem(201403, '玩法颜料-赛季', '玩法颜料', 3, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201404, '玩法颜料-赛季', '玩法颜料', 4, 'GAMEPLAY', '雪原中出现随机赛季怪物；\n该颜料升至5级及以上时，转变为随机赛季玩法'),
    S8PlotItem(201405, '玩法颜料-玩偶', '玩法颜料', 5, 'GAMEPLAY', '雪原中出现玩偶玩法'),
    S8PlotItem(201406, '玩法颜料-玩偶', '玩法颜料', 6, 'GAMEPLAY', '雪原中出现玩偶玩法'),
    S8PlotItem(201407, '玩法颜料-玩偶', '玩法颜料', 7, 'GAMEPLAY', '雪原中出现玩偶玩法；\n玩法拥有专属机制'),
)


S8_DIFFICULTIES: tuple[tuple[int, str, str, str], ...] = (
    # (难度ID, 画纸名, 说明, 解锁条件)
    (3, '苍蓝雪纸', '战斗难度根据等级动态', '通关无色雪纸'),
    (4, '幽紫雪纸', '战斗难度等同于异界时刻7\n掉落数量额外+10%\n专属辅助技能词缀最高为T2', '达到80级且通关无色雪纸\n掉落数量额外+10%\n专属辅助技能词缀最高为T2'),
    (5, '暖橙雪纸', '战斗难度等同于异界时刻8\n掉落数量额外+60%\n专属辅助技能词缀最高为T1', '达到80级且通关幽紫雪纸\n掉落数量额外+60%\n专属辅助技能词缀最高为T1'),
    (6, '赤红雪纸', '战斗难度等同于异界幽邃时刻8\n掉落数量额外+100%\n专属辅助技能词缀最高为T0', '达到80级且通关暖橙雪纸\n掉落数量额外+100%\n专属辅助技能词缀最高为T0'),
    (7, '绮彩雪纸', '战斗难度等同于异界深空\n掉落数量额外+200%\n专属辅助技能词缀最高为T0', '达到80级且通关赤红雪纸\n掉落数量额外+200%\n专属辅助技能词缀最高为T0'),
    (101, '无色雪纸', '画中雪原玩法的教学难度', ''),
)

S8_REVIVES: tuple[tuple[int, int, str], ...] = (
    # (ID, 摧毁颜料数, 官方文案)
    (1001, 1, '随机将地图上的【1】个颜料连同该地块一起摧毁，变为失色地块'),
    (2001, 2, '随机将地图上的【2】个颜料连同该地块一起摧毁，变为失色地块'),
    (3001, 3, '随机将地图上的【3】个颜料连同该地块一起摧毁，变为失色地块'),
)


# 官方玩法界面文案（键 -> 原文），棋盘/地块/方向/阶段的权威定义。
S8_UI_TEXT: dict[str, str] = {
    'Func_S8Gameplay_UI_BlockDes2001': '可以放入任意的调色颜料或颜料',
    'Func_S8Gameplay_UI_BlockDes2002': '无法放入任何颜料。调色颜料的笔触效果可以使虚空地块转变为空白地块，或使自身可以放置在虚空地块上。',
    'Func_S8Gameplay_UI_BlockDes2003': '已经被摧毁的地块。调色颜料的笔触效果可以使失色地块转变为空白地块，或使自身可以放置在失色地块上。',
    'Func_S8Gameplay_UI_BlockDes3001': '当前无法放入颜料。调色颜料的笔触效果可以使阻挡地块转变为空白地块，或使自身可以放置在阻挡地块上。',
    'Func_S8Gameplay_UI_BlockDes3002': '无法放入任何颜料。调色颜料的笔触效果可以使虚空地块转变为空白地块，或使自身可以放置在虚空地块上。',
    'Func_S8Gameplay_UI_BlockTitle2001': '空白地块',
    'Func_S8Gameplay_UI_BlockTitle2002': '虚空地块',
    'Func_S8Gameplay_UI_BlockTitle2003': '失色地块',
    'Func_S8Gameplay_UI_BlockTitle3001': '阻挡地块',
    'Func_S8Gameplay_UI_BlockTitle3002': '虚空地块',
    'Func_S8Gameplay_UI_Button1': '展开',
    'Func_S8Gameplay_UI_Button10': '跳过',
    'Func_S8Gameplay_UI_Button2': '开始绘制',
    'Func_S8Gameplay_UI_Button3': '强制调色',
    'Func_S8Gameplay_UI_Button4': '重新绘制',
    'Func_S8Gameplay_UI_Button5': '选择',
    'Func_S8Gameplay_UI_Button6': '落笔',
    'Func_S8Gameplay_UI_Button7': '完成绘制',
    'Func_S8Gameplay_UI_Button8': '入画',
    'Func_S8Gameplay_UI_Button9': '领取',
    'Func_S8Gameplay_UI_Collect': '画室',
    'Func_S8Gameplay_UI_Collect_Button': '#!领取奖励',
    'Func_S8Gameplay_UI_Collect_MapDes': '#!这里记录了你所绘制的最完美的画卷。\\n无数绮丽的色彩如花绽放，就此定格。',
    'Func_S8Gameplay_UI_Collect_MapTabItem': '#!道具收藏',
    'Func_S8Gameplay_UI_Collect_MapTabMap': '#!地图收藏',
    'Func_S8Gameplay_UI_Collect_MapTitle': '#!定格的画卷',
    'Func_S8Gameplay_UI_Collect_NoItem': '#!还未获得该地块道具',
    'Func_S8Gameplay_UI_Collect_NoMap': '#!还未获得任何可收藏的地图',
    'Func_S8Gameplay_UI_ConfText1': '放入颜料的位置不在调色范围内，是否继续？',
    'Func_S8Gameplay_UI_ConfText2': '放入颜料的类型不符合调色目标，是否继续？',
    'Func_S8Gameplay_UI_ConfText3': '重绘会回到集雪阶段，所有已经消耗的雪纸碎片、收集的颜料、完成的地块都会被销毁，是否继续？',
    'Func_S8Gameplay_UI_ConfText4': '重新绘制会拆下所有已放入地块的颜料，需要重新作画，是否继续？',
    'Func_S8Gameplay_UI_ConfText5': '完成绘制后无法回退，是否继续？',
    'Func_S8Gameplay_UI_ConfText6': '使用强制调色，是否继续？',
    'Func_S8Gameplay_UI_ConfText7': '使用强制调色，自动完成当前调色目标\\n%s\\n是否继续？',
    'Func_S8Gameplay_UI_CurrentEffect': '使%s的%s颜料提升%s级',
    'Func_S8Gameplay_UI_CurrentTask': '在%s地块上放置%s颜料',
    'Func_S8Gameplay_UI_Direction1': '横向',
    'Func_S8Gameplay_UI_Direction2': '纵向',
    'Func_S8Gameplay_UI_Direction3': '对角斜向',
    'Func_S8Gameplay_UI_Direction4': '相邻',
    'Func_S8Gameplay_UI_Direction5': '全图',
    'Func_S8Gameplay_UI_Direction6': '自身',
    'Func_S8Gameplay_UI_DropGet': '获得',
    'Func_S8Gameplay_UI_ItemType1': '怪物',
    'Func_S8Gameplay_UI_ItemType2': '玩法',
    'Func_S8Gameplay_UI_ItemType3': '所有',
    'Func_S8Gameplay_UI_LeaveMap': '队长已离开地图',
    'Func_S8Gameplay_UI_Map': '地图',
    'Func_S8Gameplay_UI_MapComplete': '已完成',
    'Func_S8Gameplay_UI_MapLevel': '地块等级',
    'Func_S8Gameplay_UI_Retest': '重绘',
    'Func_S8Gameplay_UI_Stage1': '集雪',
    'Func_S8Gameplay_UI_Stage2': '拾彩',
    'Func_S8Gameplay_UI_Stage3': '作画',
    'Func_S8Gameplay_UI_Stage4': '入画',
    'Func_S8Gameplay_UI_Talent': '技艺',
    'Func_S8Gameplay_UI_TalentButton': '一键清空',
    'Func_S8Gameplay_UI_TalentButton2': '加点',
    'Func_S8Gameplay_UI_TalentText1': '所有修改在下一次绘制地图时才会生效',
    'Func_S8Gameplay_UI_TalentText2': '还需要加%s点前置天赋',
    'Func_S8Gameplay_UI_TalentText3': '需要先加同行的大天赋点',
    'Func_S8Gameplay_UI_TalentTextChoice': '无法选择更多的特殊天赋点',
    'Func_S8Gameplay_UI_TalentTextConfirm': '是否一键清空所有天赋点？',
    'Func_S8Gameplay_UI_TalentTextVertical': '需要先加上一行的大天赋点',
    'Func_S8Gameplay_UI_TalentTitle1': '当前生效：',
    'Func_S8Gameplay_UI_TalentTitle2': '下级生效：',
    'Func_S8Gameplay_UI_TaskOver': '调色成功',
    'Func_S8Gameplay_UI_TeamMapView': '正在预览队长绘制的地图',
    'Func_S8Gameplay_UI_Text1': '当前画纸底色',
    'Func_S8Gameplay_UI_Text10': '进度奖励',
    'Func_S8Gameplay_UI_Text11': '进度',
    'Func_S8Gameplay_UI_Text12': '地块玩法总进度：',
    'Func_S8Gameplay_UI_Text13': '结算奖励需求进度：',
    'Func_S8Gameplay_UI_Text14': '点击已完成地块可进行传送',
    'Func_S8Gameplay_UI_Text15': '快速绘制',
    'Func_S8Gameplay_UI_Text16': '颜料已全部放置，可完成绘制',
    'Func_S8Gameplay_UI_Text17': '离开地图',
    'Func_S8Gameplay_UI_Text2': '选择本次作画中，新调色颜料固定出现的笔触',
    'Func_S8Gameplay_UI_Text3': '当前颜料最大等级',
    'Func_S8Gameplay_UI_Text4': '当前调色目标',
    'Func_S8Gameplay_UI_Text5': '选择一个空白地块放入',
    'Func_S8Gameplay_UI_Text6': '选择笔触方向',
    'Func_S8Gameplay_UI_Text7': '选择笔触效果',
    'Func_S8Gameplay_UI_Text8': '可入画次数',
    'Func_S8Gameplay_UI_Text9': '完成地块进度',
    'Func_S8Gameplay_UI_Tips1': '等级',
    'Func_S8Gameplay_UI_Tips2': '等级加成',
    'Func_S8Gameplay_UI_Tips3': '调色目标',
    'Func_S8Gameplay_UI_Tips4': '调色效果',
    'Func_S8Gameplay_UI_Tips5': '颜料内容',
    'Func_S8Gameplay_UI_Tips6': '笔触效果',
    'Func_S8Gameplay_UI_Title1': '技艺效果',
    'Func_S8Gameplay_UI_Title2': '画中雪原',
}


# 哪个官方记录带权重表（档位 -> 权重）
S8_AFFIX_WEIGHT_IDS: tuple[int, ...] = tuple(a.id for a in S8_SELECTABLE_EFFECTS if a.weights)


def affix_option_distribution(affix_id: int) -> dict[int, float]:
    """把官方权重表换算成概率分布：``{档位: 概率}``，概率之和为 1。

    档位含义随效果而异（等级 / 作用范围 / 作用对象），由对应条目的官方文案决定。
    例：``affix_option_distribution(4005)`` 返回 1 级 7%、2 级 18 % ... 7 级 1 %。
    """
    for item in S8_SELECTABLE_EFFECTS:
        if item.id != affix_id:
            continue
        total = sum(weight for _option, weight in item.weights)
        if total <= 0:
            return {}
        return {option: weight / total for option, weight in item.weights}
    return {}


def affix_option_expectation(affix_id: int) -> float:
    """权重表档位的期望值（档位按数值解读时才是等级期望）。"""
    distribution = affix_option_distribution(affix_id)
    return sum(option * probability for option, probability in distribution.items())


# ---------------------------------------------------------------------------
# 颜料等级出现几率（官方权重表，万分比）
# ---------------------------------------------------------------------------
#
# 官方记录 ``4005``（转锋·随机转变自身的等级，内容不变）内嵌权重串
#   ``1:700;2:1800;3:2000;4:3000;5:2000;6:400;7:100``
# 合计正好 10000，即**万分比**。这是客户端里唯一一处给出
# 「1..7 级 → 权重」的官方数据，因此被当作颜料等级的出现分布。
#
# ⚠️ 边界：该权重串的官方归属是「转锋随机改等级」，不是「拾彩阶段掉落等级」。
#    后者由服务器下发（客户端无表）。在拿到服务器数据前，本模块把它作为
#    **现有最接近的官方口径**使用，并在 :data:`PAINT_LEVEL_WEIGHT_SOURCE` 里
#    显式标注来源，调用方可以据此决定是否采用。
PAINT_LEVEL_WEIGHT_SOURCE = 'S8_Gameplay_SelectableEffect 4005 内嵌权重串（万分比）'

#: ``{等级: 权重}``，官方原始整数，合计 10000。
PAINT_LEVEL_WEIGHTS: dict[int, int] = dict(
    affix_option_distribution(4005) and {
        level: weight
        for level, weight in next(
            a.weights for a in S8_SELECTABLE_EFFECTS if a.id == 4005
        )
    }
)


def paint_level_distribution() -> dict[int, float]:
    """颜料等级出现概率 ``{1..7: 概率}``，按官方权重归一化。"""
    total = sum(PAINT_LEVEL_WEIGHTS.values())
    if total <= 0:
        return {}
    return {level: weight / total for level, weight in PAINT_LEVEL_WEIGHTS.items()}


def paint_level_expectation() -> float:
    """按官方权重算出的期望等级（约 3.54）。"""
    return sum(level * p for level, p in paint_level_distribution().items())


# ---------------------------------------------------------------------------
# 颜料家族目录（按官方 PlotItem 表统计）
# ---------------------------------------------------------------------------

def _subtype_of(name: str) -> str:
    return name.split('-', 1)[1] if '-' in name else name


def _family_catalog() -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {}
    for item in S8_PLOT_ITEMS:
        subtype = _subtype_of(item.name)
        entry = catalog.setdefault(subtype, {'category': item.category, 'levels': []})
        levels = entry['levels']
        assert isinstance(levels, list)
        if item.level not in levels:
            levels.append(item.level)
    for entry in catalog.values():
        levels = entry['levels']
        assert isinstance(levels, list)
        levels.sort()
    return catalog


#: ``{家族名: {'category': 'MONSTER'/'GAMEPLAY', 'levels': [1..7]}}``
S8_PAINT_FAMILIES: dict[str, dict[str, object]] = _family_catalog()

#: 有完整 1..7 级的家族（可任意等级出现）。
S8_FULL_LEVEL_FAMILIES: tuple[str, ...] = tuple(
    name for name, meta in S8_PAINT_FAMILIES.items()
    if meta['levels'] == list(range(1, 8))
)

#: 只出现在部分等级段的家族 —— 它们的等级不是自由变量。
S8_PARTIAL_LEVEL_FAMILIES: dict[str, list[int]] = {
    name: list(meta['levels'])
    for name, meta in S8_PAINT_FAMILIES.items()
    if meta['levels'] != list(range(1, 8))
}
