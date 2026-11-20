"""大模型托管层的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


# 动作类型。和游戏里能做的四件事一一对应，另外留 unknown 给「看不懂」。
ACTION_PLACE = "place"          # 把当前颜料放到某块地
ACTION_BACKPACK = "backpack"    # 从背包选一枚颜料
ACTION_BRUSH1 = "brush_stage1"  # 选笔触方向
ACTION_BRUSH2 = "brush_stage2"  # 选笔触效果
ACTION_FINISH = "finish"        # 结束绘制
ACTION_UNKNOWN = "unknown"

ACTION_LABELS = {
    ACTION_PLACE: "放置颜料",
    ACTION_BACKPACK: "选背包颜料",
    ACTION_BRUSH1: "选笔触方向",
    ACTION_BRUSH2: "选笔触效果",
    ACTION_FINISH: "结束绘制",
    ACTION_UNKNOWN: "未给出动作",
}


@dataclass(frozen=True)
class AdvisorConfig:
    """一次托管调用的配置。全部来自 ``runtime/strategy_config.json`` 的 ``llm`` 段。"""

    enabled: bool = False
    provider: str = "openai_compat"   # openai_compat | ollama
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""
    model: str = "glm-4.6v-flash"
    timeout: float = 30.0
    # 两次调用之间至少隔这么久。大模型单次 2~5 秒，节流是为了不让它排队排到天上去。
    min_interval: float = 2.0
    # 送进去的图先缩到这个边长，既省 token 也省时间。
    max_image_side: int = 1280
    temperature: float = 0.1
    # 输出上限。推理型模型（DeepSeek 这类）如果思考没关掉，会把额度全烧在
    # reasoning_content 上，content 一个字都不剩 —— 给太小就永远拿到空回复。
    max_tokens: int = 600
    # 原样并进请求体的额外字段。各家私有开关走这里，不用为每家加一个配置项。
    # 例：智谱 GLM-4.6V / DeepSeek 关掉思考 {"thinking": {"type": "disabled"}}
    extra_body: dict = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """配置齐不齐（**不看 enabled 开关**）。

        工具栏那个「大模型托管」勾选框就是 enabled 开关本身，所以它必须能在
        enabled=false 的情况下把配置打开；否则勾上立刻弹回、提示缺 key，永远开不了。
        """
        if self.provider == "ollama":
            return bool(self.base_url and self.model)
        return bool(self.base_url and self.model and self.api_key)

    @property
    def ready(self) -> bool:
        return bool(self.enabled) and self.configured


@dataclass(frozen=True)
class AdvisorAdvice:
    """大模型给出的一条建议。"""

    ok: bool
    action: str = ACTION_UNKNOWN
    row: int | None = None
    col: int | None = None
    option: str | None = None
    reason: str = ""
    raw: str = ""
    elapsed: float = 0.0
    error: str | None = None
    # 这次是复用的上一次结果（节流窗口内），没有真的调模型
    cached: bool = False

    @property
    def action_label(self) -> str:
        return ACTION_LABELS.get(self.action, self.action)

    @property
    def cell(self) -> tuple[int, int] | None:
        if self.row is None or self.col is None:
            return None
        return (int(self.row), int(self.col))

    def summary(self) -> str:
        if not self.ok:
            return f"大模型不可用：{self.error or '未知原因'}"
        parts = [self.action_label]
        if self.cell is not None:
            parts.append(f"({self.row},{self.col})")
        if self.option:
            parts.append(self.option)
        text = " ".join(parts)
        return f"{text} —— {self.reason}" if self.reason else text
