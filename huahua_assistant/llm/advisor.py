"""托管调度：节流、缓存、失败降级。

这一层不管提示词也不管 HTTP，只回答一个问题：**现在该不该再问一次模型，
问完之后这次结果能不能用**。

规则：

1. 配置没开 / 缺 key → 直接 ``ok=False``，上层走本地链路，不报错；
2. 距离上次调用不到 ``min_interval`` → 返回上次结果（``cached=True``）；
3. 调用失败 → ``ok=False`` + 错误原因，**不清空**上次的好结果；
4. 调用成功 → 记下来，下一次节流窗口内复用它。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from ..config import config_value
from .client import LlmError, RateLimitedError, chat_vision, encode_image
from .prompt import SYSTEM_PROMPT, build_user_content, parse_advice
from .types import AdvisorAdvice, AdvisorConfig

# 被限流后静默这么久再试。免费池的 429 是「整池子挤」，隔一两秒重试只会继续撞墙。
RATE_LIMIT_COOLDOWN = 10.0


def load_config() -> AdvisorConfig:
    """从 ``runtime/strategy_config.json`` 的 ``llm`` 段读配置。"""
    extra = config_value("llm.extra_body", {})
    return AdvisorConfig(
        enabled=bool(config_value("llm.enabled", False)),
        provider=str(config_value("llm.provider", "openai_compat") or "openai_compat"),
        base_url=str(config_value("llm.base_url", "") or ""),
        api_key=str(config_value("llm.api_key", "") or ""),
        model=str(config_value("llm.model", "") or ""),
        timeout=float(config_value("llm.timeout", 30.0) or 30.0),
        min_interval=float(config_value("llm.min_interval", 2.0) or 2.0),
        max_image_side=int(config_value("llm.max_image_side", 1280) or 1280),
        temperature=float(config_value("llm.temperature", 0.1) or 0.1),
        max_tokens=int(config_value("llm.max_tokens", 600) or 600),
        extra_body=dict(extra) if isinstance(extra, dict) else {},
    )


def _reuse(last: AdvisorAdvice) -> AdvisorAdvice:
    """把上次的结果原样再给一遍（节流窗口内 / 限流冷却中）。"""
    return AdvisorAdvice(
        ok=last.ok,
        action=last.action,
        row=last.row,
        col=last.col,
        option=last.option,
        reason=last.reason,
        raw=last.raw,
        elapsed=last.elapsed,
        error=last.error,
        cached=True,
    )


class Advisor:
    """把一帧画面交给大模型，换回一条动作建议。"""

    def __init__(self, config: AdvisorConfig | None = None) -> None:
        self._config = config or load_config()
        self._lock = threading.Lock()
        self._last: AdvisorAdvice | None = None
        self._last_at = 0.0
        # 被限流后的冷却截止时间（monotonic）
        self._blocked_until = 0.0

    @property
    def config(self) -> AdvisorConfig:
        return self._config

    def reload(self) -> None:
        with self._lock:
            self._config = load_config()
            self._last = None
            self._last_at = 0.0
            self._blocked_until = 0.0

    def advise(
        self,
        image: np.ndarray,
        context: str = "",
        *,
        force: bool = False,
    ) -> AdvisorAdvice:
        """同步调用。**会阻塞几秒**，不要在 UI 线程直接调。"""
        config = self._config
        if not config.ready:
            return AdvisorAdvice(
                ok=False,
                error="大模型托管未开启或缺少配置（llm.enabled / base_url / model / api_key）",
            )

        now = time.monotonic()
        with self._lock:
            last = self._last
            last_at = self._last_at
            blocked_until = self._blocked_until
        if not force and last is not None and now - last_at < config.min_interval:
            return _reuse(last)
        if not force and now < blocked_until:
            # 刚被限流过：有上次的好结果就继续用它，没有就说清楚在等
            if last is not None and last.ok:
                return _reuse(last)
            return AdvisorAdvice(
                ok=False,
                error=f"限流冷却中（还剩 {blocked_until - now:.0f}s）",
            )

        started_at = time.perf_counter()
        try:
            image_b64, mime = encode_image(image, config.max_image_side)
            text = chat_vision(
                config,
                system_prompt=SYSTEM_PROMPT,
                user_text=build_user_content(context),
                image_b64=image_b64,
                mime=mime,
            )
        except RateLimitedError as exc:
            advice = AdvisorAdvice(ok=False, elapsed=time.perf_counter() - started_at, error=str(exc))
            with self._lock:
                self._last_at = time.monotonic()
                self._blocked_until = time.monotonic() + RATE_LIMIT_COOLDOWN
            return advice
        except LlmError as exc:
            advice = AdvisorAdvice(ok=False, elapsed=time.perf_counter() - started_at, error=str(exc))
            # 失败不覆盖上一次的好结果，只是不在这个窗口里复用
            with self._lock:
                self._last_at = time.monotonic()
            return advice
        except Exception as exc:  # noqa: BLE001 - 任何意外都不该把主循环带崩
            advice = AdvisorAdvice(ok=False, elapsed=time.perf_counter() - started_at, error=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._last_at = time.monotonic()
            return advice

        advice = parse_advice(text, elapsed=time.perf_counter() - started_at)
        with self._lock:
            self._last = advice
            self._last_at = time.monotonic()
            self._blocked_until = 0.0
        return advice

    def last(self) -> AdvisorAdvice | None:
        with self._lock:
            return self._last


_advisor: Advisor | None = None
_advisor_lock = threading.Lock()


def get_advisor() -> Advisor:
    global _advisor
    with _advisor_lock:
        if _advisor is None:
            _advisor = Advisor()
        return _advisor


def reset_advisor() -> None:
    """配置改了之后调用，下次会重新读配置。"""
    global _advisor
    with _advisor_lock:
        _advisor = None
