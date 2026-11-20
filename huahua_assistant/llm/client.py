"""OpenAI 兼容的视觉对话客户端。

不引第三方库，直接用 ``urllib``：云端（智谱 / 通义 / DeepSeek / OpenAI）和
本地 Ollama 都吃同一套 ``/chat/completions``，只是 ``base_url`` 和 ``model`` 不同。
"""

from __future__ import annotations

import base64
import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from .types import AdvisorConfig


class LlmError(RuntimeError):
    """调用失败。上层据此降级回本地视觉链路。"""


class RateLimitedError(LlmError):
    """被限流（429）。免费模型在高峰期非常常见，上层要据此进入冷却。"""


def encode_image(image: np.ndarray, max_side: int) -> tuple[str, str]:
    """把 BGR 图缩放到 ``max_side`` 以内并编码成 base64。

    JPEG 比 PNG 小一个数量级，VLM 对压缩不敏感，值得。
    """
    if image is None or image.size == 0:
        raise LlmError("空图")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(1, max(height, width)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise LlmError("图片编码失败")
    return base64.b64encode(buffer.tobytes()).decode("ascii"), "image/jpeg"


def _endpoint(config: AdvisorConfig) -> str:
    base = config.base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _headers(config: AdvisorConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


# 免费模型的「该模型当前访问量过大」就是 429。退避几次再放弃，间隔逐次拉长。
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_BACKOFF = (1.5, 4.0)
_RETRY_SLEEP_CAP = 8.0
# 平台过载时这些错误码也会用 HTTP 200 包在 error 体里返回，一样要重试。
_RATE_LIMIT_CODES = frozenset({1305, "1305"})
_TRANSIENT_CODES = frozenset({500, 502, 503, 504, "500", "502", "503", "504"})


def _retry_after(exc: urllib.error.HTTPError, default: float) -> float:
    raw = ""
    try:
        raw = (exc.headers.get("Retry-After") or "").strip()
    except AttributeError:
        return default
    try:
        return max(0.0, min(_RETRY_SLEEP_CAP, float(raw)))
    except (TypeError, ValueError):
        return default


def _error_payload(decoded: dict) -> tuple[bool, str, Any]:
    """返回 (是否错误, 提示, 错误码)。"""
    error = decoded.get("error")
    if not isinstance(error, dict):
        return False, "", None
    message = str(error.get("message") or "").strip()
    if not message:
        return False, "", None
    return True, message, error.get("code")


def _is_rate_limit(message: str, code: Any) -> bool:
    return code in _RATE_LIMIT_CODES or "访问量" in message or "限流" in message


def _is_transient(message: str, code: Any) -> bool:
    if _is_rate_limit(message, code):
        return True
    if code in _TRANSIENT_CODES:
        return True
    # 「Request method 'GET' not supported」是平台在过载时把 POST 打回来的表现，
    # HTTP 状态码却是 200，只能按文案认。
    return "GET" in message and "not supported" in message


def _post(request: urllib.request.Request, config: AdvisorConfig) -> dict:
    """发请求并解析成 dict。限流和服务端抖动都退避重试，其余错误直接抛。

    免费池（GLM-4.6V-Flash 这类）高峰期几乎必然 429，一次就降级回本地的话，
    托管模式在白天基本等于没开。重试完仍然限流就抛 ``RateLimitedError``，
    由 ``Advisor`` 记一段冷却时间，别把主循环按住。
    """
    attempts = len(_RETRY_BACKOFF) + 1
    last_detail = ""
    for attempt in range(attempts):
        retry = attempt < attempts - 1
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code in _RETRY_STATUS and retry:
                time.sleep(_retry_after(exc, _RETRY_BACKOFF[attempt]))
                continue
            if exc.code == 429:
                raise RateLimitedError(f"HTTP 429：{last_detail}") from exc
            raise LlmError(f"HTTP {exc.code}：{last_detail}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"连不上 {config.base_url}：{exc.reason}") from exc
        except (ConnectionError, http.client.IncompleteRead) as exc:
            # 中转站/网关负载高时会直接掐连接（RemoteDisconnected），或者响应读到一半断掉。
            # 这类抖动重试一次基本就好了，没必要整轮降级回本地。
            last_detail = f"{type(exc).__name__}: {exc}"
            if retry:
                time.sleep(_RETRY_BACKOFF[attempt])
                continue
            raise LlmError(f"连接被对端断开：{last_detail}") from exc
        except TimeoutError as exc:
            raise LlmError(f"超时（{config.timeout:.0f}s）") from exc

        try:
            decoded = json.loads(raw)
        except ValueError as exc:
            raise LlmError(f"返回不是 JSON：{raw[:200]}") from exc

        failed, message, code = _error_payload(decoded)
        if not failed:
            return decoded
        last_detail = message
        if _is_transient(message, code) and retry:
            time.sleep(_RETRY_BACKOFF[attempt])
            continue
        if _is_rate_limit(message, code):
            raise RateLimitedError(f"服务端限流：{message}")
        raise LlmError(f"服务端返回错误：{message}（code={code}）")

    if last_detail:
        raise LlmError(f"重试后仍失败：{last_detail}")
    raise LlmError("重试后仍失败")


def chat_vision(
    config: AdvisorConfig,
    *,
    system_prompt: str,
    user_text: str,
    image_b64: str,
    mime: str = "image/jpeg",
) -> str:
    """发一次视觉对话，返回模型输出的原始文本。"""
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            },
        ],
    }
    if config.extra_body:
        payload.update(config.extra_body)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _endpoint(config),
        data=body,
        headers=_headers(config),
        method="POST",
    )
    decoded = _post(request, config)

    choices = decoded.get("choices") or []
    if not choices:
        raise LlmError(f"返回没有 choices：{json.dumps(decoded, ensure_ascii=False)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # 有些实现把 content 拆成多段
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        # 推理型模型（DeepSeek / o 系）如果没关掉思考，正文可能一个字都没有，
        # 答案全在 reasoning_content 里。这里兜底捞一次，总比直接判失败强。
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        raise LlmError(
            f"返回内容为空（finish_reason={choices[0].get('finish_reason')}）："
            f"{json.dumps(decoded, ensure_ascii=False)[:200]}"
        )
    return content
