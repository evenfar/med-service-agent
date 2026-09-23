"""会话持久化（第2期）：JSON 原子写 + 损坏文件降级。

工程细节：先写 .tmp 再 os.replace —— 崩溃时不会留下半截会话文件；
load 对损坏/版本不符的文件返回 None 视为新会话（用户体验优先于报错）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

SESSION_VERSION = 1


def save_session(path: str, messages: list[dict], summary: Optional[str],
                 short_term_memory: Optional[dict] = None) -> None:
    payload = {
        "version": SESSION_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
        "short_term_memory": short_term_memory,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)  # 原子替换


def load_session(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != SESSION_VERSION or not isinstance(data.get("messages"), list):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def delete_session(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
