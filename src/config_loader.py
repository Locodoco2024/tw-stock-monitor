from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


_ALLOWED_ROOT_KEYS = {"user", "institutional_candidates", "stocks"}
_ALLOWED_USER_KEYS = {"id", "enabled", "discord_webhook_key"}
_ALLOWED_STOCK_KEYS = {"symbol", "average_cost", "profit_alerts"}
_ALLOWED_INSTITUTIONAL_KEYS = {
    "enabled",
    "max_new_candidates",
    "include_state_updates",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"設定檔不存在: {target}")
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"設定檔根節點必須是物件: {target}")
    return data


def load_user_configs(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    files = sorted(target.glob("*.yaml")) if target.is_dir() else [target]
    configs: list[dict[str, Any]] = []
    for file in files:
        config = load_yaml(file)
        user = config.get("user", {})
        if not user.get("enabled", True):
            continue
        validate_user_config(config, file)
        config["_source_file"] = str(file)
        configs.append(config)
    if not configs:
        raise ConfigError(f"找不到啟用中的使用者設定: {target}")
    return configs


def validate_user_config(config: dict[str, Any], path: Path | None = None) -> None:
    prefix = f"{path}: " if path else ""
    _reject_unknown(config, _ALLOWED_ROOT_KEYS, prefix, "根節點")

    user = config.get("user")
    if not isinstance(user, dict) or not user.get("id"):
        raise ConfigError(f"{prefix}user.id 必填")
    _reject_unknown(user, _ALLOWED_USER_KEYS, prefix, "user")

    institutional = config.get("institutional_candidates", {})
    if institutional is not None and not isinstance(institutional, dict):
        raise ConfigError(f"{prefix}institutional_candidates 必須是物件")
    institutional = institutional or {}
    _reject_unknown(
        institutional,
        _ALLOWED_INSTITUTIONAL_KEYS,
        prefix,
        "institutional_candidates",
    )
    if "enabled" in institutional and not isinstance(institutional["enabled"], bool):
        raise ConfigError(f"{prefix}institutional_candidates.enabled 必須是 true 或 false")
    max_new = institutional.get("max_new_candidates", 6)
    if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new < 1:
        raise ConfigError(f"{prefix}institutional_candidates.max_new_candidates 必須是正整數")
    if "include_state_updates" in institutional and not isinstance(
        institutional["include_state_updates"], bool
    ):
        raise ConfigError(
            f"{prefix}institutional_candidates.include_state_updates 必須是 true 或 false"
        )

    stocks = config.get("stocks")
    if not isinstance(stocks, list) or not stocks:
        raise ConfigError(f"{prefix}stocks 必須至少有一筆")

    symbols: set[str] = set()
    for index, stock in enumerate(stocks):
        if not isinstance(stock, dict):
            raise ConfigError(f"{prefix}stocks[{index}] 必須是物件")
        _reject_unknown(stock, _ALLOWED_STOCK_KEYS, prefix, f"stocks[{index}]")
        symbol = str(stock.get("symbol") or "").strip()
        if not symbol:
            raise ConfigError(f"{prefix}stocks[{index}].symbol 必填")
        if symbol in symbols:
            raise ConfigError(f"{prefix}stocks 股票代號重複: {symbol}")
        symbols.add(symbol)

        average_cost = stock.get("average_cost")
        if (
            not isinstance(average_cost, (int, float))
            or isinstance(average_cost, bool)
            or float(average_cost) <= 0
        ):
            raise ConfigError(f"{prefix}stocks[{index}].average_cost 必須是大於 0 的數字")

        alerts = stock.get("profit_alerts")
        if not isinstance(alerts, list) or not alerts:
            raise ConfigError(f"{prefix}stocks[{index}].profit_alerts 必須是非空陣列")
        normalized: list[float] = []
        for alert in alerts:
            if (
                not isinstance(alert, (int, float))
                or isinstance(alert, bool)
                or float(alert) <= 0
            ):
                raise ConfigError(
                    f"{prefix}stocks[{index}].profit_alerts 只能包含大於 0 的數字"
                )
            normalized.append(float(alert))
        if len(normalized) != len(set(normalized)):
            raise ConfigError(f"{prefix}stocks[{index}].profit_alerts 不可重複")
        if normalized != sorted(normalized):
            raise ConfigError(f"{prefix}stocks[{index}].profit_alerts 必須由小到大排列")


def enabled_stocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [deepcopy(stock) for stock in config.get("stocks", [])]


def institutional_candidates_settings(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("institutional_candidates") or {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_new_candidates": int(raw.get("max_new_candidates", 6)),
        "include_state_updates": bool(raw.get("include_state_updates", True)),
    }


def _reject_unknown(
    value: dict[str, Any],
    allowed: set[str],
    prefix: str,
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{prefix}{label} 含不再支援的欄位: {unknown}")
