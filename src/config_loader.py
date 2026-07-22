from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


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
    user = config.get("user")
    if not isinstance(user, dict) or not user.get("id"):
        raise ConfigError(f"{prefix}user.id 必填")
    stocks = config.get("stocks")
    if not isinstance(stocks, list) or not stocks:
        raise ConfigError(f"{prefix}stocks 必須至少有一筆")
    for index, stock in enumerate(stocks):
        if not isinstance(stock, dict) or not stock.get("symbol"):
            raise ConfigError(f"{prefix}stocks[{index}].symbol 必填")
        holding = stock.get("holding", {})
        if holding.get("enabled") and not isinstance(holding.get("average_cost"), (int, float)):
            raise ConfigError(
                f"{prefix}stocks[{index}].holding.enabled=true 時 average_cost 必須是數字"
            )


def enabled_stocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [deepcopy(stock) for stock in config.get("stocks", []) if stock.get("enabled", True)]
