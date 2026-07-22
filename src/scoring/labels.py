from __future__ import annotations

from typing import Any


def objective_label(score: float, labels: dict[str, Any]) -> str:
    if score >= float(labels.get("strong_positive", 70)):
        return "強烈偏多"
    if score >= float(labels.get("positive", 40)):
        return "偏多"
    if score >= float(labels.get("slight_positive", 15)):
        return "輕微偏多"
    if score <= float(labels.get("strong_negative", -70)):
        return "強烈偏空"
    if score <= float(labels.get("negative", -40)):
        return "偏空"
    if score <= float(labels.get("slight_negative", -15)):
        return "輕微偏空"
    return "中性觀察"


def operation_label(score: float, holding_enabled: bool, labels: dict[str, Any]) -> str:
    if holding_enabled:
        if score >= float(labels.get("strong_positive", 70)):
            return "強烈加碼傾向"
        if score >= float(labels.get("positive", 40)):
            return "加碼傾向"
        if score >= float(labels.get("slight_positive", 15)):
            return "輕微加碼傾向"
        if score <= float(labels.get("strong_negative", -70)):
            return "強烈減碼傾向"
        if score <= float(labels.get("negative", -40)):
            return "減碼傾向"
        if score <= float(labels.get("slight_negative", -15)):
            return "輕微減碼傾向"
        return "續抱觀察"
    if score >= float(labels.get("strong_positive", 70)):
        return "強烈買入傾向"
    if score >= float(labels.get("positive", 40)):
        return "買入傾向"
    if score >= float(labels.get("slight_positive", 15)):
        return "輕微買入傾向"
    if score <= float(labels.get("strong_negative", -70)):
        return "強烈避開傾向"
    if score <= float(labels.get("negative", -40)):
        return "避開傾向"
    if score <= float(labels.get("slight_negative", -15)):
        return "輕微避開傾向"
    return "中性觀察"
