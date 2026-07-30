# 台股持股損益與三大法人候選通知

正式監控只保留兩項功能：

1. 配置股的即時價格、成本損益與獲利門檻通知。
2. TPEx 三大法人 `return_rank_score` 候選及 20～40 日生命週期通知。

舊規則分析、財報、同業、大盤、技術評分、加碼與買賣建議已移除。

## 配置格式

```yaml
user:
  id: example
  enabled: true
  discord_webhook_key: default

stocks:
  - symbol: "2330"
    average_cost: 620
    profit_alerts: [30, 50, 75]
```

每檔股票只需要：

- `symbol`：股票代號
- `average_cost`：平均成本
- `profit_alerts`：由小到大的獲利百分比門檻

若需要個別調整法人候選通知，可選填：

```yaml
institutional_candidates:
  enabled: true
  max_new_candidates: 6
  include_state_updates: true
```

未填時預設啟用；GitHub Actions 的全域開關仍必須開啟才會發送法人通知。

## 本機測試

```powershell
.\scripts\run-offline.ps1
```

## 正式執行

```bash
python -m src.main
```

啟用法人通知計畫：

```bash
python -m src.main --enable-institutional-candidates
```
