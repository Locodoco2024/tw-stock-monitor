# Phase 5G 更新說明

## 目的

正式監控改為：

- 持股損益門檻通知
- TPEx 法人指數與生命週期通知

## 已移除

- `src/scoring/`
- `src/reports/`
- 舊分析資料聚合器
- FinMind 正式監控來源
- 官方重大事件與同業比較
- `configs/scoring.yaml`
- 舊模型 HTML 報告與 GitHub Pages 部署
- 舊模型相關測試

## 新配置

每檔只保留：

```yaml
- symbol: "1815"
  average_cost: 97.85
  profit_alerts: [30, 50, 75]
```

## 狀態遷移

舊 `runtime/state.json` 可直接讀取。第一次執行會把當前已達門檻設為新基準，不會因格式遷移重新發送所有舊停利通知。之後若股價跌破門檻再重新站回，仍會再次通知。
