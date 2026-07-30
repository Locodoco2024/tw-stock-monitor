# Phase 5H：TPEx 每日法人資料與通知計畫

Phase 5H 將 Phase 5D 最終 `return_rank_score` 模型部署到 GitHub Actions。

## 資料來源

- 最新資料：TPEx OpenAPI 的上櫃股票行情與三大法人買賣明細。
- 中斷補資料：TPEx 官方依日期查詢的上櫃行情與三大法人日報表。
- 不需要 `FINMIND_TOKEN`。
- 自營商只使用自行買賣，不包含避險。

## 一次性本機 seed

```powershell
.\scripts\seed-institutional-deployment.ps1
```

預設從 `research/data/institutional_phase1.sqlite` 匯出最近 100 個市場交易日到：

```text
runtime/institutional/
```

接著發佈到 `state` branch：

```powershell
.\scripts\publish-institutional-seed.ps1
```

## GitHub 手動驗證

在 Actions 執行：

```text
TPEx institutional daily update
```

第一版 workflow 只有 `workflow_dispatch`，不會自動排程。成功後會在 `state` branch 更新：

```text
institutional/rolling_market_data.csv.gz
institutional/universe.csv
institutional/latest_scores.csv.gz
institutional/lifecycle_events.csv
institutional/lifecycle_notifications.csv
institutional/notification_plan.csv
institutional/update_manifest.json
```

## 中斷補資料

若 seed 最新日與 TPEx 最新日相差數天，管線會逐日查詢官方歷史日報，補齊中間交易日；週末與休市日自動略過。預設最多補 31 個日曆日，超過時拒絕直接跳日並要求重新建立 seed。

## 通知邊界

- 所有法人事件固定為 `TRACK_ONLY`。
- 只有新市場日期產生的計畫會設為 `ready_to_send=1`。
- 同日重跑會變成 `ALREADY_CURRENT`，不重新發送。
- 原盤中持股損益 workflow 保持原排程。
- 法人候選全域開關仍預設關閉。
