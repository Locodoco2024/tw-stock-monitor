# 台股持股損益與三大法人候選通知

正式監控只保留兩項功能：

1. 配置股的即時價格、成本損益與獲利門檻通知。
2. TPEx 三大法人 `return_rank_score` 候選及 20～40 日生命週期通知。
3. TWSE 三大法人 40 日排名模型候選與生命週期通知。

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

## 持股損益監控

```bash
python -m src.main
```

啟用已產生的法人通知計畫：

```bash
python -m src.main --enable-institutional-candidates
```

## Phase 5H 法人每日管線

第一次在本機用既有 SQLite 建立最近 100 個市場日的部署 seed：

```powershell
.\scripts\seed-institutional-deployment.ps1
```

將 seed 發佈至 GitHub 的 `state` branch：

```powershell
.\scripts\publish-institutional-seed.ps1
```

之後可在 GitHub Actions 手動執行：

```text
TPEx institutional daily update
```

管線使用 TPEx 官方資料，會補齊 seed 與最新日之間缺少的交易日，計算 22 個法人特徵、同日百分位、生命週期及 `runtime/institutional/notification_plan.csv`。第一版盤後 workflow 尚未設定自動排程，法人通知全域開關也仍預設關閉。

## Phase 5I 本機追價風險研究

```powershell
.\scripts\run-institutional-phase5i.ps1
```

Phase 5I 研究法人候選的近期淨買超推估成本帶及進場價偏離，不會修改目前 Discord 通知。推估成本不是法人真實持倉成本；詳細限制見 `PHASE5I_UPDATE.md`。

## Phase 5J 法人推估成本資訊

法人 Discord 通知會額外顯示三法人合計近 20 日推估成本帶、中間值、訊號日收盤價、相對偏離及淨買超日數。推估成本沿用 Phase 5I 定義，只供使用者自行判斷，不是法人真實持倉成本，也不產生買賣建議。詳細規格見 `PHASE5J_UPDATE.md`。

## Phase 6C — TWSE lifecycle and liquidity validation

After Phase 6B passes, run:

```powershell
.\scripts\run-institutional-twse-lifecycle.ps1
```

Phase 6C keeps the Phase 6B model frozen and compares TWSE entry thresholds,
consecutive confirmation days, cooldown periods, 40/60/80-day outcomes, and
money-plus-board-lot liquidity universes. It only produces research reports and
does not modify the production TPEx pipeline or Discord notifications.


## Phase 6D — TWSE production pipeline

After Phase 6B and Phase 6C pass:

```powershell
.\scripts\run-institutional-twse-final.ps1
.\scripts\seed-twse-institutional-deployment.ps1
.\scripts\publish-twse-institutional-seed.ps1
```

The TWSE production universe requires a 20-day median trading amount of at least
NT$100 million and median volume of at least 300 board lots. A tracking event is
created only after three consecutive days in the TWSE top 10%, and ends after 40
market days. See `PHASE6D_TWSE_DEPLOYMENT_UPDATE.md`.
