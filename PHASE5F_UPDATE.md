# Phase 5F：正式通知接入（預設關閉）

## 目的

將 Phase 5E 已產生的 `phase5e_notification_plan.csv` 接入既有 `src.main` 與 Discord，
同時保持原本股票評分、持股成本、停利與加碼邏輯不變。

法人通知固定為 `TRACK_ONLY`，不會產生買進或賣出建議。

## 雙重開關

法人通知必須同時通過兩層設定才會執行：

1. 使用者 YAML：

```yaml
institutional_candidates:
  enabled: true
  max_new_candidates: 6
  include_state_updates: true
```

2. 執行參數：

```powershell
python -m src.main --enable-institutional-candidates
```

GitHub Actions 可透過手動輸入 `institutional_candidates=true`，或 Repository Variable：

```text
ENABLE_INSTITUTIONAL_CANDIDATES=true
```

預設全部為 `false`，套用更新不會自動啟用。

## 配置欄位

### institutional_candidates.enabled

是否讓該使用者接收三大法人候選與生命週期通知。

### institutional_candidates.max_new_candidates

單次最多加入幾檔不在 `stocks` 清單中的新法人候選。
配置股與既有法人事件的狀態更新不受這個數量限制。

### institutional_candidates.include_state_updates

- `true`：包含布局確認、第 20 日延長等狀態更新。
- `false`：只通知新進前 10% 與直接布局確認。

## peers 是否需要

`peers` 仍為選填，只供既有規則模型使用：

- 產業與同業模組權重為 20。
- 市場定價模組會使用同業近 5 日報酬做相對比較。
- 沒有 peers 時，正式股票分析仍能執行；相對報酬退回只比較大盤，產業與同業模組不可用。
- 三大法人模型不使用 peers，自動法人候選也不要求設定 peers。

若沒有真正業務相近的公司，寧可省略或使用 `peers: []`，不要為了填欄位加入不相干股票。

## 發送去重

成功發送後，以下唯一鍵會保存於 `runtime/state.json`：

```text
user_id | event_id | notification_type | signal_date
```

GitHub Actions 每 30 分鐘重跑時，不會重複發送同一法人狀態。

## 通知內容

配置股若同時有法人事件，法人通知會附上原本的：

- 操作判斷
- 方向分數
- 目前價格（有資料時）

但法人狀態不會改變原分數。

非配置股則只發法人候選訊息，包含：

- 同日法人百分位
- 事件進度
- 主要正向與負向法人因素
- `TRACK_ONLY` 提示

## 資料前提

正式接入只讀取：

```text
research/output/phase5e_notification_plan.csv
```

只會載入：

- `eligible_for_future_github = 1`
- `ready_to_send = 1`
- `trade_action = TRACK_ONLY`

本更新不負責產生最新 Phase 5E 計畫。全域開關啟用但檔案不存在時，程式會明確失敗，避免在沒有法人資料時假裝已完成通知。
