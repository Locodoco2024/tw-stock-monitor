# 法人佈局模型 Phase 1

此目錄是獨立的本機研究模組，不會改動目前正式監控、Discord 通知或 GitHub Actions。

## Phase 1 目的

先驗證以下資料鏈，不進行正式模型訓練：

```text
法人歷史資料
→ 排除自營商避險
→ 股價與公司行動對齊
→ 計算公司行動還原後報酬
→ 依全市場交易日曆建立未來 5／10／20 個交易日結果
→ 以 10 日正負 5% 產生 UP／FLAT／DOWN 標籤
```

法人輸入只保留：

- `Foreign_Investor`
- `Investment_Trust`
- `Dealer_self`

`Dealer_Hedging` 會保存於 SQLite 供核對，但不會計入 `selected_total_net`，之後也不得進入模型特徵。

## 40 檔驗證股票

`config/validation_universe_v1.csv` 包含：

- 21 檔目前持股
- 19 檔資料邊界案例

這 40 檔只用於驗證資料鏈；正式模型仍會使用全體上市普通股，以及通過流動性門檻的上櫃普通股。

## 執行前準備

在專案根目錄啟用虛擬環境並安裝原專案套件：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PowerShell 設定 FinMind Token：

```powershell
$env:FINMIND_TOKEN="你的 Token"
```

Token 不會寫入資料庫或輸出報告。

## 先跑還原邏輯自我檢查

```powershell
python -m research.institutional_model.cli self-check
```

## 執行完整 Phase 1

```powershell
python -m research.institutional_model.cli phase1
```

也可以拆開執行：

```powershell
python -m research.institutional_model.cli init
python -m research.institutional_model.cli download
python -m research.institutional_model.cli labels
python -m research.institutional_model.cli report
```

只驗證少數股票：

```powershell
python -m research.institutional_model.cli phase1 --symbols 2330,2409,8070
```

重新抓取既有區間：

```powershell
python -m research.institutional_model.cli download --force
```

## 本機產物

不提交 GitHub：

```text
research/data/institutional_phase1.sqlite
research/output/phase1_validation_10d.csv
research/output/phase1_stock_summary.csv
research/output/phase1_corporate_actions.csv
research/output/phase1_signal_date_audit.csv
research/output/phase1_zero_price_audit.csv
```

### `phase1_validation_10d.csv`

每列是一個法人訊號日，包含：

- T+1 進場日與開盤價
- 第 10 個全市場交易日與該日收盤價
- 原始報酬
- 公司行動還原後報酬
- 期間最高／最低還原報酬
- 公司行動種類
- UP／FLAT／DOWN 標籤
- 無法計算時的明確原因

## 還原規則

- 進場日：`收盤價 ÷ 開盤價`
- 一般持有日：`當日收盤價 ÷ 前一交易日收盤價`
- 公司行動日：`當日收盤價 ÷ 官方恢復／除權息參考價`
- T+1 本身若為公司行動日，因為是當日開盤才進場，因此不套用前一日權益調整。

同一天出現多筆公司行動且官方參考價差異超過 1% 時，不自行猜測，該筆標記為 `invalid_data` 供人工核對。

## Phase 1.1 交易日對齊

- 法人訊號日必須同時存在於 `TaiwanStockTradingDate`，且個股當日有正常成交。
- T+1 與第 N 日都依全市場交易日曆決定，不會因個股停牌而延後數月湊滿 N 筆股價。
- T+1 沒有有效開盤價時標記 `unavailable_entry_price`。
- 固定目標交易日沒有有效收盤價時標記 `unavailable_target_price`。
- 持有期間跨越終止上市櫃日時標記 `delisted_before_target`，暫不自行假設下市處分價格。
- 非交易日法人列與零價格原始列分別輸出到兩份稽核 CSV。

## Phase 1.2 資料清理

- 公司行動資料重新下載時，會以同一股票、來源與日期區間完整替換，避免 API 修正後舊資料殘留。
- 若同一公司行動在短期內出現相同內容的重複日期，只保留有正常交易價格的實際交易日。
- 下市日當天及之後的股價不參與標籤、流動性與最近 20 日統計。
- `phase1_zero_price_audit.csv` 會另外標示 `post_delisting_price`，供確認資料來源是否仍回傳下市後價格。
- `phase1_stock_summary.csv` 新增 `excluded_post_delisting_price_days`，並將 `price_end` 限制在下市日前最後一筆價格。

## Phase 2：全市場歷史資料回補

Phase 2 只負責建立股票母體與回補 2015 年至今的歷史資料，尚不產生模型特徵，也不訓練模型。

### 股票母體

- 目前上市、上櫃普通股：以 TWSE／TPEx 官方現行公司基本資料為準，`TaiwanStockInfo` 只補名稱與產業別。
- 掛牌日期：優先使用 TWSE／TPEx 官方公司基本資料，避免把股票仍在興櫃期間的價格誤當成上市櫃資料。
- 研究期間內下市櫃候選：由 `TaiwanStockDelisting` 保留，市場別由 `TaiwanStockInfo` 的歷史紀錄補足。
- 終止上市公司：另用 TWSE 官方終止上市清單確認市場別。
- 歷史下市櫃股票完成全區間下載後，以首筆有效法人交易日作為保守可用起日，避免把興櫃期間納入模型。
- 存託憑證、ETF、ETN、特別股、權證與受益證券不進入普通股研究母體。
- 可在 `config/market_overrides_v1.csv` 補上人工核對後的市場別與掛牌日期。

### 每檔回補資料

```text
TaiwanStockPrice
TaiwanStockInstitutionalInvestorsBuySell
TaiwanStockDividendResult
TaiwanStockCapitalReductionReferencePrice
```

全市場共用資料仍沿用：

```text
TaiwanStockInfo
TaiwanStockTradingDate
TaiwanStockSplitPrice
TaiwanStockParValueChange
TaiwanStockDelisting
```

### 分批執行

免費 API 需要逐檔查詢。預設每次最多處理 100 檔，約最多 400 個逐檔 API 請求；額度用完時程式會安全停止，重跑後從未完成位置繼續。

PowerShell：

```powershell
$env:FINMIND_TOKEN="你的 Token"
.\scripts\run-institutional-phase2.ps1
```

調整批次大小：

```powershell
.\scripts\run-institutional-phase2.ps1 --max-stocks 50
```

只處理上市或上櫃：

```powershell
.\scripts\run-institutional-phase2.ps1 --market twse
.\scripts\run-institutional-phase2.ps1 --market tpex
```

尚未確認市場別的歷史下市櫃候選預設不下載；確認要一併回補時：

```powershell
.\scripts\run-institutional-phase2.ps1 --include-unclassified
```

也可拆開執行：

```powershell
python -m research.institutional_model.cli phase2-universe
python -m research.institutional_model.cli phase2-download --max-stocks 100
python -m research.institutional_model.cli phase2-report
```

### Phase 2 報告

```text
research/output/phase2_universe.csv
research/output/phase2_download_progress.csv
research/output/phase2_unresolved_universe.csv
research/output/phase2_summary.csv
```

- `phase2_universe.csv`：完整股票母體、掛牌／下市日期與是否可進模型。
- `phase2_download_progress.csv`：每檔四個資料集的下載狀態與筆數。
- `phase2_unresolved_universe.csv`：市場別、掛牌日期或代號重用仍待確認的股票。
- `phase2_summary.csv`：母體數量、完成股票數及預估剩餘 API 請求數。

## Phase 3A：法人特徵與訓練前資料集

Phase 3A 不再呼叫 FinMind API，直接沿用 Phase 2 SQLite。它只建立可重跑、可續接的本機資料集，尚不訓練模型。

### 執行

```powershell
.\scripts\run-institutional-phase3.ps1
```

中途休眠、關機或關閉 PowerShell 後，重新執行相同指令即可。每完成一檔股票就會寫入：

```text
research/data/phase3_shards/<設定簽章>/<股票代號>.csv.gz
```

未完成的單一股票會在下次重建；已完成分片會直接略過。

只先測少數股票：

```powershell
.\scripts\run-institutional-phase3.ps1 --symbols 2330,2303
```

限制本次最多處理 100 檔：

```powershell
.\scripts\run-institutional-phase3.ps1 --phase3-max-stocks 100
```

### 資料時間規則

- 訊號日為 T，特徵只使用 T 收盤以前資料。
- 進場價固定為 T+1 全市場交易日開盤。
- 報酬目標為第 5、10、20 個全市場交易日收盤。
- 10 日還原報酬先固定為小數第 10 位，再以大於等於 5% 判為 `UP`、小於等於 -5% 判為 `DOWN`，其餘為 `FLAT`。
- 公司行動沿用 Phase 1 已驗證的官方參考價還原規則。
- 掛牌日晚於 Phase 2 固定研究截止日的股票，不進入本次研究快照。

### 法人特徵

模型輸入只來自：

- 外資
- 投信
- 自營商自行買賣
- 上述三類合計

每類建立：

- 1／3／5／10／20 日淨買賣超占同期成交量比例
- 5／10／20 日淨買超天數比例
- 截至 T 的連續淨買超或淨賣超天數

另建立三類法人方向一致程度，以及法人合計 5 日相對 20 日的流向加速度。

`Dealer_Hedging` 不會寫入 Phase 3 訓練資料，也不會出現在特徵字典中。成交量只用來將不同股票的法人張數標準化，不作為獨立技術指標。

### TPEx 流動性條件

每一個上櫃訊號日都使用最近 20 個全市場交易日動態判斷：

- 至少 18 個正常交易日
- 成交金額中位數至少新台幣 2,000 萬元
- 不得連續 3 個市場交易日零成交量
- T+1 必須有有效進場價格

主訓練集使用 2,000 萬元門檻，同時保留 1,000／2,000／5,000 萬／1 億元四組旗標，供下一階段比較樣本數與穩定性。

### 產物

完整訓練資料使用 gzip CSV，避免單一未壓縮 CSV 過大：

```text
research/output/phase3_training_twse.csv.gz
research/output/phase3_training_tpex.csv.gz
```

驗證報告：

```text
research/output/phase3_summary.csv
research/output/phase3_stock_summary.csv
research/output/phase3_label_distribution.csv
research/output/phase3_exclusion_reasons.csv
research/output/phase3_liquidity_thresholds.csv
research/output/phase3_feature_dictionary.csv
research/output/phase3_dataset_manifest.csv
research/output/phase3_validation_reports.zip
```

上傳驗證時只需提供 `phase3_validation_reports.zip`，不用上傳可能很大的兩份訓練資料。

## Phase 3B：特徵品質稽核

Phase 3B 直接掃描 Phase 3A 產生的兩份 gzip 訓練檔，不呼叫 FinMind API，也不修改 SQLite 或訓練資料。

### 執行

```powershell
.\scripts\run-institutional-phase3b.ps1
```

預設每次讀取 100,000 列，並從 TWSE、TPEx 各取最多 100,000 列做分位數、相關性及標籤方向分析。可依電腦記憶體調整：

```powershell
.\scripts\run-institutional-phase3b.ps1 `
  --audit-chunk-size 50000 `
  --audit-sample-size 50000
```

### 稽核內容

- 40 個模型特徵的空值、非數字、正負無限值、最小值、最大值及分位數
- 常數、近乎常數及極端法人流量欄位
- 高度相關或數值重複的特徵組合
- 各年份平均值相對全期間的漂移程度
- UP／FLAT／DOWN 三類標籤下的特徵平均值與中位數
- 股票代號與訊號日期鍵值唯一性、排序及 manifest 筆數／SHA-256
- 10 日標籤與還原報酬是否一致
- T+1 進場日與第 10 市場交易日目標日是否正確對齊市場日曆
- 模型欄位是否混入標籤、未來報酬、價格、股票代號、市場別或自營商避險
- 以合成資料改寫 T 之後的所有資料，確認 T 日特徵完全不變

完整數值統計使用全資料流式計算；分位數與相關性使用固定雜湊抽樣，避免一次載入超過 500 MB 的訓練檔。

### 產物

```text
research/output/phase3b_summary.csv
research/output/phase3b_feature_profile.csv
research/output/phase3b_feature_issues.csv
research/output/phase3b_high_correlation.csv
research/output/phase3b_yearly_drift.csv
research/output/phase3b_label_feature_direction.csv
research/output/phase3b_key_audit.csv
research/output/phase3b_leakage_audit.csv
research/output/phase3b_validation_reports.zip
```

`phase3b_summary.csv` 的 `ready_for_modeling=1` 代表沒有阻擋模型訓練的錯誤。高度相關、年度漂移及極端值會列為警告，需在正式建模前決定保留、標準化、截尾或移除。


## Phase 3C：標籤邊界一致性修正

Phase 3C 只串流修正既有 Phase 3 分片中的 `label_10d`，不重新計算法人特徵、不呼叫 FinMind API。每完成一檔即寫入 SQLite 續接狀態；中斷後重跑相同指令即可略過已完成股票。

```powershell
.\scripts\run-institutional-phase3c.ps1
```

處理順序：

```text
統一 10 日還原報酬為小數第 10 位
→ 修正正負 5% 邊界標籤
→ 重建 TWSE／TPEx 訓練檔
→ 更新標籤分布、manifest 與 SHA-256
→ 自動重跑 Phase 3B
```

完成後上傳：

```text
research/output/phase3c_validation_reports.zip
```

## Phase 4A：時間滾動基準模型

Phase 4A 將 TWSE、TPEx 分開執行 expanding-window 驗證。每折使用更早年度訓練、測試前一年校準、下一年度真正樣本外測試。所有截尾及標準化參數只從該折訓練資料取得。

```powershell
.\scripts\run-institutional-phase4a.ps1
```

第一版使用可解釋的三類 multinomial logistic regression，輸出 `P_down`、`P_flat`、`P_up`，並計算：

```text
法人佈局指數 = 100 × (P_up - P_down)
```

每完成一折即保存結果，可在休眠、關機或中止後續跑。完整說明見專案根目錄 `PHASE4A_UPDATE.md`。完成後上傳：

```text
research/output/phase4a_validation_reports.zip
```

## Phase 4B：模型穩定化與保留／淘汰判定

Phase 4B 不重新下載資料，也不修改 Phase 3 訓練檔。它先以 2019～2022 年樣本外折作為開發期，選出每個市場自己的候選模型，再把 2023 年以後保留為未參與選型的確認期。

```powershell
.\scripts\run-institutional-phase4b.ps1
```

固定比較三組候選：

- `full40_l2_1e-4`：Phase 4A 原始 40 特徵基準；設定相同時直接沿用 Phase 4A 折結果。
- `full40_l2_1e-3`：40 特徵並提高 L2 正則化。
- `core22_l2_1e-3`：只保留三類法人各自的 1／5／20 日流量、5／20 日買超比例、連續方向，以及法人一致程度與合計加速度。

候選選擇只使用開發期，依序考慮正向高低組報酬年度數、中位高低組報酬、加權高低組報酬及原始機率 Log Loss。選定後才進入 2023 年以後確認期。

機率方案同時比較：

- 原始 softmax 機率
- 單年度 Temperature scaling
- 依校準年度選擇強度的歷史機率收縮

法人佈局指數固定使用原始機率排序，避免校準方法改變橫斷面排序。最終市場判定為：

- `PROBABILITY_AND_RANKING`：確認期排序與機率品質都通過。
- `RANKING_ONLY`：確認期排序通過，但機率仍不足以解讀為可靠發生率。
- `REJECT`：確認期排序方向不夠穩定。

每完成一個「市場＋年度＋候選」就保存折結果；中斷後重新執行相同指令會續接。產物：

```text
research/output/phase4b_summary.csv
research/output/phase4b_candidate_development.csv
research/output/phase4b_selected_candidates.csv
research/output/phase4b_market_decisions.csv
research/output/phase4b_fold_summary.csv
research/output/phase4b_metrics.csv
research/output/phase4b_index_deciles.csv
research/output/phase4b_calibration.csv
research/output/phase4b_coefficients.csv
research/output/phase4b_preprocessing.csv
research/output/phase4b_training_history.csv
research/output/phase4b_feature_sets.csv
research/output/phase4b_validation_reports.zip
```

Phase 4B 仍是本機研究，不修改 GitHub Actions 的正式監控模型。

## Phase 4C：TPEx 同日橫斷面排序驗證

Phase 4C 不重新下載資料、不重跑 Phase 1～3，也不重新挑選 Phase 4B 候選。它固定使用 Phase 4B 已選出的 TPEx `core22_l2_1e-3`，從 `phase4b_coefficients.csv` 與 `phase4b_preprocessing.csv` 還原每個測試年度的樣本外模型，再重新讀取既有 Phase 3 資料產生逐筆原始法人佈局指數。

```powershell
.\scripts\run-institutional-phase4c.ps1
```

主要差異是排名改為每個 `signal_date` 獨立計算：

```text
原始法人佈局指數 = 100 × (P_up - P_down)
同日法人佈局百分位 = 該股票在同一天 TPEx 候選股票中的相對排名
```

Phase 4C 不把 `P_up`、`P_down` 解讀為已校準機率，而是檢查法人行為指數是否真的能在同一天挑出後續表現相對較好的股票。驗證包含：

- 同日十分位、五分位、Top 10%／20% 與 Bottom 10%／20%。
- 5／10／20 日平均與中位報酬、正報酬率、正負 5% 比例、最大漲幅與最大回撤。
- 年度、月份及排除最新年度的方向穩定性。
- 1,000 萬、2,000 萬、5,000 萬、1 億元流動性門檻敏感度。
- 每日 Top 5、Top 10、Top 20 的報酬與相對全體超額報酬。
- 月份移動區塊 bootstrap 95% 信賴區間。
- Top 10%／20% 選股是否過度集中於少數股票。

1,000 萬門檻會直接讀取既有 `research/data/phase3_shards/<設定簽章>/` 分片，不會重新建立法人特徵或標籤。

產物：

```text
research/output/phase4c_oos_scores.csv.gz
research/output/phase4c_daily_rank_groups.csv
research/output/phase4c_horizon_behavior.csv
research/output/phase4c_yearly_stability.csv
research/output/phase4c_monthly_stability.csv
research/output/phase4c_liquidity_sensitivity.csv
research/output/phase4c_top_n_analysis.csv
research/output/phase4c_bootstrap_confidence.csv
research/output/phase4c_stock_concentration.csv
research/output/phase4c_summary.csv
research/output/phase4c_validation_reports.zip
```

`phase4c_summary.csv` 的 `ready_for_selection_index=1` 必須同時符合：

- 確認期 10 日 Top 20%－Bottom 20% 的同日平均報酬差為正。
- 至少 75% 確認年度方向為正。
- 至少 75% 確認年度的十分位報酬方向相關為正。
- 排除最新年度後報酬差仍為正。
- 月份移動區塊 bootstrap 的 95% 信賴區間下界大於 0。
- 5,000 萬流動性門檻下的確認期報酬差仍為正。

通過只代表可以進入「建立每日選股指數輸出器」的下一階段，不代表已可部署或保證獲利。Phase 4C 仍只在本機研究，不修改 GitHub Actions 正式監控模型。

## Phase 4D：TPEx 10／20／40 日持有期研究

Phase 4D 使用既有歷史資料比較現行 10 日標籤與 20、40 個市場交易日標籤，不下載新資料、不重跑 Phase 1～3，也不修改正式 GitHub Actions。

```powershell
.\scripts\run-institutional-phase4d.ps1
```

10／20／40 日固定使用相同 TPEx `core22_l2_1e-3` 特徵、L2 與正負 5% 標籤門檻。40 日公司行動還原報酬由現有 SQLite 計算並保存於獨立快取，不會改寫 Phase 3 凍結資料。

時間切分額外依 `target_date` purge 年底跨期樣本，確保訓練報酬在校準期開始前已發生、校準報酬在測試期開始前已發生。主要產物為：

```text
research/output/phase4d_horizon_label_distribution.csv
research/output/phase4d_fold_summary.csv
research/output/phase4d_fold_metrics.csv
research/output/phase4d_daily_rank_groups.csv.gz
research/output/phase4d_daily_spreads.csv.gz
research/output/phase4d_yearly_ranking.csv
research/output/phase4d_bootstrap_confidence.csv
research/output/phase4d_coefficients.csv
research/output/phase4d_coefficient_stability.csv
research/output/phase4d_training_history.csv
research/output/phase4d_boundary_purge.csv
research/output/phase4d_summary.csv
research/output/phase4d_horizon_validation_reports.zip
```

`ready_for_horizon_decision=1` 只表示三組期限報告完整，不代表 20 日或 40 日已自動通過。完整方法與驗收原則請見根目錄 `PHASE4D_UPDATE.md`。

## Phase 5A：TPEx 本機法人選股指數

Phase 5A 固定使用 Phase 4C 已驗證的 TPEx `core22_l2_1e-3`，以全部成熟標籤建立最終模型，再從 Phase 3 分片取得最新可用特徵日，輸出同日法人佈局排名與歷史十分位行為。

```powershell
.\scripts\run-institutional-phase5a.ps1
```

主要輸出：

```text
research/output/phase5a_selection_index.csv
research/output/phase5a_selection_index_top20.csv
research/output/phase5a_historical_behavior_lookup.csv
research/output/phase5a_summary.csv
research/output/phase5a_selection_index_reports.zip
```

Phase 5A 的流動性排名只使用訊號日當下已知資料。Phase 3 歷史旗標中的 T+1 開盤有效性只另外顯示，不會參與當日選股排名。

`institutional_selection_index`（等同 `percentile_20m`）是同日相對排名，不是上漲機率；完整規格與限制請見根目錄 `PHASE5A_UPDATE.md`。

## Phase 5B：每日選股參考防呆

```powershell
.\scripts\run-institutional-phase5b.ps1
```

Phase 5B 不重訓模型，會核對 Phase 3 分片、SQLite 市場日曆、TPEx 價格與法人資料日期。只有 `selection_readiness_status=READY` 時才輸出可使用名單；過期截面只保存在 diagnostic 檔案。

## Phase 4E：TPEx 20 日模型目標比較

Phase 4E 固定 20 日為主要期限、40 日為延伸觀察，只使用現有歷史資料比較：

- 現行三分類 multinomial 法人布局指數
- `UP vs NOT_UP` 二分類機率
- `DOWN vs NOT_DOWN` 二分類風險
- 20 日同日未來報酬百分位 Ridge 排名模型

```powershell
.\scripts\run-institutional-phase4e.ps1
```

所有模型使用相同 `core22` 特徵、L2、expanding-window 與 `target_date` purge。二分類同時輸出 Raw、Platt 與歷史機率基準；四種選股分數都比較 20 日主要結果及 40 日延伸結果。

另外輸出外資、投信、自營商自行買賣、三法人一致性四類模型貢獻。完整規格與判定原則見根目錄 `PHASE4E_UPDATE.md`。

## Phase 4F：法人布局訊號生命週期

Phase 4F 固定使用 Phase 4E 的 TPEx 20 日 `return_rank_score` 樣本外百分位，不重新訓練模型。它比較前 20%／10%／5%、連續 1／2／3／5 日確認、0／5／10／20 日冷卻，以及第 20 日維持或轉弱後的 40 日延伸效果。

```powershell
.\scripts\run-institutional-phase4f.ps1
```

規則比較會同時輸出絕對報酬及相對同日 TPEx 母體的超額報酬；bootstrap 以超額報酬為準。詳細規格見專案根目錄的 `PHASE4F_UPDATE.md`。

## Phase 5D：最終排名模型與通知狀態重播

Phase 5D 固定採用 Phase 4E 勝出的 TPEx 20 日 `return_rank_score`，以全部成熟標籤建立可保存／重載的最終 Ridge 排名模型；再使用 Phase 4E OOS 分數重播 Phase 4F 凍結的生命週期規則。

```powershell
.\scripts\run-institutional-phase5d.ps1
```

固定狀態：

- 首次進入前 10%：`NEW_CANDIDATE`
- 連續 5 日前 20%：`LAYOUT_CONFIRMED`
- 第 20 日仍前 20%：延長至第 40 日
- 第 20 日跌出前 20%：結束積極追蹤，但不代表賣出
- 第 40 日：事件結束
- 結束後 20 個交易日冷卻

完整方法、輸出與限制見根目錄 `PHASE5D_UPDATE.md`。
