# Phase 6：TWSE 上市法人模型訓練與驗證

## 目的

建立與 TPEx 完全分離的 TWSE 法人模型研究管線。TWSE 股票只和同日合格上市股票比較，不會套用 TPEx 模型的特徵順序、權重、百分位或生命週期結果。

本階段只完成歷史訓練、樣本外驗證、生命週期回測及最終模型產出；不修改目前正式的每日法人更新、Discord 通知或 state 分支格式。

## 為什麼不能直接部署

既有 Phase 4B 已經得到以下結果：

- TPEx：選出 `core22_l2_1e-3`，判定為 `RANKING_ONLY`。
- TWSE：開發期選出 `full40_l2_1e-3`，但確認期判定為 `REJECT`。
- TWSE 確認期高低組報酬差約為 `0.001447`，穩定性不足。

因此這版保留 TWSE 開發期選出的 40 特徵規格，但必須重新以 20 日相對報酬目標完成樣本外驗證。未通過時只保存研究報告，不產生可部署模型。

## 市場規格

| 市場 | 候選規格 | 特徵數 | 模型目標 |
|---|---|---:|---|
| TPEx | `core22_l2_1e-3` | 22 | 同日未來 20 日報酬排名 |
| TWSE | `full40_l2_1e-3` | 40 | 同日未來 20 日報酬排名 |

兩個市場使用不同的資料母體、前處理統計量、權重、百分位及模型檔案。

## 執行流程

```powershell
.\scripts\run-institutional-twse-training.ps1
```

腳本使用現有 Phase 3 分片，依序執行：

1. Phase 4D：TWSE 10／20／40 日持有期研究。
2. Phase 4E：TWSE 20 日 `return_rank_score` 樣本外驗證。
3. Phase 4F：TWSE 候選及生命週期驗證。
4. Phase 5D：使用全部成熟標籤訓練 TWSE 最終模型並重播歷史通知。

不需要重新下載 Phase 1～2，也不需要重建 Phase 3，前提是本機仍保留：

```text
research/data/institutional_phase1.sqlite
research/data/phase3_shards/
research/output/phase3_summary.csv
research/output/phase3_dataset_manifest.csv
research/output/phase3_stock_summary.csv
research/output/phase3b_summary.csv
```

## TWSE 樣本外放行條件

Phase 4E 的確認期排除最新年度結果必須同時符合：

- 至少 3 個完整確認年度。
- 正向年度至少占 60%，且至少 2 年。
- 前 20% 減後 20%的未來 20 日超額報酬大於 0。
- 平均每日 Spearman 相關係數大於 0。
- 月份區塊 bootstrap 95% 信賴區間下緣大於 0。

任一條件不成立，命令會在 Phase 4E 後以非零狀態停止，並保留：

```text
research/output/twse/phase4e_target_validation_reports.zip
```

這是正常的防呆，不代表程式執行失敗。

## 通過後的產物

研究報告：

```text
research/output/twse/phase4d_horizon_validation_reports.zip
research/output/twse/phase4e_target_validation_reports.zip
research/output/twse/phase4f_lifecycle_validation_reports.zip
research/output/twse/phase5d_final_model_reports.zip
```

TWSE 專用快取與模型：

```text
research/data/phase4d_cache_twse/
research/data/phase5_models_twse/phase5d_final_rank_model.json
research/data/phase5_models_twse/phase5d_final_rank_model.npz
```

TPEx 原本的：

```text
research/data/phase4d_cache/
research/data/phase5_models/
```

不會被覆蓋。

## 驗收

```powershell
python -m pytest -q
python -m compileall -q src research tests scripts
```

本次局部更新驗證結果：

```text
82 tests passed
compileall passed
```

## 尚未包含

即使 TWSE 訓練通過，本階段也不會自動修改正式排程。後續仍需另外完成：

- TWSE 官方每日行情及三法人更新。
- TWSE 每日模型推論。
- TWSE 獨立生命週期 state。
- TWSE 推估成本帶。
- Discord 通知市場標示與去重。

只有收到並審核 `phase5d_final_model_reports.zip` 後，才進入正式部署整合。
