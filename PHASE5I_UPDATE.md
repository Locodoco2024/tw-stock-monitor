# Phase 5I：法人推估成本帶與追價風險驗證

Phase 5I 不修改正式 Discord 通知或 GitHub Actions。它使用 Phase 4F 已凍結的兩種正式進榜事件：

- `top10_confirm1d`：首次進入前 10%，對應 `NEW_CANDIDATE`
- `top20_confirm5d`：連續 5 日維持前 20%，對應 `LAYOUT_CONFIRMED_DIRECT`

## 成本代理定義

公開資料無法還原法人真實庫存成本。Phase 5I 只建立「近期淨買超推估成本帶」：

1. 分別計算外資、投信、自營商自行買賣及三法人合計。
2. 比較 5、10、20 個市場交易日窗口。
3. 只使用淨買超為正的日期。
4. 以淨買超股數加權當日 `(最高價＋最低價＋收盤價)／3` 作為中間成本。
5. 同時以最低價與最高價加權，形成推估成本帶。
6. 自營商避險不納入。

這個數值不是法人真實剩餘持倉成本，也不能推論特定機構的帳面損益。

## 進場與驗證

- 訊號條件只使用訊號日及以前資料。
- 實際進場價使用事件確認後下一交易日開盤。
- 比較進場價距推估成本的偏離率。
- 同時檢查訊號前 5／20 日漲幅及近 20 日價格區間位置。
- 未來 20／40 日報酬、最大上漲與最大回撤只用於歷史驗證。

## 執行

```powershell
.\scripts\run-institutional-phase5i.ps1
```

提高 bootstrap 次數：

```powershell
.\scripts\run-institutional-phase5i.ps1 `
  --phase5i-bootstrap-iterations 2000
```

## 主要輸出

```text
research/output/phase5i_input_audit.csv
research/output/phase5i_event_cost_features.csv.gz
research/output/phase5i_cost_proxy_comparison.csv
research/output/phase5i_deviation_bucket_analysis.csv
research/output/phase5i_deviation_quantile_analysis.csv
research/output/phase5i_yearly_stability.csv
research/output/phase5i_bootstrap_confidence.csv
research/output/phase5i_overheat_rule_comparison.csv
research/output/phase5i_rule_candidates.csv
research/output/phase5i_summary.csv
research/output/phase5i_entry_risk_validation_reports.zip
```

`pipeline_status=PASS` 只代表報告完整。只有在確認期中，高成本偏離組的後續超額報酬較差、最大回撤較深，且 bootstrap 支持時，才可把該代理提升為正式「追價風險」提示。
