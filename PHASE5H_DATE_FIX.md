# Phase 5H seed 日期相容修正

修正本機 SQLite 日期欄位帶有時間（例如 `2026-07-23 00:00:00`）時，seed 日期正規化後被全部濾除，最終誤報最新日 0 檔的問題。

調整內容：

- seed 最近交易日改由 `stock_prices` 實際有資料的日期取得。
- SQL 查詢統一將日期正規化為 `YYYY-MM-DD`。
- `stock_prices` 與 `institutional_flows` 允許日期是否帶時間不同。
- Python 日期解析支援 ISO 日期時間、斜線日期、民國日期及 compact 日期。
- 新增 SQLite timestamp 回歸測試。

覆蓋後執行：

```powershell
python -m pytest -q
.\scripts\seed-institutional-deployment.ps1
```
