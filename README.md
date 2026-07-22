# TW Stock Monitor

以 Python、GitHub Actions 與 Discord Webhook 建立的台股規則式監控工具。

這個專案不使用 AI，也不直接預測未來股價。它把公開資料轉成可追溯的規則結果，輸出：

- 未持有：買入／避開傾向指數
- 已持有：加碼／續抱／減碼傾向指數
- 個股客觀分析指數
- 機會分數、風險分數與資料完整度
- 兩至三句固定模板摘要
- 完整 HTML 與 JSON 計算報告

> 傾向百分比是規則分數，不是未來上漲或下跌機率，也不是投資建議。

## 資料來源

- Fugle MarketData REST API：盤中即時報價
- FinMind：個股歷史行情、加權／櫃買報酬指數、月營收、財務報表
- TWSE OpenAPI：上市公司每日重大訊息
- TPEx OpenAPI：上櫃公司每日重大訊息

官方重大訊息採保守關鍵字與否定詞規則。一般新聞不進入分數。

## 計分架構

個股客觀分析由五個模組組成：

| 模組 | 權重 |
|---|---:|
| 官方事件與催化 | 25 |
| 市場定價狀態 | 25 |
| 產業與同業確認 | 20 |
| 公司承接能力 | 20 |
| 大盤環境 | 10 |

持有股票另加入：

- 達到 30%、50%、75% 獲利時的停利調整
- 虧損達門檻時的條件式加碼評估
- 虧損本身不會自動提高買入分數
- 只有客觀分析、資料完整度與官方事件皆符合條件時，才會列為加碼觀察

所有權重、門檻與文字規則都在 `configs/scoring.yaml`，不是寫死在 Python。

## 專案結構

```text
.
├── .github/workflows/
│   ├── hourly-monitor.yml
│   └── test.yml
├── configs/
│   ├── scoring.yaml
│   └── users/example.yaml
├── src/
│   ├── providers/
│   ├── scoring/
│   ├── notifications/
│   ├── reports/
│   └── state/
├── tests/
│   └── fixtures/sample_bundle.json
├── runtime/
├── site/
└── README.md
```

## 先用離線資料測試

不需要任何 API Key：

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest
python -m src.main `
  --offline-fixture tests/fixtures/sample_bundle.json `
  --no-discord `
  --state-file runtime/test-state.json `
  --output-dir site-test
```

完成後開啟：

```text
site-test/index.html
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
python -m src.main \
  --offline-fixture tests/fixtures/sample_bundle.json \
  --no-discord \
  --state-file runtime/test-state.json \
  --output-dir site-test
```

## 建立自己的配置

複製範例：

```bash
cp configs/users/example.yaml configs/users/locodoco.yaml
```

一個 YAML 代表一位使用者：

```yaml
user:
  id: locodoco
  enabled: true
  discord_webhook_key: default

app:
  timezone: Asia/Taipei
  market_symbol: TAIEX
  report_base_url: ""

stocks:
  - symbol: "2330"
    enabled: true

    holding:
      enabled: true
      average_cost: 620

    profit_take:
      alert_at_pct: 50
      strong_alert_at_pct: 75

    add_on:
      enabled: true
      alert_at_loss_pct: [10, 20, 30]
      minimum_analysis_score: 40
      maximum_adjustment: 10

    peers: ["2303", "2454"]
```

### 欄位說明

- `symbol`: 股票代碼，股票識別資訊中唯一必填欄位
- `name`: 選填；預設由 Fugle 或 FinMind 自動取得，只有需要覆蓋時才填
- `market`: 選填；預設自動辨識上市 `TWSE` 或上櫃 `TPEx`
- `benchmark`: 選填；上市預設使用 `app.market_symbol`，上櫃預設使用 `TPEx`
- `holding.enabled`: 是否已持有
- `average_cost`: 平均成本價，不需要股數
- `profit_take.alert_at_pct`: 一般停利提醒
- `profit_take.strong_alert_at_pct`: 強停利提醒
- `add_on.alert_at_loss_pct`: 虧損到哪些區間時重新評估
- `add_on.minimum_analysis_score`: 客觀分析至少幾分才允許列為加碼候選
- `peers`: 真正可比較的同業，建議手動指定
- `market_symbol`: 上市股票預設比較指數，通常使用 `TAIEX`；上櫃股票會自動改用 `TPEx`


### 股票名稱與市場自動辨識

一般情況只需要填股票代碼：

```yaml
stocks:
  - symbol: "2330"

  - symbol: "5347"
    holding:
      enabled: true
      average_cost: 100
```

程式會依序使用：

1. 配置中的 `name`／`market` 覆蓋值（若有）
2. Fugle 即時報價回傳的名稱與交易所
3. FinMind `TaiwanStockInfo` 作為備援

如果兩個資料來源都無法辨識，程式才會要求在該股票下選填：

```yaml
market: TWSE  # 或 TPEx
```

## 本機執行正式資料

建立 `.env` 並在終端機設定環境變數，或直接使用系統環境變數：

```text
FUGLE_API_KEY
FINMIND_TOKEN
DISCORD_WEBHOOK_URL
REPORT_BASE_URL
```

`FINMIND_TOKEN` 選填，但使用 Token 通常有較合適的請求額度。

PowerShell 範例：

```powershell
$env:FUGLE_API_KEY="your-key"
$env:FINMIND_TOKEN="your-token"
$env:DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
python -m src.main --no-discord
```

先保留 `--no-discord` 確認報告，再移除參數測試通知。

## 多人 Discord Webhook

只有一個使用者時使用：

```text
DISCORD_WEBHOOK_URL
```

多人可建立 GitHub Secret：

```text
DISCORD_WEBHOOKS_JSON
```

內容：

```json
{
  "default": "https://discord.com/api/webhooks/...",
  "alice": "https://discord.com/api/webhooks/..."
}
```

配置中的 `user.discord_webhook_key` 對應上面的 key。

## 推到 GitHub

先在 GitHub 建立空白 Repository，不要勾選自動建立 README。

```bash
git init
git add .
git commit -m "Initial stock monitor"
git branch -M main
git remote add origin https://github.com/ACCOUNT/REPOSITORY.git
git push -u origin main
```

## GitHub Secrets 與 Variables

Repository → **Settings → Secrets and variables → Actions**

### Secrets

```text
FUGLE_API_KEY
FINMIND_TOKEN
DISCORD_WEBHOOK_URL
```

多人時改用：

```text
DISCORD_WEBHOOKS_JSON
```

### Variable

```text
REPORT_BASE_URL=https://ACCOUNT.github.io/REPOSITORY
```

這不是機密，請放在 Variables，不需要放 Secrets。

## 啟用 GitHub Pages

1. Repository → Settings → Pages
2. Source 選擇 **Deploy from a branch**
3. Branch 選擇 `gh-pages`
4. Folder 選擇 `/ (root)`

第一次 Action 執行後才會自動建立 `gh-pages` branch。

## GitHub Actions 行為

`hourly-monitor.yml`：

- 每小時第 17 分執行
- 時區 `Asia/Taipei`
- 也可以由 Actions 頁面手動執行
- 產生最新 HTML／JSON 報告
- 報告發布到 `gh-pages` branch
- 通知狀態保存到獨立 `state` branch
- 不會在 `main` branch 每小時產生 commit

公開 Repository 連續 60 天沒有活動時，GitHub 可能停用 scheduled workflow；重新啟用 Action 或提交 commit 即可。

## 通知去重

系統只在以下情況通知：

- 第一次分析
- 操作傾向跨越區間
- 指數變動超過設定門檻
- 風險分數明顯上升
- 發現新的可分類官方事件
- 達到持倉停利門檻

狀態保存在 `state` branch 的 `state.json`。

## 官方重大訊息的限制

沒有 AI 的版本只處理明確文字規則，例如：

- 取得訂單
- 簽訂供貨合約
- 開始量產／出貨
- 擴產
- 終止合約
- 取消訂單

如果文字包含「並未、否認、澄清、非屬事實」等否定語境，規則會忽略正面關鍵字。

無法可靠分類的官方訊息會出現在 HTML 報告，但不會計分。這是刻意的保守設計。

## 測試

```bash
python -m pytest
ruff check src tests
```

測試涵蓋：

- EMA、RSI、ATR 與報酬計算
- 官方事件否定語境
- 公告金額與重大度
- 50% 停利調整
- 虧損不符合條件時禁止加碼
- 狀態保存與首次通知
- 離線端到端執行

## 目前 MVP 限制

- 不自動下單
- 不串券商持股
- 不使用 AI 或新聞情緒分析
- 財報欄位只辨識目前列出的常見 FinMind 科目名稱
- 金融、保險、證券業應另建專用財務模型
- ETF 不應直接套用普通公司的基本面分數
- GitHub Actions 可能延遲，不能當即時停損系統
- 免費 API 的配額與欄位可能調整，provider 已獨立封裝以便更換

## License

MIT
