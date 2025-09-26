# 自動化電子郵件分類器

這個專案提供一個簡單、可擴充的自動化電子郵件分類工作流程。核心邏輯是一個以 Python 編寫的多項式 Naive Bayes 分類器，能將寄件者、主旨與內文轉換成關鍵字，再根據已標註的資料進行學習並產生分類結果。

## 功能特色

- **純 Python 實作**：不依賴第三方機器學習套件，部署更輕量。
- **資料前處理**：內建簡易的文字正規化與停用字詞移除，提升分類效果。
- **訓練 / 驗證流程**：可設定驗證資料比例並回報準確率。
- **模型儲存與載入**：使用 `pickle` 序列化，方便部署至排程或伺服器。
- **CLI 工具**：提供訓練 (`scripts/train_model.py`) 與分類 (`scripts/classify_emails.py`) 指令。

## 快速開始

### 1. 建立虛擬環境並安裝套件

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用者改為 .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. 準備資料

- `data/sample_training_data.json`：示範性的訓練資料，包含 `sender`、`subject`、`body` 與 `label` 欄位。
- `data/sample_inbox.json`：示範性的未標註信件，可用來測試分類結果。

依據此格式即可建立自己的資料集。

### 3. 訓練模型

```bash
python scripts/train_model.py data/sample_training_data.json models/email_classifier.pkl
```

常用參數：

- `--validation-split`：保留多少比例做為驗證資料（預設 0.2）。
- `--seed`：設定亂數種子以確保結果可重現。

### 4. 分類新信件

```bash
python scripts/classify_emails.py models/email_classifier.pkl data/sample_inbox.json --show-probabilities
```

若加上 `--output result.json` 可將分類結果存成 JSON 檔。

## 套件使用方式

你也可以在程式碼中直接使用分類器：

```python
from email_classifier import EmailClassificationPipeline, EmailMessage, LabeledEmail

pipeline = EmailClassificationPipeline()
training_data = [
    LabeledEmail(EmailMessage("sender@example.com", "subject", "body"), "label"),
]
pipeline.train(training_data)

prediction = pipeline.predict(EmailMessage("new@example.com", "subject", "text"))
print(prediction)
```

## 測試

```bash
pytest
```

## 後續延伸

- 增加更多前處理步驟（如詞幹化、N-gram）。
- 將模型整合至 IMAP 收信流程，定期抓取並分類。
- 將分類結果寫入資料庫或觸發自動化工作。
