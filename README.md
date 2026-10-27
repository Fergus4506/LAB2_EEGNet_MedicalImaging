# Lab2 - EEG Classification

以提供的 BCI Competition III IIIb 資料完成 EEGNet、七組單一變因實驗及 DeepConvNet 加分比較。模型自行以 PyTorch 層組合；不使用預訓練模型。

## 安裝與執行

以下 PowerShell 指令從此 README 所在資料夾執行。使用 Python 3.12；不必啟用虛擬環境，也不必更改 PowerShell execution policy。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe lab2_EEG_classification/selfcheck.py
.\.venv\Scripts\python.exe lab2_EEG_classification/main.py --suite
```

`requirements.txt` 固定此次使用的 GPU 版 PyTorch。無 NVIDIA GPU 時亦可使用 CPU，但耗時較長；若改用其他 PyTorch build，請將環境差異記錄於報告。完整套件版本記於 `requirements-lock.txt`。

資料不隨程式庫發布。請將老師提供的四個檔案放在 `lab2_EEG_classification/data/`：`S4b_train.npz`、`X11b_train.npz`、`S4b_test.npz`、`X11b_test.npz`。路徑依 `dataloader.py` 所在位置解析，與啟動時工作目錄無關。

單組實驗範例（舊的 `-num_epochs`、`-batch_size`、`-lr` 仍可使用）：

```powershell
.\.venv\Scripts\python.exe lab2_EEG_classification/main.py --model eegnet --activation elu --alpha 1 --dropout 0.25 --num-epochs 150 --batch-size 64 --lr 0.01 --seed 42 --output lab2_EEG_classification/results_custom
.\.venv\Scripts\python.exe lab2_EEG_classification/main.py --evaluate lab2_EEG_classification/results/eegnet_baseline/best.pt
```

已有同名實驗目錄時拒絕覆寫，請指定新的 `--output`。僅載入可信來源的 checkpoint。

## 實驗設計

- 輸入 `(B,1,2,750)`，輸出 `(B,2)` logits，標籤為 int64 的 0／1；CrossEntropyLoss 前不加 softmax。
- 依受試者與類別分層，以 seed 42 將原訓練集切成 train 864／validation 216；test 1,080 不參與訓練或選模。這是相同受試者內的分層切分，不是跨受試者泛化評估。
- NaN 以 864 筆訓練子集的單一全域平均值補入 train／validation／test。沒有額外正規化或資料增強。
- 各組固定 150 epochs、batch size 64、Adam、lr 0.01、weight decay 0.001。每組重新固定種子、初始化模型與 optimizer；不提前停止。
- checkpoint 優先選 validation accuracy 高者，同分選 validation loss 低者，再同分保留較早 epoch；test 指標不參與此判斷。
- train 指標在訓練模式、更新過程中累計；validation／test 在每輪更新後以 eval 模式評估。因此 dropout 開啟的 train accuracy 不必高於 test accuracy。
- 依作業要求逐輪呈現 test 曲線，但七組設定在訓練前已固定，不依 test 結果追加搜尋。逐輪最高 test accuracy 只作描述，並非最終 checkpoint 的選擇依據。

| 代號 | 模型 | Activation | ELU alpha | Dropout |
|---|---|---|---:|---:|
| eegnet_baseline | EEGNet | ELU | 1 | 0.25 |
| eegnet_relu | EEGNet | ReLU | 不適用 | 0.25 |
| eegnet_leakyrelu | EEGNet | LeakyReLU (0.01) | 不適用 | 0.25 |
| eegnet_alpha05 | EEGNet | ELU | 0.5 | 0.25 |
| eegnet_alpha2 | EEGNet | ELU | 2 | 0.25 |
| eegnet_dropout0 | EEGNet | ELU | 1 | 0 |
| eegnet_dropout05 | EEGNet | ELU | 1 | 0.5 |
| deepconvnet_baseline | DeepConvNet | ELU | 1 | 0.5 |

EEGNet 後段採真正的 depthwise `(1,15), groups=32` 加 pointwise `(1,1)`，不同於投影片第 8 頁的普通卷積。DeepConvNet 採 25／50／100／200 filters、四次 max pooling，依作業允許省略 max norm。

## 成果與重現

本機八組皆完成 150 epochs，並通過保存權重的重載核對。依 validation 選定的設定是 **EEGNet + ReLU**（epoch 139）：validation 89.81%、test **85.28%**；逐輪最高 test accuracy 也是 85.28%（ReLU 與 LeakyReLU 並列）。**尚未達到 87% 目標**，沒有以 test 結果追加調參。

| 設定 | Validation 選定權重的 Test (%) | 逐輪 Test 峰值 (%) |
|---|---:|---:|
| EEGNet 基準 | 74.63 | 76.57 |
| ReLU | 85.28 | 85.28 |
| LeakyReLU | 82.87 | 85.28 |
| ELU alpha=0.5 | 83.80 | 83.80 |
| ELU alpha=2 | 70.83 | 73.24 |
| Dropout=0 | 69.44 | 73.89 |
| Dropout=0.5 | 73.06 | 74.63 |
| DeepConvNet | 77.59 | 79.17 |

`lab2_EEG_classification/results/` 每個實驗含：

- `config.json`：超參數、補值平均、環境及原始資料 SHA-256。
- `split.npz`：原訓練資料中的 train／validation 索引。
- `history.csv`、`train.log`：150 輪完整指標與文字紀錄。
- `best.pt`：驗證集選定的權重、設定及 epoch，訓練後會重載並核對 test 評估。
- `acc.png`、`loss.png`：train／validation／test 曲線；loss 使用對數軸以呈現早期高值與後期趨勢。
- `summary.json`：驗證集選定 checkpoint 的 test 結果與逐輪 test 峰值。

根目錄的 `comparison.csv` 彙整八組結果。`selected_test_acc` 與 `peak_test_acc` 代表不同問題，不可混用。單一 seed 不提供統計顯著性結論；同設定在不同硬體／套件版本上可能仍有差異。

報告來源與產生指令：

報告位於 `output/pdf/`。

## 驗收與提交

實際完成與待完成項目見 `COMPLETION_CRITERIA.md`。

`.gitignore` 排除資料、虛擬環境、快取、checkpoint 及冗長 log；CSV、設定與圖可隨程式庫保存。若要分享訓練權重，可另用 GitHub release 或其他下載位置。尚未提供 GitHub 網址，因此本次不會建立或發布遠端 repository，也不會代填 E3。

## 參考

- 課程提供的 `Lab2_EEGNET_spec.pdf` 與程式骨架。
- Lawhern et al., EEGNet (2018): https://arxiv.org/abs/1611.08024
- Clevert et al., ELU (ICLR 2016): https://arxiv.org/abs/1511.07289
- Schirrmeister et al., Deep ConvNets (2017/2018 version): https://arxiv.org/abs/1703.05051
- PyTorch Conv2d: https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html
