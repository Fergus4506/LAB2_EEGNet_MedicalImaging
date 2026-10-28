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

提交用報告為 V2 版：`output/docx/V2/LAB2_315553010_楊敦傑.docx` 與 `output/pdf/V2/LAB2_315553010_楊敦傑.pdf`，由 `build_report_v2_docx.py` 產生，章節依作業規格編為 1. Introduction、2. Experiment Setups and Results、3. Discussion、4. GitHub Link、5. Bonus。實際完成與待完成項目見 `COMPLETION_CRITERIA.md`。

`.gitignore` 排除資料、虛擬環境、快取、checkpoint 及冗長 log；CSV、設定與圖可隨程式庫保存。若要分享訓練權重，可另用 GitHub release 或其他下載位置。

## 參考

- 課程提供的 `Lab2_EEGNET_spec.pdf` 與程式骨架。
- Lawhern et al., EEGNet (2018): https://arxiv.org/abs/1611.08024
- Clevert et al., ELU (ICLR 2016): https://arxiv.org/abs/1511.07289
- Schirrmeister et al., Deep ConvNets (2017/2018 version): https://arxiv.org/abs/1703.05051
- PyTorch Conv2d: https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html

## V2：固定 validation、多 seed 與最終 test 評估

本節為 V2 最新流程；前文八組設定與逐輪 test 曲線是保留的 V1 歷史成果。V2 不產生逐輪 test 指標，所有訓練曲線只有 train／validation；全資料重訓只有 train。預設輸出改為 results_v2，既有目錄與 comparison.csv 均拒絕覆寫。原 results/ 全部檔案雜湊一致。

| 預先固定的報告方法 | Test accuracy (%) | 選擇依據 |
|---|---|---|
| 864 筆訓練，5 seeds 個別 checkpoint | 84.7222 ± 0.8045 | 各 seed 的 validation accuracy，再比較 loss |
| 864 筆訓練，5 seeds ensemble | 87.7778 | 五個 checkpoint 等權 softmax 平均 |
| 1,080 筆重訓，5 seeds 個別模型 | 82.9630 ± 1.9278 | 固定 78 epochs，最後一輪 |
| 1,080 筆重訓，5 seeds ensemble（主要成果） | 86.0185 | 預先指定，五個全資料模型等權平均 |

共 16 個候選，每個至少三 seeds；最佳設定 c15 再跑至五 seeds。固定 split seed 42；訓練 seed 控制初始化、shuffle、dropout 與專用增強 generator。平均值與樣本 SD（ddof=1）不得解讀為跨切分的信賴區間。主要成果 未達到 87%。

```powershell
.\.venv\Scripts\python.exe lab2_EEG_classification/selfcheck.py
# 完整重現請使用尚不存在的新目錄，避免覆寫成果
.\.venv\Scripts\python.exe lab2_EEG_classification/experiments_v2.py --phase search --output lab2_EEG_classification/results_v2_reproduce
.\.venv\Scripts\python.exe lab2_EEG_classification/experiments_v2.py --phase final --output lab2_EEG_classification/results_v2_reproduce
# 單組三 seed validation-only ablation
.\.venv\Scripts\python.exe lab2_EEG_classification/main.py --activation relu --separable full --weight-decay 0.001 --optimizer adam --scheduler cosine --lr 0.001 --normalize --augment --seeds 42 43 44 --validation-only --output lab2_EEG_classification/results_custom_v2
# 已凍結最佳設定的五 seed 全資料重訓與 ensemble
.\.venv\Scripts\python.exe lab2_EEG_classification/main.py --activation relu --alpha 1.0 --dropout 0.25 --separable full --weight-decay 0.001 --optimizer adam --scheduler none --lr 0.01 --augment --num-epochs 78 --full-train --seeds 42 43 44 45 46 --ensemble --output lab2_EEG_classification/results_final_reproduce
# 重新載入已凍結的十個 checkpoint，印出各 seed 與兩種 ensemble 的 test 結果（報告 2.6 節截圖來源）
.\.venv\Scripts\python.exe lab2_EEG_classification/show_final_v2.py
# 重現 validation-selected 五 seed ensemble（本次 87.7778%）
.\.venv\Scripts\python.exe lab2_EEG_classification/main.py --activation relu --alpha 1.0 --dropout 0.25 --separable full --weight-decay 0.001 --optimizer adam --scheduler none --lr 0.01 --augment --num-epochs 300 --seeds 42 43 44 45 46 --ensemble --output lab2_EEG_classification/results_selected_reproduce
```

所有新模型選項預設保持原超參數：depthwise、Adam、weight decay 0.001、scheduler none、不 normalize、不 augment。--split-seed 預設 42，與 --seed／--seeds 分開。--full-train 只保存固定輪數的最後權重，summary 的 selection=fixed_epochs_full_train；輪數必須先由 validation 決定。--validation-only 完全不載入 test 樣本；一般單組模式只在訓練完成後評估一次。--ensemble 需搭配 --seeds，以各模型自己的訓練統計量前處理後平均 softmax。

normalize 使用訓練子集每通道的平均與母體標準差（跨 trial 與時間），std 下限 1e-8；全資料訓練改用全部 1,080 筆重新估計。NaN 仍先由實際訓練資料的全域平均補值。增強依序為每筆 circular shift ±50、振幅 ×0.9–1.1、加入 σ=0.05×原 trial／channel 時間標準差的 Gaussian noise，僅訓練 batch 啟用。統計量、增強規則與 seed 都記錄於 config.json。

完整結果與可貼入報告的分析：[REPORT.md](lab2_EEG_classification/results_v2/REPORT.md)、[Discussion](lab2_EEG_classification/results_v2/discussion_zh.md)。逐 seed 結果為 comparison.csv；ablation.csv 列設定與三 seed 統計；ablation_stages.csv 列各階段相對 parent 的差值。每組保留 config、split、history、best.pt、重載驗證、summary、acc/loss 圖；final_summary.json 包含兩種五 seed mean±SD 及 ensemble。protocol.json、decisions.json、frozen_selection.json、final_evaluation_started.json 記錄 test 接觸前的選擇依據。

本次將資料切分固定為 seed 42、受試者與類別分層的 864／216 筆，僅改變模型初始化、batch 順序、dropout 與資料增強的訓練 seed。每個候選使用 seeds 42、43、44，比較各自 validation 選定 checkpoint 的平均 accuracy，同分時比較平均 loss。共測試 16 個不同設定，最後固定 c15，再補 seeds 45、46；沒有根據新增兩個 seed 的結果重選設定。所有 test 評估均在設定、五個 seeds、重訓輪數與 ensemble 規則凍結後進行。

將後段 depthwise＋pointwise 改為一般時間卷積後，該卷積區塊的權重由 32×15＋32×32＝1,504 增為 32×32×15＝15,360，增加 13,856 個參數；全模型因此由 4,018 增為 17,874，約 4.45 倍。一般卷積可為每一對輸入／輸出特徵學習獨立的時間濾波器；depthwise＋pointwise 則先逐特徵濾波，再做跨特徵線性混合，因此具有較強的結構限制。此處的 full 是投影片式的一般卷積，並非嚴格意義的 separable convolution。初始三 seed 比較中，full 相對 depthwise 的平均 validation accuracy 改變 +1.3889 個百分點，這是此資料與設定下的實證結果；不能僅憑參數較多便推論一定較佳。

最終設定為 full、relu（α=1.0）、dropout=0.25、adam、lr=0.01、weight decay=0.001、scheduler=none、normalize=False、augment=True，搜尋訓練上限 300 epochs。正規化只從實際訓練資料估計每通道 mean/std；增強只作用於訓練 batch，使用各 trial／channel 的時間標準差設定噪聲。較小 learning rate、cosine 衰減、weight decay、dropout 與增強分別改變最佳化或正則化，但是否採用完全依 ablation 的 validation 結果。這是依序的貪婪比較，後續對照是在前面選定設定上進行，不能解讀為所有改動互相獨立的因果效果，也沒有涵蓋全部交互作用。

五 seed validation-selected checkpoint 的 test accuracy 為 84.7222 ± 0.8045%，全資料固定輪數重訓為 82.9630 ± 1.9278%；± 表示五次訓練的樣本標準差（ddof=1），不是信賴區間。全資料訓練輪數 78 為五個 validation 最佳輪次 [243, 49, 70, 214, 78] 的中位數，各模型只取最後一輪。這種作法使用更多訓練資料，但不保證原切分的最佳輪數能直接轉移至全資料訓練。兩種五模型等權 softmax ensemble 的 test accuracy 分別為 87.7778% 與 86.0185%；主要成果在評估前已指定為後者，未達到 87% 目標。Ensemble 可平均部分初始化與訓練噪聲，但五個模型共享資料，其誤差並非獨立，改善幅度應以實測為準。

這些 seeds 共用同一個 validation 切分，因此標準差只反映訓練隨機性，不涵蓋重新切分資料或跨受試者的不確定性。多次搜尋也可能對固定 validation 過度擬合。舊版 test 85.28% 已在此前作業中被檢視，本次不能宣稱是完全未曾接觸的全新盲測；但新流程沒有以 test 挑選 checkpoint、設定、輪數或追加實驗。Ablation 的 test 數值僅在最後統一補列，不參與搜尋，沒有使用逐輪 test 峰值。

預先列入報告的 validation-selected ensemble 為 87.7778%，達到 87%；它屬於預先固定的另一種評估方法，不能事後取代主要成果。

進一步看逐項比較，加入增強後平均 validation accuracy 增加 0.6173 個百分點，最終補列的 test 平均增加 0.1852 個百分點。時間位移讓模型接觸不同時間對齊，振幅縮放與噪聲增加訊號擾動，可能降低對特定波形細節的依賴；但這三項是共同開啟，本實驗無法分辨哪一項貢獻最大。150 與 300 epochs 的三 seed 平均 validation accuracy 差為 0.0000 個百分點，因此 300 epochs 是依較低的 validation loss 勝出，不能描述成 validation accuracy 提升。正規化、較低 weight decay、較低 learning rate、cosine 與 AdamW 在本次逐步比較中均未獲選；它們不是普遍無效，只是目前樣本、固定切分與搜尋路徑下沒有得到較佳驗證結果。

V2 整合報告：Word 位於 output/docx/V2/LAB2_315553010_楊敦傑.docx，PDF 位於 output/pdf/V2/LAB2_315553010_楊敦傑.pdf，含 11 個可編輯表格、原生 ELU 公式、最高 testing accuracy 執行輸出截圖與各組曲線。舊版保留，提交請使用 V2 目錄版本。
