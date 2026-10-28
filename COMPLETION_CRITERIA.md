# Lab2 Completion Criteria

檢查對象：本機實際產物。`[x]` 代表已驗證；`[ ]` 代表未完成，不能只因有程式碼便勾選。

## 已確認

- [x] 四個原始資料檔齊全，可讀取且兩類平衡。
- [x] 輸入 `(N,1,2,750)`，原訓練／測試各 1,080 筆。
- [x] seed 42、受試者＋類別分層，train 864／validation 216；索引無重疊且涵蓋原訓練集。
- [x] 原始 train/test 的 NaN 分別為 905／689；補值平均僅由訓練子集估計。
- [x] 程式語法檢查通過。
- [x] 使用者提供學號 315553010、姓名楊敦傑。
- [x] 已閱讀 E3 作業頁，截止為 2026/11/05 23:59（Asia/Taipei）；檢查當下尚未繳交。

## 執行驗收

- [x] 專案環境完成，PyTorch 2.9.1+cu126 可使用 RTX 4070；`pip check` 通過，完整依賴版本已保存。
- [x] 兩模型與三種 activation 的 forward/backward、有限梯度檢查通過。
- [x] 五筆資料分成 2／2／1 的 batch，平均 loss 與一次直接計算相符。
- [x] checkpoint 選擇含同分規則，兩輪 smoke run 重載後重現指標。
- [x] 不同工作目錄資料讀取與既有目錄防覆寫檢查通過。
- [x] 八組各 150 epochs 均完成（共 1,200 輪），個別保存設定、CSV、權重與曲線。
- [x] 八組 checkpoint 均在訓練後重載，重現選定輪次的 test accuracy 與 loss。
- [x] 各實驗的 split 索引完全一致。
- [x] 比較 CSV、摘要、峰值輪次、報告與原始紀錄一致；報告產生前的 assert 檢查通過。
- [x] 實際 Chrome 結果截圖已保存並放入報告，明確標示為結果頁截圖。
- [x] 報告包含 Introduction、Experiment setups and result、Discussion、GitHub Link 待填欄及 Bonus。
- [x] 報告檔名為 `LAB2_315553010_楊敦傑.pdf`，12 頁均已視覺檢查；公式缺字已修正並重新確認。
- [x] README、環境清單與可上傳 GitHub 的程式包已備妥：`output/source/LAB2_315553010_楊敦傑_code.zip`；排除原始資料、虛擬環境與 checkpoint。

## 效能目標

- [x] 2026/10/28 依使用者要求將原圖 2 網頁截圖替換為 Word 原生數值表；第 137／139 輪的六項指標均與原始 CSV 核對，後續圖號已重排。
- [x] 新版 `output/docx/LAB2_315553010_楊敦傑_數值表版.docx` 包含 7 個可編輯表格與 18 張圖，保留原 Word 檔已填寫的 GitHub 網址；11 頁經 Word 匯出及視覺檢查。本次未重新驗證遠端或提交狀態。

- [x] 依 Lab1 報告統一 Word 版字型、標題、表格及頁首頁尾，已填入學號與姓名。
- [x] 可編輯 DOCX 已產生：`output/docx/LAB2_315553010_楊敦傑.docx`；文字、6 個表格及 ELU 公式均為原生可編輯內容，19 張圖保留為圖片。
- [x] DOCX 經 Microsoft Word 匯出，共 11 頁，已逐頁檢查圖表、公式及分頁；Lab1 來源檔未修改。

- [x] V2 預先規劃的 validation-selected 五 seed ensemble：**87.7778%**。
- [ ] V2 預先指定主要成果（全資料五 seed ensemble）≥87%：**86.0185%**，未達到目標。V1 的 85.2778% 保留為歷史基準。

依 validation 選出的設定為 EEGNet＋ReLU，epoch 139，validation 89.8148%、test 85.2778%。逐輪最高 test accuracy 在 ReLU 與 LeakyReLU 兩組並列；並未依 test 峰值保存權重。DeepConvNet 選定權重 test accuracy 為 77.5926%，逐輪峰值為 79.1667%。

## 提交前待辦

- [x] GitHub repository `https://github.com/Fergus4506/LAB2_EEGNet_MedicalImaging` 已建立並可公開存取，網址已填入 V2 報告第 4 節。
- [x] 2026/10/28 V2 程式碼、README 與 results_v2 成果（不含 checkpoint）已 commit 並 push 至 GitHub。
- [x] V2 報告章節改為規格的 1. Introduction～5. Bonus，新增 2.6 Highest Testing Accuracy (Screenshot)，移除內部備註；DOCX 已重新產生。
- [ ] 以 Word 將 V2 DOCX 另存為 PDF（本機 Word COM 自動化無法使用），逐頁檢查。
- [ ] 上傳 `output/pdf/V2/LAB2_315553010_楊敦傑.pdf` 至 E3，確認提交狀態成功。

效能目標為 87%，以實測結果為準；單一 seed 不提供穩定達標保證。本次不以 test 峰值挑選權重，也不會為追求門檻追加 test 導向搜尋。

驗證紀錄：`output/verification/selfcheck.log`；獨立 CLI 重載結果：`lab2_EEG_classification/results/evaluation.json`。最佳設定重載測試為 85.2778%，loss 0.41139699，與選定 epoch 的紀錄一致。
## V2 效能目標與驗收（2026/10/28）

- [x] A–F 選項實作，舊權重相容；固定 split seed 42，訓練 seeds 分離。
- [x] selfcheck PASS：兩種卷積、梯度、平均 loss、增強可重現、train-only 正規化、重載、全資料最後權重與防覆寫。
- [x] 16 個候選至少三 seeds，最佳設定五 seeds，全資料五 seeds；搜尋只依 validation。
- [x] test 前已凍結設定、78 輪重訓與主要成果；無 test 峰值選擇。
- [x] comparison／ablation CSV、逐組曲線、最終圖表與中文 Discussion 完成；舊 results/ 雜湊未變。
- [x] 兩種五 seed mean±SD 與 softmax ensemble 已完整報告；± 為樣本 SD。
- [x] V2 已整合至 output/docx/V2/LAB2_315553010_楊敦傑.docx 與 output/pdf/V2/LAB2_315553010_楊敦傑.pdf；重寫摘要、實驗銜接、Discussion 與結論，15 頁經 Word 匯出及逐頁視覺檢查，核心數值與原始紀錄一致。

| 預先固定的報告方法 | Test accuracy (%) | 選擇依據 |
|---|---|---|
| 864 筆訓練，5 seeds 個別 checkpoint | 84.7222 ± 0.8045 | 各 seed 的 validation accuracy，再比較 loss |
| 864 筆訓練，5 seeds ensemble | 87.7778 | 五個 checkpoint 等權 softmax 平均 |
| 1,080 筆重訓，5 seeds 個別模型 | 82.9630 ± 1.9278 | 固定 78 epochs，最後一輪 |
| 1,080 筆重訓，5 seeds ensemble（主要成果） | 86.0185 | 預先指定，五個全資料模型等權平均 |

V2 結果以 results_v2/final_summary.json 為準。原先的「不追加搜尋」敘述是 V1 歷史紀錄；本次依使用者新指示追加 validation-only 搜尋，沒有依新 test 結果繼續調參。未 commit、push 或提交 E3。
