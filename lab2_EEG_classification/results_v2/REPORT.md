# EEGNet V2：validation 選模與最終評估

## 固定流程

資料切分 seed 42；搜尋 seeds 42／43／44；最佳設定追加 45／46。所有候選皆依 validation 選權重，依三 seed 平均 validation 選設定。全資料固定輪數與主要 ensemble 成果於 test 評估前寫入 frozen_selection.json。Test 沒有逐輪曲線，僅有最後評估。原 results/ 以 SHA-256 核對未修改。

## 最終成果

| 預先固定的報告方法 | Test accuracy (%) | 選擇依據 |
|---|---|---|
| 864 筆訓練，5 seeds 個別 checkpoint | 84.7222 ± 0.8045 | 各 seed 的 validation accuracy，再比較 loss |
| 864 筆訓練，5 seeds ensemble | 87.7778 | 五個 checkpoint 等權 softmax 平均 |
| 1,080 筆重訓，5 seeds 個別模型 | 82.9630 ± 1.9278 | 固定 78 epochs，最後一輪 |
| 1,080 筆重訓，5 seeds ensemble（主要成果） | 86.0185 | 預先指定，五個全資料模型等權平均 |

| Seed | Validation-selected epoch | Validation (%) | Selected test (%) | Full-train test (%) |
|---|---|---|---|---|
| 42 | 243 | 88.4259 | 84.9074 | 82.5926 |
| 43 | 49 | 89.8148 | 85.0000 | 84.8148 |
| 44 | 70 | 90.7407 | 83.9815 | 84.3519 |
| 45 | 214 | 88.4259 | 85.8333 | 79.9074 |
| 46 | 78 | 89.3519 | 83.8889 | 83.1481 |

## 候選設定

| 候選 | Separable | WD | LR / scheduler | Normalize | Augment | Dropout | Activation / α | Optimizer | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| c00 | depthwise | 0.001 | 0.01 / none | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c01 | full | 0.001 | 0.01 / none | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c02 | full | 0.0 | 0.01 / none | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c03 | full | 0.0001 | 0.01 / none | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c04 | full | 0.001 | 0.01 / cosine | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c05 | full | 0.001 | 0.001 / none | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c06 | full | 0.001 | 0.001 / cosine | False | False | 0.25 | relu / 1.0 | adam | 150 |
| c07 | full | 0.001 | 0.01 / none | True | False | 0.25 | relu / 1.0 | adam | 150 |
| c08 | full | 0.001 | 0.01 / none | False | True | 0.25 | relu / 1.0 | adam | 150 |
| c09 | full | 0.001 | 0.01 / none | False | True | 0.1 | relu / 1.0 | adam | 150 |
| c10 | full | 0.001 | 0.01 / none | False | True | 0.5 | relu / 1.0 | adam | 150 |
| c11 | full | 0.001 | 0.01 / none | False | True | 0.25 | leakyrelu / 1.0 | adam | 150 |
| c12 | full | 0.001 | 0.01 / none | False | True | 0.25 | elu / 0.5 | adam | 150 |
| c13 | full | 0.001 | 0.01 / none | False | True | 0.25 | elu / 1.0 | adam | 150 |
| c14 | full | 0.001 | 0.01 / none | False | True | 0.25 | relu / 1.0 | adamw | 150 |
| c15 | full | 0.001 | 0.01 / none | False | True | 0.25 | relu / 1.0 | adam | 300 |

## 依序 ablation

以下每個對照使用相同三個 seeds，Δ 單位為百分點。Test 僅在所有選擇凍結後補列；不是 test 排行榜。候選若與先前完全相同即重用，避免重跑。

| 階段 | Parent → candidate | 採用 | Validation mean ± SD (%) | Δ val (pp) | Test mean ± SD (%) | Δ test (pp) |
|---|---|---|---|---|---|---|
| separable | c00 → c00 | False | 87.6543 ± 2.3302 | +0.0000 | 83.1790 ± 1.8199 | +0.0000 |
| separable | c00 → c01 | True | 89.0432 ± 1.4144 | +1.3889 | 83.5802 ± 1.0157 | +0.4012 |
| weight_decay | c01 → c02 | False | 85.3395 ± 3.0826 | -3.7037 | 80.9568 ± 1.4882 | -2.6235 |
| weight_decay | c01 → c03 | False | 85.1852 ± 3.2407 | -3.8580 | 82.3457 ± 2.4747 | -1.2346 |
| weight_decay | c01 → c01 | True | 89.0432 ± 1.4144 | +0.0000 | 83.5802 ± 1.0157 | +0.0000 |
| lr_scheduler | c01 → c01 | True | 89.0432 ± 1.4144 | +0.0000 | 83.5802 ± 1.0157 | +0.0000 |
| lr_scheduler | c01 → c04 | False | 86.8827 ± 2.2837 | -2.1605 | 83.1481 ± 1.0678 | -0.4321 |
| lr_scheduler | c01 → c05 | False | 87.1914 ± 1.3365 | -1.8519 | 82.6543 ± 0.5658 | -0.9259 |
| lr_scheduler | c01 → c06 | False | 85.4938 ± 2.2837 | -3.5494 | 83.1173 ± 1.7380 | -0.4630 |
| normalize | c01 → c01 | True | 89.0432 ± 1.4144 | +0.0000 | 83.5802 ± 1.0157 | +0.0000 |
| normalize | c01 → c07 | False | 87.1914 ± 1.6259 | -1.8519 | 82.6235 ± 1.7962 | -0.9568 |
| augment | c01 → c01 | False | 89.0432 ± 1.4144 | +0.0000 | 83.5802 ± 1.0157 | +0.0000 |
| augment | c01 → c08 | True | 89.6605 ± 1.1651 | +0.6173 | 83.7654 ± 1.3556 | +0.1852 |
| dropout | c08 → c09 | False | 88.7346 ± 0.7072 | -0.9259 | 83.8889 ± 0.7232 | +0.1235 |
| dropout | c08 → c08 | True | 89.6605 ± 1.1651 | +0.0000 | 83.7654 ± 1.3556 | +0.0000 |
| dropout | c08 → c10 | False | 89.5062 ± 1.4882 | -0.1543 | 83.6420 ± 0.4567 | -0.1235 |
| activation | c08 → c08 | True | 89.6605 ± 1.1651 | +0.0000 | 83.7654 ± 1.3556 | +0.0000 |
| activation | c08 → c11 | False | 88.8889 ± 1.2249 | -0.7716 | 86.0494 ± 0.7875 | +2.2840 |
| activation | c08 → c12 | False | 89.6605 ± 0.2673 | +0.0000 | 81.9444 ± 0.7579 | -1.8210 |
| activation | c08 → c13 | False | 82.7160 ± 4.6833 | -6.9444 | 76.4198 ± 4.3403 | -7.3457 |
| optimizer | c08 → c08 | True | 89.6605 ± 1.1651 | +0.0000 | 83.7654 ± 1.3556 | +0.0000 |
| optimizer | c08 → c14 | False | 88.8889 ± 1.2249 | -0.7716 | 83.9815 ± 2.8927 | +0.2160 |
| epochs | c08 → c08 | False | 89.6605 ± 1.1651 | +0.0000 | 83.7654 ± 1.3556 | +0.0000 |
| epochs | c08 → c15 | True | 89.6605 ± 1.1651 | +0.0000 | 84.6296 ± 0.5632 | +0.8642 |

## Discussion（可直接貼入報告）

本次將資料切分固定為 seed 42、受試者與類別分層的 864／216 筆，僅改變模型初始化、batch 順序、dropout 與資料增強的訓練 seed。每個候選使用 seeds 42、43、44，比較各自 validation 選定 checkpoint 的平均 accuracy，同分時比較平均 loss。共測試 16 個不同設定，最後固定 c15，再補 seeds 45、46；沒有根據新增兩個 seed 的結果重選設定。所有 test 評估均在設定、五個 seeds、重訓輪數與 ensemble 規則凍結後進行。

將後段 depthwise＋pointwise 改為一般時間卷積後，該卷積區塊的權重由 32×15＋32×32＝1,504 增為 32×32×15＝15,360，增加 13,856 個參數；全模型因此由 4,018 增為 17,874，約 4.45 倍。一般卷積可為每一對輸入／輸出特徵學習獨立的時間濾波器；depthwise＋pointwise 則先逐特徵濾波，再做跨特徵線性混合，因此具有較強的結構限制。此處的 full 是投影片式的一般卷積，並非嚴格意義的 separable convolution。初始三 seed 比較中，full 相對 depthwise 的平均 validation accuracy 改變 +1.3889 個百分點，這是此資料與設定下的實證結果；不能僅憑參數較多便推論一定較佳。

最終設定為 full、relu（α=1.0）、dropout=0.25、adam、lr=0.01、weight decay=0.001、scheduler=none、normalize=False、augment=True，搜尋訓練上限 300 epochs。正規化只從實際訓練資料估計每通道 mean/std；增強只作用於訓練 batch，使用各 trial／channel 的時間標準差設定噪聲。較小 learning rate、cosine 衰減、weight decay、dropout 與增強分別改變最佳化或正則化，但是否採用完全依 ablation 的 validation 結果。這是依序的貪婪比較，後續對照是在前面選定設定上進行，不能解讀為所有改動互相獨立的因果效果，也沒有涵蓋全部交互作用。

五 seed validation-selected checkpoint 的 test accuracy 為 84.7222 ± 0.8045%，全資料固定輪數重訓為 82.9630 ± 1.9278%；± 表示五次訓練的樣本標準差（ddof=1），不是信賴區間。全資料訓練輪數 78 為五個 validation 最佳輪次 [243, 49, 70, 214, 78] 的中位數，各模型只取最後一輪。這種作法使用更多訓練資料，但不保證原切分的最佳輪數能直接轉移至全資料訓練。兩種五模型等權 softmax ensemble 的 test accuracy 分別為 87.7778% 與 86.0185%；主要成果在評估前已指定為後者，未達到 87% 目標。Ensemble 可平均部分初始化與訓練噪聲，但五個模型共享資料，其誤差並非獨立，改善幅度應以實測為準。

這些 seeds 共用同一個 validation 切分，因此標準差只反映訓練隨機性，不涵蓋重新切分資料或跨受試者的不確定性。多次搜尋也可能對固定 validation 過度擬合。舊版 test 85.28% 已在此前作業中被檢視，本次不能宣稱是完全未曾接觸的全新盲測；但新流程沒有以 test 挑選 checkpoint、設定、輪數或追加實驗。Ablation 的 test 數值僅在最後統一補列，不參與搜尋，沒有使用逐輪 test 峰值。

預先列入報告的 validation-selected ensemble 為 87.7778%，達到 87%；它屬於預先固定的另一種評估方法，不能事後取代主要成果。

進一步看逐項比較，加入增強後平均 validation accuracy 增加 0.6173 個百分點，最終補列的 test 平均增加 0.1852 個百分點。時間位移讓模型接觸不同時間對齊，振幅縮放與噪聲增加訊號擾動，可能降低對特定波形細節的依賴；但這三項是共同開啟，本實驗無法分辨哪一項貢獻最大。150 與 300 epochs 的三 seed 平均 validation accuracy 差為 0.0000 個百分點，因此 300 epochs 是依較低的 validation loss 勝出，不能描述成 validation accuracy 提升。正規化、較低 weight decay、較低 learning rate、cosine 與 AdamW 在本次逐步比較中均未獲選；它們不是普遍無效，只是目前樣本、固定切分與搜尋路徑下沒有得到較佳驗證結果。
