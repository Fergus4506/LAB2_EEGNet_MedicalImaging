本次將資料切分固定為 seed 42、受試者與類別分層的 864／216 筆，僅改變模型初始化、batch 順序、dropout 與資料增強的訓練 seed。每個候選使用 seeds 42、43、44，比較各自 validation 選定 checkpoint 的平均 accuracy，同分時比較平均 loss。共測試 16 個不同設定，最後固定 c15，再補 seeds 45、46；沒有根據新增兩個 seed 的結果重選設定。所有 test 評估均在設定、五個 seeds、重訓輪數與 ensemble 規則凍結後進行。

將後段 depthwise＋pointwise 改為一般時間卷積後，該卷積區塊的權重由 32×15＋32×32＝1,504 增為 32×32×15＝15,360，增加 13,856 個參數；全模型因此由 4,018 增為 17,874，約 4.45 倍。一般卷積可為每一對輸入／輸出特徵學習獨立的時間濾波器；depthwise＋pointwise 則先逐特徵濾波，再做跨特徵線性混合，因此具有較強的結構限制。此處的 full 是投影片式的一般卷積，並非嚴格意義的 separable convolution。初始三 seed 比較中，full 相對 depthwise 的平均 validation accuracy 改變 +1.3889 個百分點，這是此資料與設定下的實證結果；不能僅憑參數較多便推論一定較佳。

最終設定為 full、relu（α=1.0）、dropout=0.25、adam、lr=0.01、weight decay=0.001、scheduler=none、normalize=False、augment=True，搜尋訓練上限 300 epochs。正規化只從實際訓練資料估計每通道 mean/std；增強只作用於訓練 batch，使用各 trial／channel 的時間標準差設定噪聲。較小 learning rate、cosine 衰減、weight decay、dropout 與增強分別改變最佳化或正則化，但是否採用完全依 ablation 的 validation 結果。這是依序的貪婪比較，後續對照是在前面選定設定上進行，不能解讀為所有改動互相獨立的因果效果，也沒有涵蓋全部交互作用。

五 seed validation-selected checkpoint 的 test accuracy 為 84.7222 ± 0.8045%，全資料固定輪數重訓為 82.9630 ± 1.9278%；± 表示五次訓練的樣本標準差（ddof=1），不是信賴區間。全資料訓練輪數 78 為五個 validation 最佳輪次 [243, 49, 70, 214, 78] 的中位數，各模型只取最後一輪。這種作法使用更多訓練資料，但不保證原切分的最佳輪數能直接轉移至全資料訓練。兩種五模型等權 softmax ensemble 的 test accuracy 分別為 87.7778% 與 86.0185%；主要成果在評估前已指定為後者，未達到 87% 目標。Ensemble 可平均部分初始化與訓練噪聲，但五個模型共享資料，其誤差並非獨立，改善幅度應以實測為準。

這些 seeds 共用同一個 validation 切分，因此標準差只反映訓練隨機性，不涵蓋重新切分資料或跨受試者的不確定性。多次搜尋也可能對固定 validation 過度擬合。舊版 test 85.28% 已在此前作業中被檢視，本次不能宣稱是完全未曾接觸的全新盲測；但新流程沒有以 test 挑選 checkpoint、設定、輪數或追加實驗。Ablation 的 test 數值僅在最後統一補列，不參與搜尋，沒有使用逐輪 test 峰值。

預先列入報告的 validation-selected ensemble 為 87.7778%，達到 87%；它屬於預先固定的另一種評估方法，不能事後取代主要成果。

進一步看逐項比較，加入增強後平均 validation accuracy 增加 0.6173 個百分點，最終補列的 test 平均增加 0.1852 個百分點。時間位移讓模型接觸不同時間對齊，振幅縮放與噪聲增加訊號擾動，可能降低對特定波形細節的依賴；但這三項是共同開啟，本實驗無法分辨哪一項貢獻最大。150 與 300 epochs 的三 seed 平均 validation accuracy 差為 0.0000 個百分點，因此 300 epochs 是依較低的 validation loss 勝出，不能描述成 validation accuracy 提升。正規化、較低 weight decay、較低 learning rate、cosine 與 AdamW 在本次逐步比較中均未獲選；它們不是普遍無效，只是目前樣本、固定切分與搜尋路徑下沒有得到較佳驗證結果。
