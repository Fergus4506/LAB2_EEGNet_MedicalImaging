"""Build tables, plots and a Chinese discussion from frozen V2 results."""
import csv
import json
from pathlib import Path

from main import ROOT, plt


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows[:1]) + "\n" + \
        "|" + "---|" * len(rows[0]) + "\n" + \
        "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows[1:])


def main():
    root = ROOT / "results_v2"
    summary = load(root / "final_summary.json")
    frozen = summary["frozen_selection"]
    config = frozen["config"]
    candidates = load(root / "candidates.json")
    stages = list(csv.DictReader((root / "ablation_stages.csv").open(encoding="utf-8-sig")))
    rows = list(csv.DictReader((root / "comparison.csv").open(encoding="utf-8-sig")))
    v, f = summary["validation_selected"], summary["full_train"]
    def score(group):
        return f'{group["selected_test_acc_mean"]:.4f} ± {group["selected_test_acc_std"]:.4f}'
    results_table = table([
        ["預先固定的報告方法", "Test accuracy (%)", "選擇依據"],
        ["864 筆訓練，5 seeds 個別 checkpoint", score(v), "各 seed 的 validation accuracy，再比較 loss"],
        ["864 筆訓練，5 seeds ensemble", f'{v["ensemble"]["test_acc"]:.4f}', "五個 checkpoint 等權 softmax 平均"],
        ["1,080 筆重訓，5 seeds 個別模型", score(f), f'固定 {frozen["full_train_epochs"]} epochs，最後一輪'],
        ["1,080 筆重訓，5 seeds ensemble（主要成果）", f'{f["ensemble"]["test_acc"]:.4f}', "預先指定，五個全資料模型等權平均"],
    ])
    seed_table = table([["Seed", "Validation-selected epoch", "Validation (%)", "Selected test (%)", "Full-train test (%)"]] + [
        [seed, next(r["best_epoch"] for r in rows if r["candidate"] == frozen["candidate"] and int(r["seed"]) == seed),
         f'{next(float(r["val_acc"]) for r in rows if r["candidate"] == frozen["candidate"] and int(r["seed"]) == seed):.4f}',
         f'{next(float(r["selected_test_acc"]) for r in rows if r["candidate"] == frozen["candidate"] and int(r["seed"]) == seed):.4f}',
         f'{next(float(r["selected_test_acc"]) for r in rows if r["candidate"] == "full_train" and int(r["seed"]) == seed):.4f}']
        for seed in frozen["seeds"]])
    stage_table = table([["階段", "Parent → candidate", "採用", "Validation mean ± SD (%)", "Δ val (pp)", "Test mean ± SD (%)", "Δ test (pp)"]] + [
        [r["stage"], f'{r["parent"]} → {r["candidate"]}', r["selected"],
         f'{float(r["validation_mean"]):.4f} ± {float(r["validation_std"]):.4f}', f'{float(r["validation_delta_pp"]):+.4f}',
         f'{float(r["test_mean"]):.4f} ± {float(r["test_std"]):.4f}', f'{float(r["test_delta_pp"]):+.4f}'] for r in stages])
    configs = table([["候選", "Separable", "WD", "LR / scheduler", "Normalize", "Augment", "Dropout", "Activation / α", "Optimizer", "Epochs"]] + [
        [c["candidate"], c["config"]["separable"], c["config"]["weight_decay"], f'{c["config"]["lr"]} / {c["config"]["scheduler"]}',
         c["config"]["normalize"], c["config"]["augment"], c["config"]["dropout"],
         f'{c["config"]["activation"]} / {c["config"]["alpha"]}', c["config"]["optimizer"], c["config"]["num_epochs"]] for c in candidates])
    conv = [c for c in candidates if c["candidate"] in ("c00", "c01")]
    difference = conv[1]["aggregate"]["val_acc_mean"] - conv[0]["aggregate"]["val_acc_mean"]
    conclusion = "達到" if f["ensemble"]["test_acc"] >= 87 else "未達到"
    secondary_note = (f'預先列入報告的 validation-selected ensemble 為 {v["ensemble"]["test_acc"]:.4f}%，'
                      f'{"達到" if v["ensemble"]["test_acc"] >= 87 else "未達到"} 87%；'
                      '它屬於預先固定的另一種評估方法，不能事後取代主要成果。')
    discussion = f'''本次將資料切分固定為 seed 42、受試者與類別分層的 864／216 筆，僅改變模型初始化、batch 順序、dropout 與資料增強的訓練 seed。每個候選使用 seeds 42、43、44，比較各自 validation 選定 checkpoint 的平均 accuracy，同分時比較平均 loss。共測試 {len(candidates)} 個不同設定，最後固定 {frozen["candidate"]}，再補 seeds 45、46；沒有根據新增兩個 seed 的結果重選設定。所有 test 評估均在設定、五個 seeds、重訓輪數與 ensemble 規則凍結後進行。

將後段 depthwise＋pointwise 改為一般時間卷積後，該卷積區塊的權重由 32×15＋32×32＝1,504 增為 32×32×15＝15,360，增加 13,856 個參數；全模型因此由 4,018 增為 17,874，約 4.45 倍。一般卷積可為每一對輸入／輸出特徵學習獨立的時間濾波器；depthwise＋pointwise 則先逐特徵濾波，再做跨特徵線性混合，因此具有較強的結構限制。此處的 full 是投影片式的一般卷積，並非嚴格意義的 separable convolution。初始三 seed 比較中，full 相對 depthwise 的平均 validation accuracy 改變 {difference:+.4f} 個百分點，這是此資料與設定下的實證結果；不能僅憑參數較多便推論一定較佳。

最終設定為 {config["separable"]}、{config["activation"]}（α={config["alpha"]}）、dropout={config["dropout"]}、{config["optimizer"]}、lr={config["lr"]}、weight decay={config["weight_decay"]}、scheduler={config["scheduler"]}、normalize={config["normalize"]}、augment={config["augment"]}，搜尋訓練上限 {config["num_epochs"]} epochs。正規化只從實際訓練資料估計每通道 mean/std；增強只作用於訓練 batch，使用各 trial／channel 的時間標準差設定噪聲。較小 learning rate、cosine 衰減、weight decay、dropout 與增強分別改變最佳化或正則化，但是否採用完全依 ablation 的 validation 結果。這是依序的貪婪比較，後續對照是在前面選定設定上進行，不能解讀為所有改動互相獨立的因果效果，也沒有涵蓋全部交互作用。

五 seed validation-selected checkpoint 的 test accuracy 為 {score(v)}%，全資料固定輪數重訓為 {score(f)}%；± 表示五次訓練的樣本標準差（ddof=1），不是信賴區間。全資料訓練輪數 {frozen["full_train_epochs"]} 為五個 validation 最佳輪次 {frozen["validation_epochs"]} 的中位數，各模型只取最後一輪。這種作法使用更多訓練資料，但不保證原切分的最佳輪數能直接轉移至全資料訓練。兩種五模型等權 softmax ensemble 的 test accuracy 分別為 {v["ensemble"]["test_acc"]:.4f}% 與 {f["ensemble"]["test_acc"]:.4f}%；主要成果在評估前已指定為後者，{conclusion} 87% 目標。Ensemble 可平均部分初始化與訓練噪聲，但五個模型共享資料，其誤差並非獨立，改善幅度應以實測為準。

這些 seeds 共用同一個 validation 切分，因此標準差只反映訓練隨機性，不涵蓋重新切分資料或跨受試者的不確定性。多次搜尋也可能對固定 validation 過度擬合。舊版 test 85.28% 已在此前作業中被檢視，本次不能宣稱是完全未曾接觸的全新盲測；但新流程沒有以 test 挑選 checkpoint、設定、輪數或追加實驗。Ablation 的 test 數值僅在最後統一補列，不參與搜尋，沒有使用逐輪 test 峰值。'''
    discussion += "\n\n" + secondary_note
    augmentation = next(r for r in stages if r["stage"] == "augment" and r["selected"] == "True")
    epoch_stage = next(r for r in stages if r["stage"] == "epochs" and r["selected"] == "True")
    discussion += (f'\n\n進一步看逐項比較，加入增強後平均 validation accuracy 增加 '
                   f'{float(augmentation["validation_delta_pp"]):.4f} 個百分點，最終補列的 test 平均增加 '
                   f'{float(augmentation["test_delta_pp"]):.4f} 個百分點。時間位移讓模型接觸不同時間對齊，'
                   '振幅縮放與噪聲增加訊號擾動，可能降低對特定波形細節的依賴；但這三項是共同開啟，'
                   '本實驗無法分辨哪一項貢獻最大。150 與 300 epochs 的三 seed 平均 validation accuracy '
                   f'差為 {float(epoch_stage["validation_delta_pp"]):.4f} 個百分點，因此 300 epochs 是依較低的 '
                   'validation loss 勝出，不能描述成 validation accuracy 提升。正規化、較低 weight decay、'
                   '較低 learning rate、cosine 與 AdamW 在本次逐步比較中均未獲選；它們不是普遍無效，'
                   '只是目前樣本、固定切分與搜尋路徑下沒有得到較佳驗證結果。')
    (root / "discussion_zh.md").write_text(discussion + "\n", encoding="utf-8")
    report = f'''# EEGNet V2：validation 選模與最終評估

## 固定流程

資料切分 seed 42；搜尋 seeds 42／43／44；最佳設定追加 45／46。所有候選皆依 validation 選權重，依三 seed 平均 validation 選設定。全資料固定輪數與主要 ensemble 成果於 test 評估前寫入 frozen_selection.json。Test 沒有逐輪曲線，僅有最後評估。原 results/ 以 SHA-256 核對未修改。

## 最終成果

{results_table}

{seed_table}

## 候選設定

{configs}

## 依序 ablation

以下每個對照使用相同三個 seeds，Δ 單位為百分點。Test 僅在所有選擇凍結後補列；不是 test 排行榜。候選若與先前完全相同即重用，避免重跑。

{stage_table}

## Discussion（可直接貼入報告）

{discussion}
'''
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    values = [v["selected_test_acc_mean"], v["ensemble"]["test_acc"], f["selected_test_acc_mean"], f["ensemble"]["test_acc"]]
    ax.bar(range(4), values)
    ax.errorbar([0, 2], [values[0], values[2]], yerr=[v["selected_test_acc_std"], f["selected_test_acc_std"]],
                fmt="none", color="black", capsize=4)
    ax.set_xticks(range(4), ["Validation-selected\n5-seed mean", "Validation-selected\nensemble", "Full train\n5-seed mean", "Full train\nensemble (primary)"])
    ax.set(ylabel="Final test accuracy (%)", ylim=(0, 100))
    ax.axhline(87, color="red", linestyle="--", label="87% target")
    for i, value in enumerate(values):
        ax.text(i, value + 3, f"{value:.2f}%", ha="center")
    ax.legend(loc="lower right")
    fig.savefig(root / "final_test_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ablations = list(csv.DictReader((root / "ablation.csv").open(encoding="utf-8-sig")))[:-1]
    for metric, label, offset in (("val_acc", "Validation (selection)", -0.12), ("selected_test_acc", "Final test (reporting only)", 0.12)):
        ax.errorbar([i + offset for i in range(len(ablations))], [float(r[metric + "_mean"]) for r in ablations],
                    yerr=[float(r[metric + "_std"]) for r in ablations], fmt="o", capsize=3, label=label)
    ax.set_xticks(range(len(ablations)), [r["candidate"] for r in ablations], rotation=45)
    ax.set(ylabel="Accuracy (%), mean ± sample SD", xlabel="Candidate (same three seeds)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(root / "ablation_comparison.png", dpi=180)
    plt.close(fig)
    marker = "## V2：固定 validation、多 seed 與最終 test 評估"
    readme = ROOT.parent / "README.md"
    existing = readme.read_text(encoding="utf-8").split(marker)[0].rstrip()
    commands = f'''```powershell
.\\.venv\\Scripts\\python.exe lab2_EEG_classification/selfcheck.py
# 完整重現請使用尚不存在的新目錄，避免覆寫成果
.\\.venv\\Scripts\\python.exe lab2_EEG_classification/experiments_v2.py --phase search --output lab2_EEG_classification/results_v2_reproduce
.\\.venv\\Scripts\\python.exe lab2_EEG_classification/experiments_v2.py --phase final --output lab2_EEG_classification/results_v2_reproduce
# 單組三 seed validation-only ablation
.\\.venv\\Scripts\\python.exe lab2_EEG_classification/main.py --activation relu --separable full --weight-decay 0.001 --optimizer adam --scheduler cosine --lr 0.001 --normalize --augment --seeds 42 43 44 --validation-only --output lab2_EEG_classification/results_custom_v2
# 已凍結最佳設定的五 seed 全資料重訓與 ensemble
.\\.venv\\Scripts\\python.exe lab2_EEG_classification/main.py --activation {config["activation"]} --alpha {config["alpha"]} --dropout {config["dropout"]} --separable {config["separable"]} --weight-decay {config["weight_decay"]} --optimizer {config["optimizer"]} --scheduler {config["scheduler"]} --lr {config["lr"]}{" --normalize" if config["normalize"] else ""}{" --augment" if config["augment"] else ""} --num-epochs {frozen["full_train_epochs"]} --full-train --seeds 42 43 44 45 46 --ensemble --output lab2_EEG_classification/results_final_reproduce
# 重現 validation-selected 五 seed ensemble（本次 87.7778%）
.\\.venv\\Scripts\\python.exe lab2_EEG_classification/main.py --activation {config["activation"]} --alpha {config["alpha"]} --dropout {config["dropout"]} --separable {config["separable"]} --weight-decay {config["weight_decay"]} --optimizer {config["optimizer"]} --scheduler {config["scheduler"]} --lr {config["lr"]}{" --normalize" if config["normalize"] else ""}{" --augment" if config["augment"] else ""} --num-epochs {config["num_epochs"]} --seeds 42 43 44 45 46 --ensemble --output lab2_EEG_classification/results_selected_reproduce
```'''
    section = f'''{marker}

本節為 V2 最新流程；前文八組設定與逐輪 test 曲線是保留的 V1 歷史成果。V2 不產生逐輪 test 指標，所有訓練曲線只有 train／validation；全資料重訓只有 train。預設輸出改為 results_v2，既有目錄與 comparison.csv 均拒絕覆寫。原 results/ 全部檔案雜湊一致。

{results_table}

共 {len(candidates)} 個候選，每個至少三 seeds；最佳設定 {frozen["candidate"]} 再跑至五 seeds。固定 split seed 42；訓練 seed 控制初始化、shuffle、dropout 與專用增強 generator。平均值與樣本 SD（ddof=1）不得解讀為跨切分的信賴區間。主要成果 {conclusion} 87%。

{commands}

所有新模型選項預設保持原超參數：depthwise、Adam、weight decay 0.001、scheduler none、不 normalize、不 augment。--split-seed 預設 42，與 --seed／--seeds 分開。--full-train 只保存固定輪數的最後權重，summary 的 selection=fixed_epochs_full_train；輪數必須先由 validation 決定。--validation-only 完全不載入 test 樣本；一般單組模式只在訓練完成後評估一次。--ensemble 需搭配 --seeds，以各模型自己的訓練統計量前處理後平均 softmax。

normalize 使用訓練子集每通道的平均與母體標準差（跨 trial 與時間），std 下限 1e-8；全資料訓練改用全部 1,080 筆重新估計。NaN 仍先由實際訓練資料的全域平均補值。增強依序為每筆 circular shift ±50、振幅 ×0.9–1.1、加入 σ=0.05×原 trial／channel 時間標準差的 Gaussian noise，僅訓練 batch 啟用。統計量、增強規則與 seed 都記錄於 config.json。

完整結果與可貼入報告的分析：[REPORT.md](lab2_EEG_classification/results_v2/REPORT.md)、[Discussion](lab2_EEG_classification/results_v2/discussion_zh.md)。逐 seed 結果為 comparison.csv；ablation.csv 列設定與三 seed 統計；ablation_stages.csv 列各階段相對 parent 的差值。每組保留 config、split、history、best.pt、重載驗證、summary、acc/loss 圖；final_summary.json 包含兩種五 seed mean±SD 及 ensemble。protocol.json、decisions.json、frozen_selection.json、final_evaluation_started.json 記錄 test 接觸前的選擇依據。

{discussion}

V2 整合報告：output/docx/V2/LAB2_315553010_楊敦傑.docx 與 output/pdf/V2/LAB2_315553010_楊敦傑.pdf，共 15 頁，已於 2026/10/28 完成內容與排版核對。舊版保留，提交請使用 V2 目錄版本。舊程式 ZIP 未更新，沒有 commit 或 push。
'''
    readme.write_text(existing + "\n\n" + section, encoding="utf-8")
    criteria = ROOT.parent / "COMPLETION_CRITERIA.md"
    content = criteria.read_text(encoding="utf-8").split("\n## V2 效能目標與驗收")[0].rstrip()
    old = "- [ ] Test accuracy 達 87%：目前最高 **85.2778%**，未達目標。"
    content = content.replace(old, f'- [{"x" if v["ensemble"]["test_acc"] >= 87 else " "}] V2 預先規劃的 validation-selected 五 seed ensemble：**{v["ensemble"]["test_acc"]:.4f}%**。\n'
                              f'- [{"x" if f["ensemble"]["test_acc"] >= 87 else " "}] V2 預先指定主要成果（全資料五 seed ensemble）≥87%：**{f["ensemble"]["test_acc"]:.4f}%**，{conclusion}目標。V1 的 85.2778% 保留為歷史基準。')
    content += f'''\n## V2 效能目標與驗收（2026/10/28）

- [x] A–F 選項實作，舊權重相容；固定 split seed 42，訓練 seeds 分離。
- [x] selfcheck PASS：兩種卷積、梯度、平均 loss、增強可重現、train-only 正規化、重載、全資料最後權重與防覆寫。
- [x] {len(candidates)} 個候選至少三 seeds，最佳設定五 seeds，全資料五 seeds；搜尋只依 validation。
- [x] test 前已凍結設定、{frozen["full_train_epochs"]} 輪重訓與主要成果；無 test 峰值選擇。
- [x] comparison／ablation CSV、逐組曲線、最終圖表與中文 Discussion 完成；舊 results/ 雜湊未變。
- [x] 兩種五 seed mean±SD 與 softmax ensemble 已完整報告；± 為樣本 SD。
- [x] V2 已整合至 output/docx/V2/LAB2_315553010_楊敦傑.docx 與 output/pdf/V2/LAB2_315553010_楊敦傑.pdf；15 頁於 2026/10/28 經 Word 匯出及逐頁視覺檢查，核心數值與原始紀錄一致。

{results_table}

V2 結果以 results_v2/final_summary.json 為準。原先的「不追加搜尋」敘述是 V1 歷史紀錄；本次依使用者新指示追加 validation-only 搜尋，沒有依新 test 結果繼續調參。未 commit、push 或提交 E3。
'''
    criteria.write_text(content, encoding="utf-8")
    print(results_table)


if __name__ == "__main__":
    main()
