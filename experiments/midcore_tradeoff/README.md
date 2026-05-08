# Mid-Core Trade-off Experiments

研究 ChestMNIST 多標籤分類中 **mid core 標籤** `[5, 4, 7, 8, 12]` 表現不佳的成因。

## 假設

> Mid core 表現由兩個對立因素決定：
> - **Coverage**（label set 涵蓋廣度）：提供 label 相關性資訊 → 正向
> - **Imbalance**（樣本分佈不均）：loss 被 dominant label 主導 → 負向
>
> 存在最佳涵蓋範圍，太窄（資訊不足）和太寬（imbalance 失控）都不好。

## 實驗清單

| Script | 用途 |
|---|---|
| [run_midcore_highocc_group_tradeoff.sh](run_midcore_highocc_group_tradeoff.sh) | Greedy 擴張 label set，觀察 trade-off 曲線是否存在 |
| [run_midcore_coverage_imbalance_tradeoff.sh](run_midcore_coverage_imbalance_tradeoff.sh) | 解耦 coverage 與 imbalance，分別量化兩者貢獻 |
| [plot_midcore_tradeoff.py](plot_midcore_tradeoff.py) | 將兩個實驗結果繪成 trade-off 圖 |

所有 script 都會自動 `cd` 回 repo root，可從任何位置執行。

---

## Experiment 1：Greedy Group Expansion（觀察 trade-off）

`run_midcore_highocc_group_tradeoff.sh`

**設計**：
- 從 mid core `[5, 4, 7, 8, 12]` 出發（`label_head=5, label_tail=12`）
- 依排序序列 `[3, 2, 0, 5, 4, 7, 8, 12, 1, 10, 9, 11, 6, 13]` 進行 greedy
- 每一步比較左/右鄰居的 normalized co-occurrence gain，選 gain 較大的一側
- 重複到覆蓋全部 14 個標籤（共 10 步：S0..S9）

**輸出**：
- `results/default/chestmnist/MedViT_tiny/class_{head}_to_{tail}_sample_10pct/`
- `results/tradeoff_plots/midcore_greedy_group_summary.csv`
- `results/tradeoff_plots/midcore_tradeoff_group.{csv,png}`

**讀圖**：x = MeanIR，y = co-occurrence info（additive），顏色 = mid-core AUC，曲線連接 S0→S9

---

## Experiment 2：Coverage × Imbalance Trade-off（解耦變因，Experiment A）

`run_midcore_coverage_imbalance_tradeoff.sh`

**設計**：
- 三個 coverage 設定（從 group 實驗挑出的代表點）：
  - **S0**：`label_head=5, label_tail=12`（5 labels，core only）
  - **S7**：`label_head=3, label_tail=11`（12 labels，best trade-off）
  - **S9**：`label_head=3, label_tail=13`（14 labels，full）
- 對每種 coverage 掃描 `class_balanced_alpha`：`0.0, 0.5, 1.0`
  - `0.0`：natural 分佈（高 imbalance）
  - `1.0`：完全 inverse-frequency 重抽樣（低 imbalance）
- 共 3 × 3 = 9 個訓練組合

**對應 main.py 參數**：
- `--label_head {5|3} --label_tail {12|11|13}`
- `--sample 0.1`
- `--class_balanced_sampling True`
- `--class_balanced_alpha {0.0|0.5|1.0}`

**輸出**：
- `results/default/chestmnist/MedViT_tiny/class_{H}_to_{T}_sample_10pct_clsbal_a{alpha}/`
- `results/tradeoff_plots/midcore_tradeoff_coverage_imbalance.{csv,png}`
  - x = `alpha_balance`，y = mid-core AUC，每條線一個 coverage 等級

**支持假設的預期結果**：
- S9 + balance > S9 + natural → 寬 coverage 的紅利被 imbalance 抵銷
- S0 + balance ≈ S0 + natural → 窄 coverage 的瓶頸是資訊不足，不是 imbalance
- Balanced 曲線變單調（S0 < S7 ≤ S9）→ group 觀察到的「先升後降」就是 imbalance 造成的

---

## 繪圖：plot_midcore_tradeoff.py

模式：
- `--mode group`：讀取 group 擴張結果
- `--mode coverage_imbalance`：讀取 coverage × imbalance 結果
- `--mode all`：兩者都畫

通常不需要手動執行，兩個 .sh 結尾會自動呼叫對應模式。

---

## 跑法

```bash
# 從 repo root 跑（任何位置都行，scripts 會自動 cd）
./experiments/midcore_tradeoff/run_midcore_highocc_group_tradeoff.sh
./experiments/midcore_tradeoff/run_midcore_coverage_imbalance_tradeoff.sh
```

需要 `.experiment.env`（在 repo root）設定 `VENV_ACTIVATE` 指向 venv。
