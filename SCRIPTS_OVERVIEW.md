# MedViTV2 Scripts Overview

## Shell Scripts

### `.experiment.env` — 共用實驗環境設定

**目的**：集中管理 shell script 執行時要啟用的 Python 虛擬環境路徑。

**目前欄位**：
- `VENV_ACTIVATE`：venv activate script 路徑（可相對或絕對路徑）

兩個 mid-core 實驗腳本會先讀取 `./.experiment.env`，再執行：
- `source "$VENV_ACTIVATE"`

若沒有此檔或路徑不存在，腳本會 fallback 到 `../LABenv/bin/activate`；若仍不存在，則使用目前 shell 的 Python 環境繼續執行。

---

### `run_all_experiment.sh` — 損失函數比較實驗

**目的**：評估不同損失函數對同一模型的訓練效果，找出最適合長尾分布的損失函數。

**設定**：
- 模型：`MedViT_tiny`
- 資料集：`chestmnist`
- Epochs：50，取樣比例：10%

**執行的實驗**：

| 損失函數 | 搭配 Sampler |
|---|---|
| ASL | ✗ |
| BCE | ✗ |
| CBLoss | ✗ |
| CBLossOriginal | ✗ |
| LDACE_CCL | ✗ |
| Focal | ✗ |
| BCE | ✓ |
| DBFocal | ✓ |

共 8 次訓練，結果儲存在 `results/{LossName}/chestmnist/`。

---

### `run_group.sh` — 標籤分組策略比較實驗

**目的**：將 ChestMNIST 的 14 個標籤依照 imbalance ratio 分成若干群組，各群組獨立訓練一個模型，比較不同分組粗細度的效果。

**設定**：與 `run_all_experiment.sh` 相同（MedViT_tiny, chestmnist, 50 epochs, 10% 取樣）

**執行的三組實驗**：

| 實驗 | 分組門檻 (mean_ir / CVIR) | 分群數 | 訓練模型數 |
|---|---|---|---|
| Exp 1 | 1.5 / 0.2 | 6 群 | 6 |
| Exp 2 | 2.5 / 0.4 | 4 群 | 4 |
| Exp 3 | 3.5 / 0.6 | 3 群 | 3 |

門檻越高 → 分群越粗 → 每群包含更多標籤。每個實驗結束後呼叫 `aggregate_group_results.py` 彙整指標，結果儲存在 `results/group_evaluation/`。

---

### `run_group_expand_algo.sh` — 標籤範圍擴展演算法實驗

**目的**：以 Exp 2 的 4 群分組為基準，系統性地測試每個群組的 `label_head` / `label_tail` 邊界要如何延伸，才能達到最佳的核心標籤 AUC。

**核心群組定義（Exp 2）**：

| 群組 | 核心標籤 |
|---|---|
| Group 0 | [3, 2, 0] |
| Group 1 | [5, 4, 7, 8, 12, 1, 10, 9] |
| Group 2 | [11, 6] |
| Group 3 | [13] |

**實驗規模**：

| 群組 | head 組合數 | tail 組合數 | 實驗數 |
|---|---|---|---|
| Group 0 | 3 | 12 | 36 |
| Group 1 | 4 | 4 | 16 |
| Group 2 | 12 | 2 | 24 |
| Group 3 | 14 | 1 | 14 |
| **合計** | | | **90** |

最後呼叫 `evaluate_group_metrics.py` 與 `merge_group_expanding_algo_evaluations.py` 找出每個群組的最佳邊界配置，結果儲存在 `results/grouping_expanding_algo_evaluation/`。

---

### Mid-Core Trade-off Experiments

Mid core 標籤 `[5, 4, 7, 8, 12]` 的 coverage × imbalance trade-off 系列實驗，已獨立放在 [experiments/midcore_tradeoff/](experiments/midcore_tradeoff/README.md)，包含：

- `run_midcore_highocc_group_tradeoff.sh` — Greedy 擴張觀察 trade-off
- `run_midcore_coverage_imbalance_tradeoff.sh` — 解耦 coverage 與 imbalance（Experiment A）
- `plot_midcore_tradeoff.py` — 對應繪圖

詳情請見該目錄的 [README.md](experiments/midcore_tradeoff/README.md)。

---

## Python Scripts

### 核心訓練

#### `main.py`
訓練與評估的主程式。接受以下關鍵參數：
- `--model_name`：模型規格（`MedViT_tiny/small/base/large`）
- `--dataset`：目標資料集
- `--loss_function`：損失函數種類
- `--use_sampler`：是否啟用 Class-Aware Sampler
- `--label_head` / `--label_tail`：群組訓練時指定包含的標籤範圍
- `--sample`：取樣比例

#### `MedViT.py`
定義 MedViT 模型架構（tiny / small / base / large），為 MedViTV2 的核心 Vision Transformer，整合了 `NeighborhoodAttention2D` 與 FasterKAN 模組。

#### `fasterkan.py`
實作 FasterKAN（Kolmogorov-Arnold Network）層，作為 MedViT 中部分全連接層的替代，使用可學習的樣條基函數。

---

### 資料處理

#### `datasets.py`
建構資料集 DataLoader，支援從 MedMNIST 載入資料。包含 `build_dataset()` 函式與 `get_label_groups()` 函式（定義各資料集的標籤分群策略）。

#### `sampler.py`
實作 **Class-Aware Sampler**，針對多標籤不平衡資料集進行平衡取樣。核心類別：
- `RandomCycleIter`：無限循環迭代器，每輪結束後重新洗牌
- `ClassAwareSampler`：依類別頻率反比抽樣，抑制多數類、放大少數類

#### `longtail_splitter.py`
將資料集類別依樣本數分為 Head / Middle / Tail 三層：
- **Head**：≥ 7500 筆
- **Middle**：2500–7500 筆
- **Tail**：≤ 2500 筆

---

### 損失函數（`losses/`）

| 模組 | 損失函數 | 適用場景 |
|---|---|---|
| `bce_loss.py` | `BCELoss` | 多標籤基準線 |
| `cb_loss.py` | `CBLoss`, `CBLossOriginal` | Class-Balanced Loss，依有效樣本數加權 |
| `asl_loss.py` | `ASLoss`, `ASLSingleLabel` | Asymmetric Loss，對多標籤假負例懲罰非對稱 |
| `ldace_ccl_loss.py` | `LDACELoss`, `CCLLoss`, `LDACECCLLoss` | 建模標籤共現關係 + 校準誤差（ECE） |
| `focal_loss.py` | `FocalLoss` | 聚焦難分樣本 |
| `dbfocal_loss.py` | `DBFocalLoss`, `ResampleLoss` | Distribution-Balanced Focal Loss，專為長尾設計 |

---

### 分析與彙整

#### `label_grouper.py`
實作以動態規劃（DP）為基礎的 **DPLabelGrouper**，依照 imbalance ratio（IRlbl）與 CVIR（Coefficient of Variation of Imbalance Ratio）門檻自動將標籤分群。

#### `aggregate_group_results.py`
彙整多個群組模型的最佳 epoch 結果，計算 Head / Middle / Tail 各層的整體指標，輸出 per-class 與 tier summary CSV。由 `run_group.sh` 呼叫。

#### `evaluate_group_metrics.py`
從 `run_group_expand_algo.sh` 產生的大量訓練結果中，擷取每個 head/tail 組合對核心標籤的 AUC 指標，輸出到 `results/grouping_expanding_algo_evaluation/`。

#### `merge_group_expanding_algo_evaluations.py`
讀取各群組的最佳邊界配置，合併為包含全部 14 個類別的完整評估結果，並分析 Head / Middle / Tail 性能。

#### `analyze_single_result.py`
單一 CSV 分析工具，從指定的 `model_metrics.csv` 找出最佳 epoch，並計算 Head / Middle / Tail 指標。

```bash
python analyze_single_result.py --csv_path ./results/CBLoss/chestmnist/MedViT_tiny/sample_10pct/model_metrics.csv
```

#### `aggregate_auc_comparison.py`
從所有訓練方法（8 種損失函數配置 + 3 種分組策略 + 擴展演算法）彙整 per-class AUC，輸出比較表 `results/auc_comparison.csv`。

#### `visualize_ap_comparison.py`
產生各類別的 AP 比較圖，對比有無分組訓練的差異，圖表存於 `results/visualize/`。

---

## 實驗流程總覽

```
# Step 1：訓練不分組的 Baseline（產生 MedViT_tiny_chestmnist_metrics.csv）
python main.py --model_name MedViT_tiny --dataset chestmnist --sample 0.1 --epochs 50

# Step 2：比較損失函數（產生各損失函數的 per_class_metrics.csv）
run_all_experiment.sh

# Step 3：比較分組粗細度（產生 exp1/exp2/exp3 的 per_class CSV 與 group{0-3}_metrics.csv）
run_group.sh

        ↓ Step 1 + Step 3 (Exp 2) 完成後，可執行 ↓

# [visualize] 比較有無分組的 per-class AP 曲線圖（輸出到 results/visualize/）
python visualize_ap_comparison.py

# Step 4：搜尋最佳標籤邊界（90 種組合，產生 grouping_expanding_algo_evaluation/）
run_group_expand_algo.sh

# Step 5：Mid core 群組擴張 trade-off（middle core = [5,4,7,8,12]）
./experiments/midcore_tradeoff/run_midcore_highocc_group_tradeoff.sh

# Step 6：Coverage × Imbalance trade-off（Experiment A）
./experiments/midcore_tradeoff/run_midcore_coverage_imbalance_tradeoff.sh

        ↓ Step 2 + Step 3 + Step 4 全部完成後，可執行 ↓

# [visualize] 彙整所有方法的 per-class AUC 比較表（輸出 results/auc_comparison.csv）
python aggregate_auc_comparison.py
```

### 各 Visualize 腳本的前置需求

| 腳本 | 需先完成 | 輸出 |
|---|---|---|
| `visualize_ap_comparison.py` | Step 1（Baseline）+ Step 3 Exp 2（4 群組） | `results/visualize/class{n}.png` |
| `aggregate_auc_comparison.py` | Step 2 + Step 3 + Step 4 全部完成 | `results/auc_comparison.csv` |
| `analyze_single_result.py` | 任一訓練完成後即可單獨使用 | 終端機輸出 |
