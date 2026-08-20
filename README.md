# FedPerGC-Framework

A Federated Learning framework combining **FedPer** with **G**radient **C**ontri-bution (FedPerGC) for rice leaf disease image classification across non-IID clients.

## Repository contents

| File | Description |
|---|---|
| [`FedPerGC.ipynb`](FedPerGC.ipynb) | Original research notebook (exploratory, Colab-based). |
| [`FedPerGC_improve.ipynb`](FedPerGC_improve.ipynb) | Refactored, production-oriented version of the same experiment: deduplicated cells, centralized configuration, and code organized into small, documented, reusable functions. Recommended entry point to understand and reproduce the pipeline. |

## `FedPerGC_improve.ipynb` overview

The notebook is organized into 14 sections that mirror the FedPerGC pipeline end-to-end:

1. **Environment setup** — GPU availability check, optional Google Drive mount (Colab).
2. **Configuration** — `FLConfig` dataclass (paths, image size, batch size, local epochs, number of rounds, L2 regularization) and the `CLIENTS_INFO` client registry.
3. **Model architecture** — CBAM-style attention residual backbone (`build_efficient_fl_network`) used by both the global model and every client model.
4. **Data pipeline** — per-client `ImageDataGenerator` flows (`load_data`) and dataset inspection (`count_images_by_class`).
5. **Training utilities** — epoch timing callback, balanced class weights, learning-rate schedule.
6. **System resource monitoring** — CPU/RAM/GPU/disk usage logging per round (`log_system_usage`).
7. **FL analytics utilities** — JSON persistence helpers, communication-overhead estimation, gradient-based client contribution scoring, weight padding/delta helpers.
8. **Federated round core logic** — the FedPerGC training step, split into: local weight synchronization, per-client local training, contribution-weighted aggregation, and global model evaluation.
9. **Federated learning orchestration** — `run_federated_learning`, the main round loop.
10. **History persistence** — save/load global and per-client training history.
11. **Model loading & evaluation** — unified `evaluate_model_on_dir` (confusion matrix, ROC curves, classification report, inference timing), reused for single-round, multi-round, and per-client evaluation.
12. **Benchmarking** — inference throughput/latency/memory benchmarking (`benchmark_model_with_format`).
13. **Visualization** — training curves, contribution scores, gradient similarity, resource usage plots.
14. **Main execution** — the runnable pipeline: data exploration → run FL → visualize → evaluate → benchmark.

## FedPerGC algorithm

FedPerGC combines **FedAvg-style delta aggregation**, **personalization** (per-client local model persistence), and **gradient-based contribution weighting** to mitigate the effect of non-IID client data during aggregation.

At each communication round `t`:

1. **Broadcast** — every client loads its own persisted local model (or a fresh copy of the global model on the first round) and synchronizes all layers except the classification head with the current global model weights.
2. **Local training** — each client trains locally for `epochs_local` epochs on its own data, using per-client balanced class weights to counter local class imbalance.
3. **Update capture** — for each client, compute:
   - the weight delta `Δw = w_after − w_before` (used for FedAvg-style aggregation), and
   - the full post-training weights excluding the classification head (used for FedPer-style aggregation).
4. **Gradient similarity** — compute the pairwise cosine similarity of client weight deltas to monitor client drift/heterogeneity across rounds.
5. **Contribution scoring** — score each client's contribution from the magnitude of its weight update scaled by its local sample count, then normalize scores across active clients to obtain aggregation weights (falling back to sample-count weighting if all contributions are zero).
6. **Aggregation** — apply the contribution-weighted sum of client deltas to the global model weights (shape-padding is applied when client/global layer shapes differ, e.g. a different number of output classes).
7. **Evaluation & logging** — evaluate the updated global model on the shared validation set, log system resource usage, and persist round metrics, contribution scores, and gradient similarities to disk.
8. **Checkpointing** — save the global model for the round and repeat until `rounds` communication rounds are completed.

## Step-by-step reproduction

1. **Environment**: Python 3.10+, TensorFlow 2.x, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `psutil`, `GPUtil`. A GPU runtime is strongly recommended (the notebook aborts early if no GPU is detected).
2. **Data layout**: prepare one folder per client under a common root, each containing `train/`, `val/` (and optionally `test/`) subfolders, themselves split into one subfolder per class (e.g. `Brown_Spot`, `Leaf_Blast`, `Leaf_Blight`, `Normal`). Prepare a similarly structured global `valid/` (and `test/`) set for centralized evaluation.
3. **Configure**: open [`FedPerGC_improve.ipynb`](FedPerGC_improve.ipynb) and edit the `FLConfig` fields and `CLIENTS_INFO` paths (Section 2) to point to your dataset locations, then set `img_size`, `batch_size`, `epochs_local`, and `rounds` as needed.
4. **Build/load the base model**: run Sections 1–4 to set up the model architecture and data pipeline. Point `initial_model_path` (Section 14.2) to a pretrained checkpoint, or let the notebook build a fresh `build_efficient_fl_network` model if none is found.
5. **Run federated training**: execute Section 14.2 to call `run_federated_learning`, which loops for `config.rounds` rounds, persisting a global model checkpoint, contribution scores, gradient similarities, and resource logs after every round.
6. **Inspect training dynamics**: run Section 14.3 to plot global/per-client accuracy and loss curves, contribution scores, gradient similarity, and resource usage.
7. **Evaluate**: run Section 14.4 to evaluate the final (or any intermediate round's) global model on the shared test set and on each client's own held-out test set — producing classification reports, confusion matrices, and ROC curves.
8. **Benchmark**: run Section 14.5 to measure inference throughput, latency, and memory footprint across batch sizes.

## Citation / attribution note

The FedPerGC algorithm and this framework were developed as part of the master's thesis of **Nguyễn Tiểu Phụng**, and were published in the peer-reviewed scientific article:

> DOI: [https://doi.org/10.1016/j.compag.2026.112249](https://doi.org/10.1016/j.compag.2026.112249)

If you use this code or the FedPerGC algorithm in your research, please cite the above publication.