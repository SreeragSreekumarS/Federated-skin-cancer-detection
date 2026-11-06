# Compact Federated Skin-Lesion Classifier

This project implements a privacy-preserving, non-IID federated learning workflow, redesigned for a laptop NVIDIA GTX 1650 (usually 4 GB VRAM). It uses a pretrained MobileNetV3-Large instead of ViT-B/16: 5.5 M parameters rather than about 86 M. A lightweight CBAM feature-attention head brings lesion-focused local and spatial attention without the memory cost of a Swin-plus-DenseNet hybrid. Client updates are trained one at a time, so GPU memory does not grow with the number of simulated hospitals.

## Important clinical note

This is a research tool, not a medical device. Do not use its output to diagnose, triage, or treat patients. Accuracy can only be established on a patient-level held-out test set that matches the intended clinical population; no implementation can honestly guarantee it exceeds every published result.

## What is improved over the reference paper

- Patient/lesion groups are assigned to one simulated hospital first, then split into train/validation/test within that hospital, preventing group leakage across partitions.
- Deterministic, class-dependent Dirichlet hospital assignment creates non-IID clients; class-balanced focal loss, pretrained transfer learning, and early stopping improve robust performance rather than training accuracy alone.
- FedProx reduces client drift. Server aggregation considers sample count and validation quality.
- Macro F1, balanced accuracy, per-class sensitivity/specificity, ROC-AUC, and a confusion matrix are reported, not only accuracy.
- Mixed precision, configurable input sizes (160 px for the GTX-1650 profile and 224 px for the full HAM10000/ISIC 2019 profiles), bounded GPU memory, sequential client simulation, adaptive learning-rate reduction, and one best checkpoint support the supplied hardware profiles.
- The optional CBAM head adapts the attention idea from multi-task lesion research to a classification-only dataset. A compact LayerNorm/dropout classifier head regularizes the CBAM-refined features without failing on small hospital batches. It does **not** claim segmentation, because segmentation masks are not part of this project's manifest.
- `tools/audit_manifest.py` identifies exact visual duplicates before combining datasets or splitting by patient, addressing the duplicate-image risk highlighted in transformer dataset research.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install the CUDA build of PyTorch suitable for your NVIDIA driver from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.
3. Prepare a CSV manifest with `image_path,label,patient_id` columns. Paths may be absolute or relative to the manifest. A true patient identifier is preferred; a lesion identifier groups images of the same lesion but does not guarantee that separate lesions from one patient stay together.
4. For HAM10000, use the helper below. For ISIC 2019, include its metadata file so available `lesion_id` values are used, then audit the manifest:

```powershell
python tools/prepare_ham10000.py --metadata C:\data\HAM10000_metadata.csv --images C:\data\HAM10000_images --output data\ham10000_manifest.csv
python tools/audit_manifest.py --manifest data\ham10000_manifest.csv --output runs\ham10000_duplicate_audit.csv
python tools/prepare_isic2019.py --ground-truth data\ISIC2019\ISIC_2019_Training_GroundTruth.csv --images data\ISIC2019\images\ISIC_2019_Training_Input --metadata data\ISIC2019\ISIC_2019_Training_Metadata.csv --output data\ISIC2019\isic2019_manifest.csv
python tools/audit_manifest.py --manifest data\ISIC2019\isic2019_manifest.csv --output runs\isic2019_duplicate_audit.csv
python train.py --manifest data\ham10000_manifest.csv --config config\gtx1650.yaml
```

The audit exits with status `1` when it finds duplicates. Review the CSV before training: remove accidental cross-dataset copies, resolve conflicting labels, and keep every duplicate group within one patient-level split. Near-duplicates are not automatically removed because doing so needs a clinical review.

The first run downloads ImageNet weights unless `--no-pretrained` is supplied. Training artifacts are kept in `runs/<name>/`: `best_model.pt`, `metrics.json`, `confusion_matrix.csv`, `hospital_metrics.json`, per-hospital confusion matrices in `hospital_metrics/`, class names, and `test_predictions.npz`. No raw images or per-client weights are copied there.

Each completed training run also creates `runs/<name>/plots/` with client training accuracy/loss by federated round, global validation curves, combined and per-hospital held-out confusion-matrix heatmaps, per-class ROC AUC, ROC curves, and precision-recall curves. ROC and precision-recall curves are omitted for classes that do not have both positive and negative examples in the evaluated split. Client training curves are computed during local optimization; global curves use validation data, and final confusion matrices/ROC plots use held-out test data.

Each patient/lesion group is assigned to one simulated hospital before that hospital's train, validation, and held-out test subsets are made. On HAM10000 with `dirichlet_alpha: 0.5`, rare classes such as DF and VASC may have very few examples at some hospitals; use combined test metrics as the primary result and interpret rare-class per-hospital metrics cautiously. `hospital_metrics.json` reports test metrics for every hospital, the combined test metrics, mean/standard deviation across hospitals, and worst-hospital macro F1. `metrics.json` also records local validation metrics for each selected hospital in every training round.

## Centralized baseline

Run the centralized comparison with the same manifest and config as the federated run:

```bash
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_gpu.yaml
```

It creates one deterministic, group-aware train/validation/test split over the full manifest and trains a single model on the full centralized training partition; it does not create simulated hospitals. The model, image transforms, loss, optimizer, and validation-based checkpointing use the matching config; FedProx and server aggregation do not apply centrally. Because the federated workflow assigns groups to hospitals before splitting within each hospital, its exact row assignments differ from this global centralized split. Results are saved separately under `runs/centralized/<run_name>/`, including `split_assignments.csv` so the rows can be audited.

Every metric report includes accuracy, balanced accuracy, micro/macro/weighted precision-recall-F1, one-vs-rest ROC-AUC, average precision, top-2 accuracy, Cohen's kappa, Matthews correlation coefficient, and log loss. For every class it includes precision (PPV), sensitivity/recall, specificity, negative predictive value, F1, ROC-AUC, average precision, false-positive/false-negative rates, TP/TN/FP/FN counts, and support. Confusion matrices are saved separately.

## Verify before using medical data

Run `python tools/check_setup.py`. For the GTX 1650 setup, confirm it identifies the GPU and reports `CUDA available: True`. On a CPU-only laptop, `CUDA available: False` is expected; continue with `python tools/smoke_test.py`. This creates temporary artificial images, performs one federated training round, checks evaluation and aggregation, prints `SMOKE TEST PASSED`, and leaves no data or model files behind.

## Optional lesion segmentation preprocessing

The ISIC 2018 Task 1 images and ground-truth masks can train an auxiliary U-Net. The segmenter can then predict lesion regions in HAM10000 and ISIC 2019 images and create padded lesion crops for a separate classifier run. It does not alter the original images or manifests. Segmentation can help focus the classifier, but accuracy gains are not guaranteed.

From the project root, train the segmenter on the extracted ISIC 2018 folders:

```bash
python -m tools.train_segmenter \
  --images data/ISIC2018_segmentation/images/ISIC2018_Task1-2_Training_Input \
  --masks data/ISIC2018_segmentation/masks/ISIC2018_Task1_Training_GroundTruth \
  --output runs/isic2018_segmenter/best_segmenter.pt \
  --epochs 40 --batch-size 8 --size 256
```

Apply the trained segmenter to each classification dataset, then train with the new manifest:

```bash
python -m tools.apply_segmenter \
  --manifest data/HAM10000/ham10000_manifest.csv \
  --checkpoint runs/isic2018_segmenter/best_segmenter.pt \
  --output-dir data/HAM10000/segmented_images \
  --output-manifest data/HAM10000/ham10000_segmented_manifest.csv
python train.py --manifest data/HAM10000/ham10000_segmented_manifest.csv --config config/ham10000_gpu.yaml

python -m tools.apply_segmenter \
  --manifest data/ISIC2019/isic2019_manifest.csv \
  --checkpoint runs/isic2018_segmenter/best_segmenter.pt \
  --output-dir data/ISIC2019/segmented_images \
  --output-manifest data/ISIC2019/isic2019_segmented_manifest.csv
python train.py --manifest data/ISIC2019/isic2019_segmented_manifest.csv --config config/isic2019_gpu.yaml
```

Keep the original manifests and baseline run outputs. Each classifier config uses the same fixed seed and split logic, so the original and segmented runs can be compared on the same held-out rows. Before claiming test performance, check for duplicate images between the ISIC 2018 segmenter training images and the HAM10000/ISIC 2019 test images; overlapping images can leak information into the evaluation. Review predicted crops from several images, especially rare or unusually large lesions, before relying on the segmented run.

## Full-image 224 px classifier runs

The HAM10000 and ISIC 2019 GPU configs use 224 px inputs. Classifier preprocessing preserves the complete original image and its proportions: it scales the image to fit a square and pads the remaining space, rather than cropping or stretching it. The configs also use new run names, so these higher-resolution runs do not replace earlier 160 px results. To run this full-image comparison, use each dataset's original manifest, not the optional `*_segmented_manifest.csv`:

```bash
python train.py --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_gpu.yaml
python train.py --manifest data/ISIC2019/isic2019_manifest.csv --config config/isic2019_gpu.yaml
```

If GPU memory allows and 224 px improves the held-out metrics, change `image_size` to `256` in the relevant config and use a distinct `run_name` for that comparison. Higher resolution can help retain small visual details, but it does not guarantee better classification scores.

## Classifier runs with segmentation

After `runs/isic2018_segmenter/best_segmenter.pt` has been created, first apply it to each dataset's original manifest. Then train using the new segmented manifest and the matching segmented config. These commands create a lesion-centered crop with padding, and the classifier resizes that crop to 224 px while preserving its proportions:

```bash
python -m tools.apply_segmenter \
  --manifest data/HAM10000/ham10000_manifest.csv \
  --checkpoint runs/isic2018_segmenter/best_segmenter.pt \
  --output-dir data/HAM10000/segmented_images \
  --output-manifest data/HAM10000/ham10000_segmented_manifest.csv
python train.py --manifest data/HAM10000/ham10000_segmented_manifest.csv --config config/ham10000_segmented_224.yaml

python -m tools.apply_segmenter \
  --manifest data/ISIC2019/isic2019_manifest.csv \
  --checkpoint runs/isic2018_segmenter/best_segmenter.pt \
  --output-dir data/ISIC2019/segmented_images \
  --output-manifest data/ISIC2019/isic2019_segmented_manifest.csv
python train.py --manifest data/ISIC2019/isic2019_segmented_manifest.csv --config config/isic2019_segmented_224.yaml
```

The segmented configs use separate run names and keep the original 224 px full-image configs available for comparison. Review some generated crops before training, and compare full-image and segmented results using the same split seed. Segmentation may improve focus, but an accuracy increase is not guaranteed.

## Tune safely

- If CUDA reports out-of-memory, set `batch_size: 8` and `image_size: 144`.
- Start with `rounds: 8` to verify data. The default 30 rounds is a realistic fine-tuning run, not a claim of clinical readiness.
- Use `clients: 3` to reproduce the paper's simulation. Federated simulation is not a substitute for real institutional deployment; raw data still stays in each configured local client in a real deployment.
- Set `use_cbam: false` if you need the original MobileNetV3 head for an ablation. For 4 GB VRAM, use `config/gtx1650.yaml` with its 160 px input; the 224 px HAM10000 and ISIC 2019 profiles may need more memory.
- Keep `strong_augmentation: false` for the baseline. Enable it only as a documented ablation: it adds low-probability blur, perspective, and erasing transforms, which may improve robustness but can also affect probability calibration.
- The global validation macro-F1 controls the learning-rate schedule. After `lr_patience` rounds without improvement, each client starts future rounds at `lr_factor` times the previous rate, down to `min_learning_rate`.
