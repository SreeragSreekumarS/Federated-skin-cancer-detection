# Compact Federated Skin-Lesion Classifier

This project implements a privacy-preserving, non-IID federated learning workflow, redesigned for a laptop NVIDIA GTX 1650 (usually 4 GB VRAM). It uses a pretrained MobileNetV3-Large instead of ViT-B/16: 5.5 M parameters rather than about 86 M. A lightweight CBAM feature-attention head brings lesion-focused local and spatial attention without the memory cost of a Swin-plus-DenseNet hybrid. Client updates are trained one at a time, so GPU memory does not grow with the number of simulated hospitals.

## Important clinical note

This is a research tool, not a medical device. Do not use its output to diagnose, triage, or treat patients. Accuracy can only be established on a patient-level held-out test set that matches the intended clinical population; no implementation can honestly guarantee it exceeds every published result.

## What is improved over the reference paper

- Patient/lesion groups are assigned to one simulated hospital first, then split into train/validation/test within that hospital, preventing group leakage across partitions.
- Deterministic, class-dependent Dirichlet hospital assignment creates non-IID clients; class-balanced focal loss, pretrained transfer learning, and early stopping improve robust performance rather than training accuracy alone.
- FedProx reduces client drift. Server aggregation considers sample count and validation quality.
- Macro F1, balanced accuracy, per-class sensitivity/specificity, ROC-AUC, and a confusion matrix are reported, not only accuracy.
- The supplied classifier profiles cover 288, 320, and 384 px inputs. Mixed precision, resized-image caching, sequential client simulation, adaptive learning-rate reduction, and gradient accumulation support the supplied hardware profiles.
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

Each patient/lesion group is assigned to one simulated hospital before that hospital's train, validation, and held-out test subsets are made. Even with `dirichlet_alpha: 1.0` on HAM10000, rare classes such as DF and VASC may have very few examples at some hospitals; use combined test metrics as the primary result and interpret rare-class per-hospital metrics cautiously. `hospital_metrics.json` reports test metrics for every hospital, the combined test metrics, mean/standard deviation across hospitals, and worst-hospital macro F1. `metrics.json` also records local validation metrics for each selected hospital in every training round.

## Centralized baseline

Run the centralized comparison with the same manifest and config as the federated run:

```bash
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_gpu.yaml
```

Centralized runs also accept `--seed`, `--run-name`, `--image-size`, and `--model-name`. `split_seed` stays fixed at 42 by default, so runs with different training seeds share the exact group-aware partitions. Temperature and per-class decision thresholds are fitted on validation predictions only; the held-out test set is used once for reporting.

To compare centralized MobileNetV3 with and without CBAM on the same data split, use a distinct run name:

```bash
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_gpu.yaml --no-cbam --run-name ham10000_mobilenetv3_no_cbam
```

The saved `config.yaml` records `use_cbam: false`; results are written to `runs/centralized/ham10000_mobilenetv3_no_cbam/`. Compare against the CBAM run using the same manifest, config, and split seed.

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

## Full-image resolution and MobileNet comparisons

The default HAM10000 and ISIC 2019 GPU configs use 288 px inputs. The 320 and 384 configs create and reuse square, pre-resized JPEG copies in `.resized_cache_<size>` beside the manifest. Preprocessing preserves the complete image by fitting and padding rather than cropping or stretching. Use the original manifest, not an optional segmented manifest:

```bash
python train.py --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_gpu.yaml
python train.py --manifest data/ISIC2019/isic2019_manifest.csv --config config/isic2019_gpu.yaml
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_320.yaml
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_384.yaml
```

For ISIC, use the matching `isic2019_320.yaml` and `isic2019_384.yaml`. Higher resolution can retain more detail but needs more GPU memory and does not guarantee better scores.

Recent `timm` releases provide MobileNetV4. Install requirements, then check the installed version and model name:

```bash
python -c "import timm; print(timm.__version__, 'mobilenetv4_conv_medium' in timm.list_models())"
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_v4_medium_320.yaml
```

For a 3–5 member ensemble, train separate centralized runs with distinct training seeds and run names while keeping `split_seed: 42`. Example:

```bash
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_320.yaml --seed 101 --run-name ham_320_seed101
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_320.yaml --seed 202 --run-name ham_320_seed202
python -m centralized_baseline.train --manifest data/HAM10000/ham10000_manifest.csv --config config/ham10000_320.yaml --seed 303 --run-name ham_320_seed303
python -m centralized_baseline.ensemble --manifest data/HAM10000/ham10000_manifest.csv --checkpoint runs/centralized/ham_320_seed101/best_model.pt --checkpoint runs/centralized/ham_320_seed202/best_model.pt --checkpoint runs/centralized/ham_320_seed303/best_model.pt --output runs/ham_320_ensemble
```

Ensemble members can use different supported model names or image sizes. The evaluator verifies that saved group splits and label order match, applies each run's configured 8-view dihedral TTA, averages logits, fits calibration on validation outputs, and reports test metrics. Compare individual and ensemble results; neither higher resolution nor ensembling guarantees improvement.

## Classifier runs with segmentation

After `runs/isic2018_segmenter/best_segmenter.pt` has been created, first apply it to each dataset's original manifest. Then train using the new segmented manifest and the matching segmented config. These commands create a lesion-centered crop with padding, and the classifier resizes that crop to 288 px while preserving its proportions:

```bash
python -m tools.apply_segmenter \
  --manifest data/HAM10000/ham10000_manifest.csv \
  --checkpoint runs/isic2018_segmenter/best_segmenter.pt \
  --output-dir data/HAM10000/segmented_images \
  --output-manifest data/HAM10000/ham10000_segmented_manifest.csv
python train.py --manifest data/HAM10000/ham10000_segmented_manifest.csv --config config/ham10000_segmented_288.yaml

python -m tools.apply_segmenter \
  --manifest data/ISIC2019/isic2019_manifest.csv \
  --checkpoint runs/isic2018_segmenter/best_segmenter.pt \
  --output-dir data/ISIC2019/segmented_images \
  --output-manifest data/ISIC2019/isic2019_segmented_manifest.csv
python train.py --manifest data/ISIC2019/isic2019_segmented_manifest.csv --config config/isic2019_segmented_288.yaml
```

The segmented configs use the same 288 px input size as the full-image configs. Review some generated crops before training. Segmentation may improve focus, but an accuracy increase is not guaranteed.

## Tune safely

- If CUDA reports out-of-memory, keep `image_size: 288`, reduce `batch_size`, and raise `gradient_accumulation` to preserve the effective batch size.
- Start with `rounds: 8` to verify data. The default 30 rounds is a realistic fine-tuning run, not a claim of clinical readiness.
- Use `clients: 3` to reproduce the paper's simulation. Federated simulation is not a substitute for real institutional deployment; raw data still stays in each configured local client in a real deployment.
- Set `use_cbam: false` if you need the original MobileNetV3 head for an ablation. The GTX 1650 profile also uses 288 px inputs with a smaller batch and gradient accumulation; 4 GB VRAM may still require further batch-size reduction.
- The GPU configs enable strong augmentation and apply MixUp or CutMix to half of training batches. Weight decay is `0.01`; stochastic depth is applied through MobileNetV3 residual blocks.
- `learning_rate` sets the classifier-head rate (`0.001`). The backbone starts at `backbone_learning_rate: 0.0001`, with `layerwise_lr_decay` reducing rates for earlier feature layers. Validation-driven reductions scale all rates together.
- Validation and final test metrics use EMA weights rather than raw weights. Local and centralized EMAs update per optimizer step (`ema_decay: 0.995`); the federated server EMA updates once per round (`server_ema_decay: 0.9`).
- The global validation macro-F1 controls the learning-rate schedule. After `lr_patience` rounds without improvement, each client scales future rates by `lr_factor`, down to `min_learning_rate`.
