# Foot segmentation dataset and training pipeline

This directory contains the reproducible path from publisher archives to a
browser-ready TensorFlow Lite foot-segmentation model. It deliberately keeps
dataset acquisition, data quality, training, evaluation, and deployment as
separate steps so a good segmentation score is not mistaken for millimetre
measurement accuracy.

## Current local status

- `FootNet_v1.0.zip` is downloaded under `ml/data/raw/footnet/`.
- Its size, MD5, and SHA-256 match the pinned release in
  `FOOTNET_CHECKSUMS.txt`.
- All 191 image/mask pairs are normalized under
  `ml/data/processed/footnet/` and pass full asset/checksum validation.
- The release contains 69 dorsal/top, 63 plantar/sole, and 59 medial/side
  samples. The app should fine-tune and evaluate primarily on the 69 dorsal
  samples because they are closest to the camera view used for measurement.
- A two-stage Colab run has produced the bundled dorsal-view model at
  `public/models/foot_segmenter_fp16_metadata.tflite`. The result ZIP was
  imported and verified on 2026-08-22; see `MODEL_CARD.md` for its hashes,
  evaluation, and release limitations.
- Native Windows GPU training is not configured on this machine; use Python
  3.11 in WSL2 or Google Colab for future training runs.

The raw and processed datasets are intentionally ignored by Git. The manifest
and lock inside the processed directory record every normalized asset hash,
the source release, the accepted license, and transformations.

## Dataset policy

| Source | Use in this project | Access and license |
| --- | --- | --- |
| [FootNet v1.0](https://zenodo.org/records/20457253) | Real-image fine-tuning; 191 image/mask pairs | Adith Vijay, Akash Prabhune, Vinay R Srihari, and Akhila Rayampalli; direct archive, CC BY 4.0. Preserve attribution. |
| [SynFoot](https://github.com/OllieBoyne/SynFoot) | Optional synthetic pretraining | Manual [publisher form](https://forms.gle/jZqoLPfQrkEa5XUF8). Treat dataset terms as non-commercial research unless written commercial permission is obtained. |
| [CMU Human Foot Keypoint Dataset](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/) | Deferred; keypoints only, not masks | Annotation host is currently unavailable and the referenced COCO image archive is very large. |

FootNet has no physical foot-length labels, no card-calibration ground truth,
and no confirmed subject IDs. It can improve the silhouette mask but cannot by
itself prove millimetre accuracy or a person-independent validation score.

## Dataset commands

Run commands from the project root. `python` below means a Python 3.11+
environment with `numpy` and `Pillow`. The bundled Python in this Codex
workspace can run all dataset commands without installing TensorFlow.

List source and license policy:

```powershell
python -m ml.footdata sources
```

Rebuild FootNet from the pinned archive:

```powershell
python -m ml.footdata import footnet `
  --archive ml/data/raw/footnet/FootNet_v1.0.zip `
  --output-root ml/data/processed/footnet `
  --expected-sha256 3c1bfe4a037f60925325421d467e2310bf65a9c8b2485314a0069b6bbe579b07 `
  --accept-license
```

Validate the manifest and every normalized file:

```powershell
python -m ml.footdata validate `
  ml/data/processed/footnet/samples.jsonl `
  --data-root ml/data/processed/footnet `
  --verify-hashes --strict
```

The strict split keeps samples without a confirmed subject ID in training.
For model development, the explicit weak policy can create file-level holdouts:

```powershell
python -m ml.footdata split `
  ml/data/processed/footnet/samples.jsonl `
  --output ml/data/processed/footnet/samples.dev.jsonl `
  --lock ml/data/processed/footnet/splits.dev.lock.json `
  --allow-unresolved-eval
```

The weak holdouts are useful for debugging and threshold selection, but their
scores are not subject-independent. Replace them with a manifest containing
real `group_id` values as soon as participant-level data is available.

To import a manually downloaded SynFoot archive:

```powershell
python -m ml.footdata import synfoot-v1 `
  --archive C:/path/to/SynFoot-v1.zip `
  --output-root ml/data/processed/synfoot-v1 `
  --accept-license
```

Do not use `--accept-license` until the terms bundled with that archive have
actually been reviewed and accepted.

## Training

Use Python 3.11 in Linux/WSL2 or Colab for GPU training. Modern TensorFlow GPU
support is not provided on native Windows; see the
[TensorFlow installation guide](https://www.tensorflow.org/install/pip).

For the simplest Colab path, open
`ml/colab/Foot_Segmentation_Training.ipynb`, select a GPU runtime, and use
**Run all**. It downloads and verifies the pinned FootNet archive, saves
checkpoints to Google Drive, performs both training stages and validation-only
threshold selection, exports TFLite, and downloads a compact result ZIP.

```bash
python3.11 -m venv ml/.venv
source ml/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ml/requirements-train.txt

python ml/train.py \
  --manifest ml/data/processed/footnet/samples.dev.jsonl \
  --real-only \
  --output ml/artifacts/footnet-all-views

python ml/train.py \
  --manifest ml/data/processed/footnet/samples.dev.jsonl \
  --views dorsal \
  --real-only \
  --initial-model ml/artifacts/footnet-all-views/best.keras \
  --learning-rate 0.0001 \
  --output ml/artifacts/footnet-dorsal
```

The first run learns from every view; the second specializes the result for
the app's dorsal/top view at a lower learning rate. The default model is a
`384x384` MobileNetV2 U-Net with two softmax classes:
`background` and `foot`. Inputs are RGB values normalized to `[-1, 1]`.
Training uses aspect-ratio-preserving letterboxing, cross-entropy plus Dice
loss, and Dice/IoU metrics.

Evaluate on validation data and choose the probability threshold there:

```bash
python ml/evaluate.py \
  --manifest ml/data/processed/footnet/samples.dev.jsonl \
  --model ml/artifacts/footnet-dorsal/best.keras \
  --split val \
  --views dorsal \
  --output ml/artifacts/footnet-dorsal/evaluation-val
```

Do not tune a threshold on the test split. The evaluator reports mask metrics
and length/width silhouette error at model resolution. Physical millimetre
errors require a separate app-captured set with a verified reference card and
measured foot dimensions.

## TensorFlow Lite and MediaPipe

```bash
python ml/export_tflite.py \
  --model ml/artifacts/footnet-dorsal/best.keras \
  --output ml/artifacts/foot_segmenter_fp16.tflite
```

Attach Image Segmenter labels and normalization metadata in a separate
Python 3.10/3.11 Linux or WSL environment. `tflite-support==0.4.4` does not
provide a native Windows wheel. For native Windows, `attach_metadata_portable.py`
can use the same package's generated FlatBuffer schemas without loading its
compiled extension.

```bash
python -m pip install -r ml/requirements-metadata.txt
python ml/add_metadata.py \
  --model ml/artifacts/foot_segmenter_fp16.tflite \
  --output ml/artifacts/foot_segmenter_fp16_metadata.tflite
```

MediaPipe metadata does not preserve aspect ratio automatically. Before
calling Image Segmenter in the browser, the app must place the camera ROI into
a square `384x384` letterboxed canvas, then remove the padding from the output
mask before contour measurement. Passing a nonsquare ROI directly would
stretch the foot and bias length/width.

Keep the existing seven-frame measurement tracker after segmentation. It
stabilizes frame-to-frame measurements; it does not replace model evaluation.

## Release gate

Before copying a model into `public/models/`, fill in `MODEL_CARD.md` and verify:

1. Person-grouped external test photos are never used for training or
   threshold selection.
2. Dice, IoU, geometry error, invalid-mask rate, and app-level millimetre
   MAE/P90 are recorded separately.
3. Keras and TFLite outputs agree on a fixed golden set.
4. The browser uses a square letterboxed input and unletterboxes the mask.
5. FootNet attribution and all other dataset terms are included with the
   released model.
