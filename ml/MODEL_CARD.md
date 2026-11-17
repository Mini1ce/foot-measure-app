# Foot segmenter model card

Status: **development model bundled for browser testing; not yet validated for physical millimetre accuracy**

## Model

- Name/version: FootSegmenter, development version 1
- Training run: two-stage FootNet training completed in Google Colab on 2026-08-22
- Raw FP16 model SHA-256: `e444456727b3d536f645072e54e36907177892fccf26a277834789b5889fd774`
- MediaPipe-metadata model SHA-256: `480a97381c7b59c32b6e29b9265ed4073c169eb4fc4bf3b7c71e67380952a130`
- Architecture: MobileNetV2 U-Net, `384x384`, two-class softmax
- Input: RGB, letterboxed to `384x384`, normalized with mean/std `127.5`
- Output labels: `background`, `foot`
- Selected foot-probability threshold: `0.5`, selected on the validation split

## Intended use

Segment one visible foot in the guided top-view camera region so the existing
contour and seven-frame tracker can estimate a stable silhouette length and
width. This model is not a medical diagnosis system and must not infer health,
identity, age, sex, or other sensitive attributes.

## Training data and attribution

- FootNet v1.0 by Adith Vijay, Akash Prabhune, Vinay R Srihari, and Akhila
  Rayampalli, Zenodo record
  [20457253](https://zenodo.org/records/20457253), CC BY 4.0.
- Additional datasets, versions, licenses, and permissions: none in this run
- App-captured participant count and consent/retention policy: none in this run
- Train/validation/test grouping method: weak sample-level split because the
  public release does not provide confirmed participant IDs

FootNet transformations must be disclosed: source metadata is removed during
normalization, masks are converted to binary PNG, and the single pinned source
image `righttop_004.jpg` is rotated clockwise to match its publisher mask.

### SynFoot V2 candidate pipeline (not yet a model release)

The repository now contains a reproducible path for SynFoot V2 pretraining,
followed by the same real-image FootNet fine-tuning stages. The local V2 set has
100,000 renders but only eight underlying `foot-id` scan groups. Its separate
`pretrain_split` holds out whole scan groups (six train, two validation), uses
the raw antialiased-mask threshold `>=128`, and can filter to non-cropped camera
views within 25° or 35° of vertical. Synthetic shape coefficients are retained
as provenance and are **not** interpreted as millimetres. Exact duplicate RGB
records are excluded from pretraining, selected package assets are checksum
verified, and candidate runs carry dataset/source/checkpoint fingerprints in a
separate provenance document.

No SynFoot-trained weights or metrics are recorded in this model card yet. A
candidate must outperform this bundled FootNet-only baseline on the fixed real
evaluation set, pass Keras/TFLite/browser parity, and then pass real-phone
millimetre tests before replacing the deployed model.

## Evaluation

Record results separately for real dorsal images and any synthetic data.

| Metric | Validation | External test |
| --- | ---: | ---: |
| Dice | best training-log value `0.9525` | `0.9683` mean |
| IoU | best training-log value `0.9094` | `0.9395` mean |
| Precision / recall | not retained in returned validation report | `0.9570` / `0.9814` mean |
| Invalid-mask rate | not retained in returned validation report | `0%` (0/11) |
| Length error (%) | not retained in returned validation report | `2.15%` mean / `8.57%` P95 |
| Width error (%) | not retained in returned validation report | `15.65%` mean / `82.52%` P95 |
| App foot-length MAE / P90 (mm) | not applicable | — |
| App foot-width MAE / P90 (mm) | not applicable | — |

Threshold selection set and rationale: `0.5`, selected on the validation split
before the 11-image dorsal test split was evaluated.

The table above is a historical result from the earlier global-silhouette
geometry evaluator. The current evaluator uses versioned
`heel-to-toe-cross-sections-v1` geometry, reports target-ineligible masks
separately from prediction failures, and must not be compared numerically with
that table until the same checkpoint is rerun. The current FootNet dorsal audit
contains 69 images: 36 touch an image border and 3 do not yield valid
measurement geometry. These are review findings, not model-accuracy results;
human include/exclude decisions remain pending.

Keras/TFLite parity result and golden-set hash: export validation confirmed the
model tensors and FP16 TFLite artifact; a fixed browser golden-set parity test
is still required.

### Application geometry smoke tests (2026-08-24)

The application calibration/contour tests are separate from model evaluation:
they record nine deterministic measurements across four idealized scenario
families (five rotations, one end-outlier profile, one curved asymmetric
profile, and clean/noisy independent projective image-to-plane transforms).
They do not run the neural network and are not physical camera measurements.

| Synthetic application metric | Result |
| --- | ---: |
| Foot-length MAE | `1.461 mm` |
| Foot-width MAE | `0.651 mm` |

These numbers only show deterministic geometry behavior for the current test
fixtures. They must not be presented as expected accuracy for real feet,
cameras, cards, lighting, or surfaces. No model weights or model threshold were
changed as part of the application calibration work.

## Limitations

- Public FootNet data has no confirmed subject IDs, card calibration, or
  physical dimension ground truth. A file-level development split can leak a
  participant across splits and must not be presented as independent testing.
- Accuracy can degrade with socks, multiple feet, severe occlusion, cropped
  toes/heel, glare, shadows, skin/background similarity, unusual camera angle,
  or a reference card outside the foot plane.
- Segmentation quality alone does not establish millimetre accuracy; camera
  perspective, lens distortion, card detection, card/foot coplanarity, and
  calibration remain separate error sources.
- A single monocular view cannot prove that the foot and card are coplanar.
  The application can only reject strong perspective/placement/stability
  violations and instruct the user to place both on the same surface.
- Shape and edge checks cannot semantically prove that a detected ID-1-sized
  rectangle is a genuine reference card.
- Input must be letterboxed. Stretching a nonsquare ROI changes geometry.

## Deployment and monitoring

- MediaPipe Tasks Vision version: `1.0.1`, self-hosted with matching WASM files
- TFLite metadata inspection result: `TFLITE_METADATA` / `M001`, RGB input,
  mean/std `127.5`, and `background` / `foot` labels verified
- Browser/device performance: initialization smoke test recorded separately;
  full camera-device timing is still required
- OpenCV fallback behavior: visual contour detection remains available when
  the model cannot initialize or does not return a usable contour
- Data retained on device/server: live frames are processed in the browser and
  are not uploaded by this app
- Rollback model and trigger: remove the bundled model or disable model
  initialization if startup or contour regressions occur; OpenCV remains the fallback
