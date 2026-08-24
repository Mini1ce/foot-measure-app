# Foot Measure App

Browser-based foot measurement built with Next.js/React, MediaPipe Tasks, and
OpenCV.js. Camera frames stay in the browser. The app uses a physical ID-1
reference card (`85.60 × 53.98 mm`) for calibration; it does not read the card
or verify identity.

## Measurement flow

1. Open the rear camera when available and wait for valid video metadata.
2. Find a convex ID-1-shaped quadrilateral inside the card guide.
3. Validate aspect ratio, corners, edge support, perspective, size, and
   continuity across frames.
4. Build an image-to-millimetre homography from the four ordered card corners.
5. Segment one foot with the bundled FootNet MediaPipe model. If that model is
   unavailable, retain the OpenCV contour fallback.
6. Transform the contour to the calibrated plane. Measure length on the
   heel-to-toe axis and estimate width from robust cross-sections perpendicular
   to that axis, excluding the heel and toe extremes.
7. Accept at least seven compatible frames, reject camera/scale/orientation/
   contour jumps, and report medians with a quality status.

The old pixels-per-millimetre (PPM) path remains as a low-confidence preview
when homography is unavailable. PPM-only results cannot be saved and do not
produce a shoe-size recommendation.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). A real phone must load the
site from a secure HTTPS origin before the browser will expose its camera
(localhost is the normal development exception).

Production verification:

```bash
npm run lint
npx tsc --noEmit --incremental false
npm run test:detection
npm run build
npm run start
```

`test:detection` covers synthetic perspective/portrait/landscape cards,
non-card rectangles, rotated feet, widest-point geometry, invalid homography,
tracking resets, object-cover mapping, processing backpressure, and resource
cleanup. These fixtures are deterministic engineering checks, not evidence of
physical accuracy on real feet or phones.

## Capture requirements

- Put a standard ID-1 reference card flat on the same surface as the bare foot.
- Hold the phone above and roughly parallel to that surface.
- Keep the whole foot and card inside their guides with even lighting.
- Keep the phone, foot, and card still while seven accepted samples are
  collected.
- Save is enabled only when the card, foot, homography, tracking, and final
  confidence policy all pass.

A single monocular image cannot prove that two objects are on the same plane,
and geometry cannot semantically distinguish every ID-1-shaped printed object
from a genuine reference card. The guides, tilt/size/edge checks, and temporal
stability reduce these risks but do not eliminate them.

## Model and training

The bundled model is
`public/models/foot_segmenter_fp16_metadata.tflite`. Dataset provenance,
training, evaluation, export, and known limitations are documented in
[`ml/README.md`](ml/README.md) and [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md).
Raw datasets and training artifacts are excluded from Git.

The measurement changes in this project do not retrain or modify the bundled
model. Real-device validation with dimensioned feet, multiple phones, lighting
conditions, surfaces, skin tones, and camera angles is still required before
making a physical-accuracy claim.
