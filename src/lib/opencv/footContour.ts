/* eslint-disable @typescript-eslint/no-explicit-any -- OpenCV.js exposes an untyped CDN runtime. */
import {
  FootContourResult,
  FootROI,
  FootSegmentationMask,
  HomographyTransform,
  LetterboxPlacement,
  Point,
} from '@/types';
import { measureFootGeometry } from '@/lib/calculations/footGeometry';

declare const cv: any;

interface FootCandidate {
  contour: any;
  score: number;
}

function emptyFootResult(): FootContourResult {
  return {
    detected: false,
    lengthPx: 0,
    widthPx: 0,
    contourPoints: [],
    heelPoint: null,
    toePoint: null,
    widestPoints: null,
    geometryConfidence: 0,
    shapeSignature: [],
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function clamp01(value: number): number {
  return clamp(value, 0, 1);
}

function oddKernelSize(value: number, minimum: number, maximum: number): number {
  const clamped = Math.round(clamp(value, minimum, maximum));
  return clamped % 2 === 0 ? clamped + 1 : clamped;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * Estimate the surface colour from the ROI corners and mark pixels that differ
 * from it. This finds a bare foot, a sock, or a shoe on a reasonably plain
 * floor and removes the old hard dependency on a skin-colour match.
 */
function createBackgroundDifferenceMask(roiMat: any): any {
  const mask = cv.Mat.zeros(roiMat.rows, roiMat.cols, cv.CV_8UC1);
  let succeeded = false;
  try {
    const rgba = roiMat.data as Uint8Array;
    const redSamples: number[] = [];
    const greenSamples: number[] = [];
    const blueSamples: number[] = [];
    const patchWidth = Math.max(4, Math.floor(roiMat.cols * 0.09));
    const patchHeight = Math.max(4, Math.floor(roiMat.rows * 0.09));
    const step = Math.max(1, Math.floor(Math.min(patchWidth, patchHeight) / 12));

    const samplePatch = (startX: number, startY: number) => {
      for (let y = startY; y < startY + patchHeight; y += step) {
        for (let x = startX; x < startX + patchWidth; x += step) {
          const index = (y * roiMat.cols + x) * 4;
          redSamples.push(rgba[index]);
          greenSamples.push(rgba[index + 1]);
          blueSamples.push(rgba[index + 2]);
        }
      }
    };

    samplePatch(0, 0);
    samplePatch(roiMat.cols - patchWidth, 0);
    samplePatch(0, roiMat.rows - patchHeight);
    samplePatch(roiMat.cols - patchWidth, roiMat.rows - patchHeight);

    const backgroundRed = median(redSamples);
    const backgroundGreen = median(greenSamples);
    const backgroundBlue = median(blueSamples);
    const sampleDistances = redSamples.map((red, index) =>
      Math.hypot(
        red - backgroundRed,
        greenSamples[index] - backgroundGreen,
        blueSamples[index] - backgroundBlue
      )
    );
    sampleDistances.sort((a, b) => a - b);
    const surfaceVariation = sampleDistances[Math.floor(sampleDistances.length * 0.9)] ?? 0;
    const foregroundThreshold = clamp(surfaceVariation + 14, 24, 72);
    const maskData = mask.data as Uint8Array;

    for (let pixel = 0, sourceIndex = 0; pixel < maskData.length; pixel++, sourceIndex += 4) {
      const distance = Math.hypot(
        rgba[sourceIndex] - backgroundRed,
        rgba[sourceIndex + 1] - backgroundGreen,
        rgba[sourceIndex + 2] - backgroundBlue
      );
      maskData[pixel] = distance >= foregroundThreshold ? 255 : 0;
    }

    succeeded = true;
    return mask;
  } finally {
    if (!succeeded) mask.delete();
  }
}

function inRange(source: any, lowerValues: number[], upperValues: number[]): any {
  let lower: any = null;
  let upper: any = null;
  let output: any = null;
  let succeeded = false;
  try {
    lower = new cv.Mat(source.rows, source.cols, source.type(), lowerValues);
    upper = new cv.Mat(source.rows, source.cols, source.type(), upperValues);
    output = new cv.Mat();
    cv.inRange(source, lower, upper, output);
    succeeded = true;
    return output;
  } finally {
    lower?.delete();
    upper?.delete();
    if (!succeeded) output?.delete();
  }
}

/** Combine HSV and YCrCb ranges so different skin tones and lighting work. */
function createSkinMask(roiMat: any): any {
  let rgb: any = null;
  let hsv: any = null;
  let yCrCb: any = null;
  let hsvWarm: any = null;
  let hsvRed: any = null;
  let yCrCbSkin: any = null;
  let combined: any = null;

  try {
    rgb = new cv.Mat();
    hsv = new cv.Mat();
    yCrCb = new cv.Mat();
    combined = new cv.Mat();
    cv.cvtColor(roiMat, rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    cv.cvtColor(rgb, yCrCb, cv.COLOR_RGB2YCrCb);

    hsvWarm = inRange(hsv, [0, 12, 35, 0], [28, 255, 255, 255]);
    hsvRed = inRange(hsv, [160, 12, 35, 0], [180, 255, 255, 255]);
    yCrCbSkin = inRange(yCrCb, [0, 133, 77, 0], [255, 180, 135, 255]);

    cv.bitwise_or(hsvWarm, hsvRed, combined);
    cv.bitwise_or(combined, yCrCbSkin, combined);
    return combined.clone();
  } finally {
    rgb?.delete();
    hsv?.delete();
    yCrCb?.delete();
    hsvWarm?.delete();
    hsvRed?.delete();
    yCrCbSkin?.delete();
    combined?.delete();
  }
}

function cleanMask(mask: any): any {
  const minimumDimension = Math.min(mask.rows, mask.cols);
  const closeSize = oddKernelSize(minimumDimension * 0.025, 7, 19);
  const openSize = oddKernelSize(minimumDimension * 0.012, 3, 9);
  let closeKernel: any = null;
  let openKernel: any = null;
  let closed: any = null;
  let cleaned: any = null;

  try {
    closeKernel = cv.Mat.ones(closeSize, closeSize, cv.CV_8U);
    openKernel = cv.Mat.ones(openSize, openSize, cv.CV_8U);
    closed = new cv.Mat();
    cleaned = new cv.Mat();
    cv.morphologyEx(mask, closed, cv.MORPH_CLOSE, closeKernel);
    cv.morphologyEx(closed, cleaned, cv.MORPH_OPEN, openKernel);
    return cleaned.clone();
  } finally {
    closeKernel?.delete();
    openKernel?.delete();
    closed?.delete();
    cleaned?.delete();
  }
}

/** Remove the already-detected reference card from a foot candidate mask. */
function excludeCardBounds(
  mask: any,
  cardCorners: Point[] | null,
  roiX: number,
  roiY: number
): void {
  if (!cardCorners || cardCorners.length === 0) return;

  const center = cardCorners.reduce(
    (sum, point) => ({ x: sum.x + point.x / 4, y: sum.y + point.y / 4 }),
    { x: 0, y: 0 }
  );
  const padding = Math.max(4, Math.round(Math.min(mask.cols, mask.rows) * 0.012));
  const expanded = cardCorners.map((point) => {
    const deltaX = point.x - center.x;
    const deltaY = point.y - center.y;
    const length = Math.max(1, Math.hypot(deltaX, deltaY));
    return {
      x: Math.round(clamp(point.x + (deltaX / length) * padding - roiX, 0, mask.cols - 1)),
      y: Math.round(clamp(point.y + (deltaY / length) * padding - roiY, 0, mask.rows - 1)),
    };
  });
  const minimumX = Math.max(0, Math.floor(Math.min(...expanded.map((point) => point.x))));
  const maximumX = Math.min(mask.cols - 1, Math.ceil(Math.max(...expanded.map((point) => point.x))));
  const minimumY = Math.max(0, Math.floor(Math.min(...expanded.map((point) => point.y))));
  const maximumY = Math.min(mask.rows - 1, Math.ceil(Math.max(...expanded.map((point) => point.y))));
  const data = mask.data as Uint8Array;
  for (let y = minimumY; y <= maximumY; y++) {
    for (let x = minimumX; x <= maximumX; x++) {
      const crossProducts = expanded.map((point, index) => {
        const next = expanded[(index + 1) % expanded.length];
        return (next.x - point.x) * (y - point.y) - (next.y - point.y) * (x - point.x);
      });
      const inside = crossProducts.every((value) => value >= 0) ||
        crossProducts.every((value) => value <= 0);
      if (inside) data[y * mask.cols + x] = 0;
    }
  }
}

function orientedContourBounds(contour: any): { longSide: number; shortSide: number } | null {
  const coordinates = contour.data32S as Int32Array;
  const pointCount = Math.floor(coordinates.length / 2);
  if (pointCount < 3) return null;

  let centerX = 0;
  let centerY = 0;
  for (let index = 0; index < pointCount; index++) {
    centerX += coordinates[index * 2] / pointCount;
    centerY += coordinates[index * 2 + 1] / pointCount;
  }
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (let index = 0; index < pointCount; index++) {
    const x = coordinates[index * 2] - centerX;
    const y = coordinates[index * 2 + 1] - centerY;
    xx += x * x;
    yy += y * y;
    xy += x * y;
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const major = { x: Math.cos(angle), y: Math.sin(angle) };
  const minor = { x: -major.y, y: major.x };
  let minimumMajor = Number.POSITIVE_INFINITY;
  let maximumMajor = Number.NEGATIVE_INFINITY;
  let minimumMinor = Number.POSITIVE_INFINITY;
  let maximumMinor = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < pointCount; index++) {
    const x = coordinates[index * 2] - centerX;
    const y = coordinates[index * 2 + 1] - centerY;
    const majorProjection = x * major.x + y * major.y;
    const minorProjection = x * minor.x + y * minor.y;
    minimumMajor = Math.min(minimumMajor, majorProjection);
    maximumMajor = Math.max(maximumMajor, majorProjection);
    minimumMinor = Math.min(minimumMinor, minorProjection);
    maximumMinor = Math.max(maximumMinor, minorProjection);
  }
  const firstSide = maximumMajor - minimumMajor;
  const secondSide = maximumMinor - minimumMinor;
  if (!Number.isFinite(firstSide) || !Number.isFinite(secondSide)) return null;
  return {
    longSide: Math.max(firstSide, secondSide),
    shortSide: Math.min(firstSide, secondSide),
  };
}

function scoreFootContour(contour: any, roiWidth: number, roiHeight: number): number | null {
  const roiArea = roiWidth * roiHeight;
  const area = Math.abs(cv.contourArea(contour));
  const areaRatio = area / roiArea;
  if (areaRatio < 0.025 || areaRatio > 0.82) return null;

  const bounds = cv.boundingRect(contour);
  const orientedBounds = orientedContourBounds(contour);
  if (!orientedBounds) return null;
  const { longSide, shortSide } = orientedBounds;
  if (shortSide < 8) return null;

  const aspectRatio = longSide / shortSide;
  if (aspectRatio < 1.25 || aspectRatio > 6.5) return null;

  const fillRatio = area / Math.max(1, longSide * shortSide);
  if (fillRatio < 0.2) return null;

  const margin = Math.max(2, Math.round(Math.min(roiWidth, roiHeight) * 0.008));
  const touches = [
    bounds.x <= margin,
    bounds.y <= margin,
    bounds.x + bounds.width >= roiWidth - margin,
    bounds.y + bounds.height >= roiHeight - margin,
  ];
  const touchCount = touches.filter(Boolean).length;
  if ((touches[0] && touches[2]) || (touches[1] && touches[3]) || touchCount >= 3) {
    return null;
  }

  const centerX = (bounds.x + bounds.width / 2) / roiWidth;
  const centerY = (bounds.y + bounds.height / 2) / roiHeight;
  const centerDistance = Math.hypot(centerX - 0.5, centerY - 0.5);
  const areaScore = clamp01(areaRatio / 0.22) * (areaRatio > 0.65 ? 0.65 : 1);
  const aspectScore = clamp01(1 - Math.abs(aspectRatio - 2.65) / 2.65);
  const fillScore = clamp01(1 - Math.abs(fillRatio - 0.58) / 0.48);
  const borderScore = clamp01(1 - touchCount * 0.2);
  const centerScore = clamp01(1 - centerDistance / 0.7);
  return (
    areaScore * 0.32 +
    aspectScore * 0.26 +
    fillScore * 0.17 +
    borderScore * 0.15 +
    centerScore * 0.1
  );
}

function findBestFootCandidate(mask: any, sourceBonus = 0): FootCandidate | null {
  let contours: any = null;
  let hierarchy: any = null;
  let best: FootCandidate | null = null;

  try {
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      try {
        const baseScore = scoreFootContour(contour, mask.cols, mask.rows);
        if (baseScore === null) continue;
        const score = clamp01(baseScore + sourceBonus);
        if (!best || score > best.score) {
          const clonedContour = contour.clone();
          best?.contour.delete();
          best = { contour: clonedContour, score };
        }
      } finally {
        contour.delete();
      }
    }
  } catch (error) {
    best?.contour.delete();
    best = null;
    throw error;
  } finally {
    contours?.delete();
    hierarchy?.delete();
  }

  return best;
}

function keepBetterCandidate(
  current: FootCandidate | null,
  candidate: FootCandidate | null
): FootCandidate | null {
  if (!candidate) return current;
  if (!current || candidate.score > current.score) {
    current?.contour.delete();
    return candidate;
  }
  candidate.contour.delete();
  return current;
}

function measureFootContour(
  contour: any,
  roiX: number,
  roiY: number,
  homography: HomographyTransform | null
): FootContourResult {
  const allPoints: Point[] = [];
  for (let i = 0; i < contour.rows; i++) {
    allPoints.push({
      x: contour.data32S[i * 2] + roiX,
      y: contour.data32S[i * 2 + 1] + roiY,
    });
  }
  return measureFootGeometry(allPoints, homography);
}

/**
 * Detect a foot inside an ROI. Skin segmentation is only one signal; adaptive
 * surface separation and intensity masks provide the main fallback for an
 * isolated top-down foot, where MediaPipe Pose commonly has no landmarks.
 */
export function detectFootContour(
  src: any,
  roi: FootROI,
  frameWidth: number,
  frameHeight: number,
  cardCorners: Point[] | null = null,
  homography: HomographyTransform | null = null
): FootContourResult {
  const roiX = clamp(Math.floor(roi.x), 0, frameWidth);
  const roiY = clamp(Math.floor(roi.y), 0, frameHeight);
  const roiW = clamp(Math.ceil(roi.width), 0, frameWidth - roiX);
  const roiH = clamp(Math.ceil(roi.height), 0, frameHeight - roiY);
  if (roiW <= 0 || roiH <= 0) return emptyFootResult();

  let roiMat: any = null;
  let roiView: any = null;
  let best: FootCandidate | null = null;

  try {
    // A Mat returned by roi() can keep the parent frame's row stride. Clone it
    // before direct rgba[] access so every row is contiguous.
    roiView = src.roi(new cv.Rect(roiX, roiY, roiW, roiH));
    roiMat = roiView.clone();
    roiView.delete();
    roiView = null;

    let foregroundRaw: any = null;
    let skinRaw: any = null;
    let foreground: any = null;
    let skin: any = null;
    let combinedRaw: any = null;
    let combined: any = null;
    try {
      foregroundRaw = createBackgroundDifferenceMask(roiMat);
      skinRaw = createSkinMask(roiMat);
      foreground = cleanMask(foregroundRaw);
      skin = cleanMask(skinRaw);
      combinedRaw = new cv.Mat();
      cv.bitwise_or(foreground, skin, combinedRaw);
      combined = cleanMask(combinedRaw);

      for (const mask of [combined, foreground, skin]) {
        excludeCardBounds(mask, cardCorners, roiX, roiY);
      }
      best = keepBetterCandidate(best, findBestFootCandidate(combined, 0.08));
      best = keepBetterCandidate(best, findBestFootCandidate(foreground, 0.04));
      best = keepBetterCandidate(best, findBestFootCandidate(skin, 0.06));
    } finally {
      foregroundRaw?.delete();
      skinRaw?.delete();
      foreground?.delete();
      skin?.delete();
      combinedRaw?.delete();
      combined?.delete();
    }

    // Low-contrast surfaces can defeat colour difference. Try both sides of an
    // Otsu split before giving up.
    if (!best || best.score < 0.68) {
      let gray: any = null;
      let blurred: any = null;
      let binaryRaw: any = null;
      let inverseRaw: any = null;
      let binary: any = null;
      let inverse: any = null;
      try {
        gray = new cv.Mat();
        blurred = new cv.Mat();
        binaryRaw = new cv.Mat();
        inverseRaw = new cv.Mat();
        cv.cvtColor(roiMat, gray, cv.COLOR_RGBA2GRAY);
        cv.GaussianBlur(gray, blurred, new cv.Size(7, 7), 0);
        cv.threshold(blurred, binaryRaw, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
        cv.bitwise_not(binaryRaw, inverseRaw);
        binary = cleanMask(binaryRaw);
        inverse = cleanMask(inverseRaw);
        excludeCardBounds(binary, cardCorners, roiX, roiY);
        excludeCardBounds(inverse, cardCorners, roiX, roiY);
        best = keepBetterCandidate(best, findBestFootCandidate(binary));
        best = keepBetterCandidate(best, findBestFootCandidate(inverse));
      } finally {
        gray?.delete();
        blurred?.delete();
        binaryRaw?.delete();
        inverseRaw?.delete();
        binary?.delete();
        inverse?.delete();
      }
    }

    if (!best || best.score < 0.46) {
      return emptyFootResult();
    }

    return measureFootContour(best.contour, roiX, roiY, homography);
  } finally {
    best?.contour.delete();
    roiView?.delete();
    roiMat?.delete();
  }
}

/**
 * Convert the FootNet probability mask back from its square letterbox into
 * camera ROI coordinates, then reuse the same contour geometry and checks as
 * the visual detector.
 */
export function detectFootContourFromModelMask(
  segmentation: FootSegmentationMask,
  roi: FootROI,
  placement: LetterboxPlacement,
  frameWidth: number,
  frameHeight: number,
  threshold: number,
  cardCorners: Point[] | null = null,
  homography: HomographyTransform | null = null
): FootContourResult {
  if (
    segmentation.width <= 0 ||
    segmentation.height <= 0 ||
    segmentation.data.length !== segmentation.width * segmentation.height ||
    !Number.isFinite(threshold) ||
    threshold < 0 ||
    threshold > 1
  ) {
    return emptyFootResult();
  }

  const roiX = clamp(Math.floor(roi.x), 0, frameWidth);
  const roiY = clamp(Math.floor(roi.y), 0, frameHeight);
  const roiWidth = clamp(Math.ceil(roi.width), 0, frameWidth - roiX);
  const roiHeight = clamp(Math.ceil(roi.height), 0, frameHeight - roiY);
  if (roiWidth <= 0 || roiHeight <= 0 || placement.targetSize <= 0) {
    return emptyFootResult();
  }

  const maskScaleX = segmentation.width / placement.targetSize;
  const maskScaleY = segmentation.height / placement.targetSize;
  const contentX = clamp(Math.round(placement.x * maskScaleX), 0, segmentation.width - 1);
  const contentY = clamp(Math.round(placement.y * maskScaleY), 0, segmentation.height - 1);
  const contentWidth = clamp(
    Math.round(placement.width * maskScaleX),
    1,
    segmentation.width - contentX
  );
  const contentHeight = clamp(
    Math.round(placement.height * maskScaleY),
    1,
    segmentation.height - contentY
  );
  const binaryData = new Uint8Array(segmentation.data.length);
  for (let index = 0; index < segmentation.data.length; index++) {
    binaryData[index] = segmentation.data[index] >= threshold ? 255 : 0;
  }

  let modelMask: any = null;
  let contentView: any = null;
  let resized: any = null;
  let cleaned: any = null;
  let best: FootCandidate | null = null;
  try {
    modelMask = cv.matFromArray(
      segmentation.height,
      segmentation.width,
      cv.CV_8UC1,
      binaryData
    );
    contentView = modelMask.roi(
      new cv.Rect(contentX, contentY, contentWidth, contentHeight)
    );
    resized = new cv.Mat();
    cv.resize(
      contentView,
      resized,
      new cv.Size(roiWidth, roiHeight),
      0,
      0,
      cv.INTER_NEAREST
    );
    cleaned = cleanMask(resized);
    excludeCardBounds(cleaned, cardCorners, roiX, roiY);
    best = findBestFootCandidate(cleaned, 0.16);
    if (!best || best.score < 0.42) return emptyFootResult();
    return measureFootContour(best.contour, roiX, roiY, homography);
  } finally {
    best?.contour.delete();
    cleaned?.delete();
    resized?.delete();
    contentView?.delete();
    modelMask?.delete();
  }
}

/** Draw the selected foot contour and its PCA measurement axes. */
export function drawFootOverlay(
  ctx: CanvasRenderingContext2D,
  result: FootContourResult,
  scaleX: number,
  scaleY: number
): void {
  if (!result.detected || result.contourPoints.length < 3) return;

  ctx.strokeStyle = 'rgba(99, 102, 241, 0.9)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  const first = result.contourPoints[0];
  ctx.moveTo(first.x * scaleX, first.y * scaleY);
  for (let i = 1; i < result.contourPoints.length; i++) {
    const point = result.contourPoints[i];
    ctx.lineTo(point.x * scaleX, point.y * scaleY);
  }
  ctx.closePath();
  ctx.stroke();

  if (result.heelPoint && result.toePoint) {
    ctx.strokeStyle = 'rgba(239,68,68,0.95)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(result.heelPoint.x * scaleX, result.heelPoint.y * scaleY);
    ctx.lineTo(result.toePoint.x * scaleX, result.toePoint.y * scaleY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (result.widestPoints) {
    ctx.strokeStyle = 'rgba(16,185,129,0.95)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(result.widestPoints[0].x * scaleX, result.widestPoints[0].y * scaleY);
    ctx.lineTo(result.widestPoints[1].x * scaleX, result.widestPoints[1].y * scaleY);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
