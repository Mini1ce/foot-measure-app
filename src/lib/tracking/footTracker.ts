import {
  FOOT_TRACK_MAX_GAP_MS,
  FOOT_TRACK_MAX_MISSES,
  FOOT_TRACK_MIN_DURATION_MS,
  FOOT_TRACK_MIN_SAMPLE_INTERVAL_MS,
  FOOT_TRACK_RESET_AFTER_OUTLIERS,
  FOOT_TRACK_TARGET_SAMPLES,
  FOOT_MAX_CARD_DISTANCE_MM,
} from '@/constants';
import type {
  CalibrationMode,
  FootContourResult,
  FootTrackingStatus,
  HomographyTransform,
  Point,
  QualityStatus,
} from '@/types';
import {
  cardPlaneCenter,
  distanceOnPlaneMm,
  transformPoint,
} from '@/lib/calculations/homography';

export interface StabilizedFootMeasurement {
  lengthMm: number;
  widthMm: number;
  ppm: number;
  confidence: number;
  quality: QualityStatus;
  calibrationMode: CalibrationMode;
  homographyValid: boolean;
}

export interface FootTrackingReference {
  center: Point;
  angleRad: number;
}

export type FootFrameCalibration =
  | {
      mode: 'homography';
      ppm: number;
      confidence: number;
      homography: HomographyTransform;
      fallbackReference: null;
    }
  | {
      mode: 'ppm-fallback';
      ppm: number;
      confidence: number;
      homography: null;
      fallbackReference: FootTrackingReference;
    };

export interface FootTrackerUpdate {
  status: FootTrackingStatus;
  stabilizedFoot: FootContourResult | null;
  measurement: StabilizedFootMeasurement | null;
  sampleAccepted: boolean;
}

interface FootTrackSample {
  result: FootContourResult;
  timestamp: number;
  ppm: number;
  lengthMm: number;
  widthMm: number;
  relativeCenterMm: Point;
  orientationRad: number;
  calibrationMode: CalibrationMode;
  confidence: number;
  shapeSignature: number[];
}

interface TrackSummary {
  sampleCount: number;
  consistency: number;
  stable: boolean;
  medianLengthMm: number;
  medianWidthMm: number;
  medianPpm: number;
  representative: FootTrackSample | null;
  confidence: number;
  quality: QualityStatus;
  calibrationMode: CalibrationMode | null;
}

export function initialFootTrackingStatus(): FootTrackingStatus {
  return {
    sampleCount: 0,
    targetSamples: FOOT_TRACK_TARGET_SAMPLES,
    isStable: false,
    consistency: 0,
    phase: 'searching',
    confidence: 0,
    quality: 'low',
    calibrationMode: null,
  };
}

/** Build a camera-motion-resistant reference from ordered card corners. */
export function cardTrackingReference(corners: Point[] | null): FootTrackingReference | null {
  if (!corners || corners.length !== 4) return null;
  const center = corners.reduce(
    (sum, point) => ({ x: sum.x + point.x / 4, y: sum.y + point.y / 4 }),
    { x: 0, y: 0 }
  );
  const topVector = {
    x: corners[1].x - corners[0].x,
    y: corners[1].y - corners[0].y,
  };
  const leftVector = {
    x: corners[3].x - corners[0].x,
    y: corners[3].y - corners[0].y,
  };
  const longAxis = Math.hypot(topVector.x, topVector.y) >= Math.hypot(leftVector.x, leftVector.y)
    ? topVector
    : leftVector;
  return { center, angleRad: Math.atan2(longAxis.y, longAxis.x) };
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

/** Range after dropping one extreme from each end of a seven-sample window. */
function trimmedRelativeRange(values: number[], center: number): number {
  if (values.length < 2 || center <= 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const trim = sorted.length >= 5 ? 1 : 0;
  return (sorted[sorted.length - 1 - trim] - sorted[trim]) / center;
}

function contourCenter(result: FootContourResult): Point | null {
  const points = result.contourPoints.length > 0
    ? result.contourPoints
    : [result.heelPoint, result.toePoint, ...(result.widestPoints ?? [])].filter(
        (point): point is Point => point !== null
      );
  if (points.length === 0) return null;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
}

function normalizeHalfTurn(angle: number): number {
  let normalized = angle;
  while (normalized >= Math.PI / 2) normalized -= Math.PI;
  while (normalized < -Math.PI / 2) normalized += Math.PI;
  return normalized;
}

function orientationDistance(first: number, second: number): number {
  return Math.abs(normalizeHalfTurn(first - second));
}

function meanOrientation(angles: number[]): number {
  if (angles.length === 0) return 0;
  const sum = angles.reduce(
    (value, angle) => ({
      sin: value.sin + Math.sin(angle * 2),
      cos: value.cos + Math.cos(angle * 2),
    }),
    { sin: 0, cos: 0 }
  );
  return Math.atan2(sum.sin, sum.cos) / 2;
}

function footOrientation(result: FootContourResult): number | null {
  if (result.heelPoint && result.toePoint) {
    return Math.atan2(
      result.toePoint.y - result.heelPoint.y,
      result.toePoint.x - result.heelPoint.x
    );
  }
  if (result.contourPoints.length < 3) return null;

  const center = result.contourPoints.reduce(
    (sum, point) => ({
      x: sum.x + point.x / result.contourPoints.length,
      y: sum.y + point.y / result.contourPoints.length,
    }),
    { x: 0, y: 0 }
  );
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const point of result.contourPoints) {
    const x = point.x - center.x;
    const y = point.y - center.y;
    xx += x * x;
    yy += y * y;
    xy += x * y;
  }
  return 0.5 * Math.atan2(2 * xy, xx - yy);
}

function createSample(
  result: FootContourResult | null,
  calibration: FootFrameCalibration | null,
  timestamp: number
): FootTrackSample | null {
  if (
    !result?.detected ||
    !Number.isFinite(result.lengthPx) ||
    !Number.isFinite(result.widthPx) ||
    result.lengthPx <= 0 ||
    result.widthPx <= 0 ||
    !calibration ||
    !Number.isFinite(calibration.ppm) ||
    calibration.ppm <= 0 ||
    !Number.isFinite(timestamp)
  ) {
    return null;
  }

  const center = contourCenter(result);
  if (!center || !result.heelPoint || !result.toePoint || !result.widestPoints) return null;

  let lengthMm: number;
  let widthMm: number;
  let relativeCenterMm: Point;
  let orientationRad: number;
  if (calibration.mode === 'homography') {
    if (!calibration.homography) return null;
    const transformedHeel = transformPoint(calibration.homography, result.heelPoint);
    const transformedToe = transformPoint(calibration.homography, result.toePoint);
    const transformedCenter = transformPoint(calibration.homography, center);
    const cardCenter = cardPlaneCenter(calibration.homography);
    const planeLength = distanceOnPlaneMm(
      calibration.homography,
      result.heelPoint,
      result.toePoint
    );
    const planeWidth = distanceOnPlaneMm(
      calibration.homography,
      result.widestPoints[0],
      result.widestPoints[1]
    );
    if (!transformedHeel || !transformedToe || !transformedCenter || !planeLength || !planeWidth) {
      return null;
    }
    lengthMm = planeLength;
    widthMm = planeWidth;
    relativeCenterMm = {
      x: transformedCenter.x - cardCenter.x,
      y: transformedCenter.y - cardCenter.y,
    };
    orientationRad = normalizeHalfTurn(
      Math.atan2(
        transformedToe.y - transformedHeel.y,
        transformedToe.x - transformedHeel.x
      )
    );
  } else {
    const reference = calibration.fallbackReference;
    const orientation = footOrientation(result);
    if (!reference || orientation === null) return null;
    lengthMm = result.lengthPx / calibration.ppm;
    widthMm = result.widthPx / calibration.ppm;
    const deltaX = (center.x - reference.center.x) / calibration.ppm;
    const deltaY = (center.y - reference.center.y) / calibration.ppm;
    const cosine = Math.cos(reference.angleRad);
    const sine = Math.sin(reference.angleRad);
    relativeCenterMm = {
      x: deltaX * cosine + deltaY * sine,
      y: -deltaX * sine + deltaY * cosine,
    };
    orientationRad = normalizeHalfTurn(orientation - reference.angleRad);
  }

  const widthRatio = widthMm / lengthMm;
  // Broad human-foot bounds reject a hand, card, or large floor region while
  // still allowing children and unusually large feet.
  if (
    lengthMm < 120 ||
    lengthMm > 400 ||
    widthMm < 35 ||
    widthMm > 190 ||
    widthRatio < 0.16 ||
    widthRatio > 0.58
  ) {
    return null;
  }
  if (Math.hypot(relativeCenterMm.x, relativeCenterMm.y) > FOOT_MAX_CARD_DISTANCE_MM) {
    return null;
  }

  const confidence = Math.min(
    clamp01(calibration.confidence),
    clamp01(result.geometryConfidence || 0)
  );

  return {
    result,
    timestamp,
    ppm: calibration.ppm,
    lengthMm,
    widthMm,
    relativeCenterMm,
    orientationRad,
    calibrationMode: calibration.mode,
    confidence: calibration.mode === 'homography' ? confidence : Math.min(0.45, confidence),
    shapeSignature: [...result.shapeSignature],
  };
}

function signatureDistance(first: number[], second: number[]): number {
  if (first.length === 0 || first.length !== second.length) return 0;
  return first.reduce((sum, value, index) => sum + Math.abs(value - second[index]), 0) /
    first.length;
}

function compatibleWithWindow(sample: FootTrackSample, samples: FootTrackSample[]): boolean {
  if (samples.length < 3) return true;

  const medianLengthMm = median(samples.map((item) => item.lengthMm));
  const medianWidthMm = median(samples.map((item) => item.widthMm));
  const medianCenterX = median(samples.map((item) => item.relativeCenterMm.x));
  const medianCenterY = median(samples.map((item) => item.relativeCenterMm.y));
  const referenceOrientation = meanOrientation(samples.map((item) => item.orientationRad));
  const lengthDelta = Math.abs(sample.lengthMm - medianLengthMm) / medianLengthMm;
  const widthDelta = Math.abs(sample.widthMm - medianWidthMm) / medianWidthMm;
  const centerDelta =
    Math.hypot(
      sample.relativeCenterMm.x - medianCenterX,
      sample.relativeCenterMm.y - medianCenterY
    );
  const angleDelta = orientationDistance(sample.orientationRad, referenceOrientation);
  const representativeSignature = samples[Math.floor(samples.length / 2)].shapeSignature;
  const shapeDelta = signatureDistance(sample.shapeSignature, representativeSignature);

  return (
    lengthDelta <= 0.1 &&
    widthDelta <= 0.2 &&
    centerDelta <= 35 &&
    angleDelta <= (25 * Math.PI) / 180 &&
    shapeDelta <= 0.055 &&
    sample.calibrationMode === samples[0].calibrationMode
  );
}

function compatiblePair(first: FootTrackSample, second: FootTrackSample): boolean {
  const lengthBase = (first.lengthMm + second.lengthMm) / 2;
  const widthBase = (first.widthMm + second.widthMm) / 2;
  return (
    Math.abs(first.lengthMm - second.lengthMm) / lengthBase <= 0.1 &&
    Math.abs(first.widthMm - second.widthMm) / widthBase <= 0.2 &&
    Math.hypot(
      first.relativeCenterMm.x - second.relativeCenterMm.x,
      first.relativeCenterMm.y - second.relativeCenterMm.y
    ) <= 35 &&
    orientationDistance(first.orientationRad, second.orientationRad) <= (25 * Math.PI) / 180 &&
    signatureDistance(first.shapeSignature, second.shapeSignature) <= 0.055 &&
    first.calibrationMode === second.calibrationMode
  );
}

function summarize(samples: FootTrackSample[]): TrackSummary {
  if (samples.length === 0) {
    return {
      sampleCount: 0,
      consistency: 0,
      stable: false,
      medianLengthMm: 0,
      medianWidthMm: 0,
      medianPpm: 0,
      representative: null,
      confidence: 0,
      quality: 'low',
      calibrationMode: null,
    };
  }

  const medianLengthMm = median(samples.map((sample) => sample.lengthMm));
  const medianWidthMm = median(samples.map((sample) => sample.widthMm));
  const medianPpm = median(samples.map((sample) => sample.ppm));
  const medianCenterX = median(samples.map((sample) => sample.relativeCenterMm.x));
  const medianCenterY = median(samples.map((sample) => sample.relativeCenterMm.y));
  const referenceOrientation = meanOrientation(samples.map((sample) => sample.orientationRad));
  const lengthSpread = trimmedRelativeRange(
    samples.map((sample) => sample.lengthMm),
    medianLengthMm
  );
  const widthSpread = trimmedRelativeRange(
    samples.map((sample) => sample.widthMm),
    medianWidthMm
  );
  const centerDistances = samples
    .map((sample) =>
      Math.hypot(
        sample.relativeCenterMm.x - medianCenterX,
        sample.relativeCenterMm.y - medianCenterY
      )
    )
    .sort((a, b) => a - b);
  const centerIndex = Math.max(0, centerDistances.length - (centerDistances.length >= 5 ? 2 : 1));
  const centerSpread = centerDistances[centerIndex];
  const orientationDistances = samples
    .map((sample) => orientationDistance(sample.orientationRad, referenceOrientation))
    .sort((a, b) => a - b);
  const orientationSpread = orientationDistances[centerIndex];
  const ppmSpread = trimmedRelativeRange(
    samples.map((sample) => sample.ppm),
    medianPpm
  );
  const shapeSpread = Math.max(
    0,
    ...samples.map((sample) => signatureDistance(sample.shapeSignature, samples[0].shapeSignature))
  );
  const duration = samples[samples.length - 1].timestamp - samples[0].timestamp;
  const consistency = clamp01(
    1 -
      Math.max(
        lengthSpread / 0.12,
        widthSpread / 0.24,
        centerSpread / 30,
        orientationSpread / ((20 * Math.PI) / 180),
        ppmSpread / 0.15,
        shapeSpread / 0.08
      )
  );
  const stable =
    samples.length >= FOOT_TRACK_TARGET_SAMPLES &&
    duration >= FOOT_TRACK_MIN_DURATION_MS &&
    lengthSpread <= 0.045 &&
    widthSpread <= 0.12 &&
    centerSpread <= 12 &&
    orientationSpread <= (10 * Math.PI) / 180 &&
    ppmSpread <= 0.08 &&
    shapeSpread <= 0.035;

  const representative = samples.reduce((best, sample) => {
    const score =
      Math.abs(sample.lengthMm - medianLengthMm) / medianLengthMm +
      Math.abs(sample.widthMm - medianWidthMm) / medianWidthMm +
      Math.hypot(
        sample.relativeCenterMm.x - medianCenterX,
        sample.relativeCenterMm.y - medianCenterY
      ) /
        35 +
      orientationDistance(sample.orientationRad, referenceOrientation) /
        ((25 * Math.PI) / 180);
    const bestScore =
      Math.abs(best.lengthMm - medianLengthMm) / medianLengthMm +
      Math.abs(best.widthMm - medianWidthMm) / medianWidthMm +
      Math.hypot(
        best.relativeCenterMm.x - medianCenterX,
        best.relativeCenterMm.y - medianCenterY
      ) /
        35 +
      orientationDistance(best.orientationRad, referenceOrientation) /
        ((25 * Math.PI) / 180);
    return score < bestScore ? sample : best;
  });

  const calibrationMode = samples.every(
    (sample) => sample.calibrationMode === samples[0].calibrationMode
  ) ? samples[0].calibrationMode : null;
  const confidence = Math.min(
    consistency,
    ...samples.map((sample) => sample.confidence)
  );
  const quality: QualityStatus = calibrationMode !== 'homography'
    ? 'low'
    : confidence >= 0.82
      ? 'high'
      : confidence >= 0.62
        ? 'medium'
        : 'low';

  return {
    sampleCount: samples.length,
    consistency,
    stable,
    medianLengthMm,
    medianWidthMm,
    medianPpm,
    representative,
    confidence,
    quality,
    calibrationMode,
  };
}

function statusFromSummary(summary: TrackSummary, forceUnstable = false): FootTrackingStatus {
  const isStable = summary.stable && !forceUnstable;
  return {
    sampleCount: summary.sampleCount,
    targetSamples: FOOT_TRACK_TARGET_SAMPLES,
    isStable,
    consistency: summary.consistency,
    phase:
      summary.sampleCount === 0
        ? 'searching'
        : summary.sampleCount < FOOT_TRACK_TARGET_SAMPLES
          ? 'tracking'
          : isStable
            ? 'stable'
          : 'stabilizing',
    confidence: summary.confidence,
    quality: summary.quality,
    calibrationMode: summary.calibrationMode,
  };
}

/**
 * Collect seven calibrated foot observations. Homography samples measure the
 * selected geometry in the card plane; PPM is retained only as a low-quality
 * preview fallback.
 */
export class FootTracker {
  private samples: FootTrackSample[] = [];
  private pendingReplacement: FootTrackSample[] = [];
  private missCount = 0;

  reset(): void {
    this.samples = [];
    this.pendingReplacement = [];
    this.missCount = 0;
  }

  getStatus(): FootTrackingStatus {
    return statusFromSummary(summarize(this.samples));
  }

  update(
    result: FootContourResult | null,
    calibration: FootFrameCalibration | null,
    timestamp: number
  ): FootTrackerUpdate {
    const sample = createSample(result, calibration, timestamp);
    if (!sample) {
      this.missCount++;
      this.pendingReplacement = [];
      if (this.missCount >= FOOT_TRACK_MAX_MISSES) this.reset();
      return this.buildUpdate(false, true);
    }

    const lastSample = this.samples[this.samples.length - 1];
    if (lastSample && sample.timestamp - lastSample.timestamp > FOOT_TRACK_MAX_GAP_MS) {
      this.samples = [sample];
      this.pendingReplacement = [];
      this.missCount = 0;
      return this.buildUpdate(true);
    }

    const pendingLast = this.pendingReplacement[this.pendingReplacement.length - 1];
    const lastObservationTimestamp = Math.max(
      lastSample?.timestamp ?? Number.NEGATIVE_INFINITY,
      pendingLast?.timestamp ?? Number.NEGATIVE_INFINITY
    );
    if (sample.timestamp - lastObservationTimestamp < FOOT_TRACK_MIN_SAMPLE_INTERVAL_MS) {
      // Apply the duplicate-frame gate before mismatch handling too. Otherwise
      // the same moved frame could count twice and incorrectly rebase a track.
      return this.buildUpdate(false, this.pendingReplacement.length > 0);
    }

    if (!compatibleWithWindow(sample, this.samples)) {
      this.missCount = 0;
      this.pendingReplacement =
        pendingLast && compatiblePair(pendingLast, sample)
          ? [...this.pendingReplacement, sample]
          : [sample];

      if (this.pendingReplacement.length >= FOOT_TRACK_RESET_AFTER_OUTLIERS) {
        this.samples = this.pendingReplacement.slice(-FOOT_TRACK_RESET_AFTER_OUTLIERS);
        this.pendingReplacement = [];
        return this.buildUpdate(true);
      }
      return this.buildUpdate(false, true);
    }

    this.missCount = 0;
    this.pendingReplacement = [];
    this.samples = [...this.samples, sample].slice(-FOOT_TRACK_TARGET_SAMPLES);
    return this.buildUpdate(true);
  }

  private buildUpdate(sampleAccepted: boolean, forceUnstable = false): FootTrackerUpdate {
    const summary = summarize(this.samples);
    const status = statusFromSummary(summary, forceUnstable);
    if (!status.isStable || !summary.representative || !summary.calibrationMode) {
      return { status, stabilizedFoot: null, measurement: null, sampleAccepted };
    }

    const stabilizedFoot: FootContourResult = summary.representative.result;
    return {
      status,
      stabilizedFoot,
      measurement: {
        lengthMm: summary.medianLengthMm,
        widthMm: summary.medianWidthMm,
        ppm: summary.medianPpm,
        confidence: summary.confidence,
        quality: summary.quality,
        calibrationMode: summary.calibrationMode,
        homographyValid: summary.calibrationMode === 'homography',
      },
      sampleAccepted,
    };
  }
}
