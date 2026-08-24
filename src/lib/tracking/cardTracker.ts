import {
  CARD_TRACK_MAX_ANGLE_SHIFT_RAD,
  CARD_TRACK_MAX_CENTER_SHIFT_RATIO,
  CARD_TRACK_MAX_GAP_MS,
  CARD_TRACK_MAX_SCALE_SHIFT,
  CARD_TRACK_MIN_SAMPLE_INTERVAL_MS,
  CARD_TRACK_TARGET_SAMPLES,
} from '@/constants';
import type { CardDetectionResult, CardIssue, CardTrackingStatus, Point } from '@/types';

interface CardTrackSample {
  timestamp: number;
  center: Point;
  scale: number;
  angle: number;
  corners: Point[];
  measurementReady: boolean;
  issue: CardIssue;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function normalizeHalfTurn(angle: number): number {
  let normalized = angle;
  while (normalized >= Math.PI / 2) normalized -= Math.PI;
  while (normalized < -Math.PI / 2) normalized += Math.PI;
  return normalized;
}

function angleDistance(first: number, second: number): number {
  return Math.abs(normalizeHalfTurn(first - second));
}

function createSample(
  result: CardDetectionResult | null,
  frameWidth: number,
  frameHeight: number,
  timestamp: number
): CardTrackSample | null {
  if (
    !result?.detected ||
    !result.corners ||
    result.corners.length !== 4 ||
    frameWidth <= 0 ||
    frameHeight <= 0
  ) return null;

  const corners = result.corners.map((point) => ({
    x: point.x / frameWidth,
    y: point.y / frameHeight,
  }));
  const center = corners.reduce(
    (sum, point) => ({ x: sum.x + point.x / 4, y: sum.y + point.y / 4 }),
    { x: 0, y: 0 }
  );
  const top = Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y);
  const right = Math.hypot(corners[2].x - corners[1].x, corners[2].y - corners[1].y);
  const bottom = Math.hypot(corners[2].x - corners[3].x, corners[2].y - corners[3].y);
  const left = Math.hypot(corners[3].x - corners[0].x, corners[3].y - corners[0].y);
  const topVector = { x: corners[1].x - corners[0].x, y: corners[1].y - corners[0].y };
  const leftVector = { x: corners[3].x - corners[0].x, y: corners[3].y - corners[0].y };
  const longAxis = (top + bottom) >= (left + right) ? topVector : leftVector;
  return {
    timestamp,
    center,
    scale: Math.sqrt(Math.max(0, ((top + bottom) / 2) * ((left + right) / 2))),
    angle: normalizeHalfTurn(Math.atan2(longAxis.y, longAxis.x)),
    corners,
    measurementReady: result.measurementReady,
    issue: result.issue,
  };
}

function difference(first: CardTrackSample, second: CardTrackSample): {
  center: number;
  scale: number;
  angle: number;
  cornerRms: number;
} {
  const center = Math.hypot(second.center.x - first.center.x, second.center.y - first.center.y);
  const scaleBase = Math.max(1e-9, (first.scale + second.scale) / 2);
  const scale = Math.abs(second.scale - first.scale) / scaleBase;
  const angle = angleDistance(first.angle, second.angle);
  const squaredCornerError = first.corners.reduce((sum, point, index) => {
    const other = second.corners[index];
    return sum + (point.x - other.x) ** 2 + (point.y - other.y) ** 2;
  }, 0);
  return { center, scale, angle, cornerRms: Math.sqrt(squaredCornerError / 4) };
}

function compatible(first: CardTrackSample, second: CardTrackSample): boolean {
  if (
    first.measurementReady !== second.measurementReady ||
    (!first.measurementReady && first.issue !== second.issue)
  ) return false;
  const delta = difference(first, second);
  return (
    delta.center <= CARD_TRACK_MAX_CENTER_SHIFT_RATIO &&
    delta.cornerRms <= CARD_TRACK_MAX_CENTER_SHIFT_RATIO &&
    delta.scale <= CARD_TRACK_MAX_SCALE_SHIFT &&
    delta.angle <= CARD_TRACK_MAX_ANGLE_SHIFT_RAD
  );
}

export class CardTracker {
  private samples: CardTrackSample[] = [];
  private anchor: CardTrackSample | null = null;
  private misses = 0;

  reset(): void {
    this.samples = [];
    this.anchor = null;
    this.misses = 0;
  }

  getStatus(): CardTrackingStatus {
    return this.status(false);
  }

  update(
    result: CardDetectionResult | null,
    frameWidth: number,
    frameHeight: number,
    timestamp: number
  ): CardTrackingStatus {
    const sample = createSample(result, frameWidth, frameHeight, timestamp);
    if (!sample) {
      this.misses++;
      const hadTrack = this.samples.length > 0;
      if (this.misses >= 2) this.reset();
      return this.status(hadTrack && this.samples.length === 0, result?.issue ?? 'not-found');
    }

    this.misses = 0;
    const previous = this.samples[this.samples.length - 1];
    if (
      previous &&
      sample.timestamp - previous.timestamp < CARD_TRACK_MIN_SAMPLE_INTERVAL_MS
    ) return this.status(false);
    if (previous && sample.timestamp - previous.timestamp > CARD_TRACK_MAX_GAP_MS) {
      this.samples = [sample];
      this.anchor = sample;
      return this.status(true);
    }
    if (
      (previous && !compatible(previous, sample)) ||
      (this.anchor && !compatible(this.anchor, sample))
    ) {
      this.samples = [sample];
      this.anchor = sample;
      return this.status(true);
    }

    if (!this.anchor) this.anchor = sample;
    this.samples = [...this.samples, sample].slice(-CARD_TRACK_TARGET_SAMPLES);
    return this.status(false);
  }

  private status(resetRequired: boolean, issueOverride?: CardTrackingStatus['issue']): CardTrackingStatus {
    let worstConsistency = 1;
    for (let index = 1; index < this.samples.length; index++) {
      const delta = difference(this.samples[index - 1], this.samples[index]);
      worstConsistency = Math.min(
        worstConsistency,
        1 - delta.center / CARD_TRACK_MAX_CENTER_SHIFT_RATIO,
        1 - delta.cornerRms / CARD_TRACK_MAX_CENTER_SHIFT_RATIO,
        1 - delta.scale / CARD_TRACK_MAX_SCALE_SHIFT,
        1 - delta.angle / CARD_TRACK_MAX_ANGLE_SHIFT_RAD
      );
    }
    const isStable = this.samples.length >= CARD_TRACK_TARGET_SAMPLES;
    return {
      sampleCount: this.samples.length,
      targetSamples: CARD_TRACK_TARGET_SAMPLES,
      isStable,
      consistency: this.samples.length > 0 ? clamp01(worstConsistency) : 0,
      issue:
        issueOverride ??
        this.samples[this.samples.length - 1]?.issue ??
        (isStable ? null : 'unstable'),
      resetRequired,
    };
  }
}
