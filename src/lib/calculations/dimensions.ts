import {
  CalibrationMode,
  MeasurementResult,
  QualityStatus,
  ShoeSizeRecommendation,
} from '@/types';
import { CARD_WIDTH_MM, SHOE_SIZE_TABLE, ShoeSizeEntry } from '@/constants';

/**
 * Calculate pixels-per-millimeter ratio from detected card width in pixels.
 * PPM = detectedCardWidthPx / CARD_WIDTH_MM
 */
export function calculatePPM(detectedCardWidthPx: number): number {
  if (detectedCardWidthPx <= 0) return 0;
  return detectedCardWidthPx / CARD_WIDTH_MM;
}

/**
 * Convert pixel distance to centimeters using PPM ratio.
 * cm = (px / PPM) / 10
 */
export function pixelsToCm(px: number, ppm: number): number {
  if (ppm <= 0 || px <= 0) return 0;
  return (px / ppm) / 10;
}

/**
 * Euclidean distance between two points.
 */
export function euclideanDistance(
  p1: { x: number; y: number },
  p2: { x: number; y: number }
): number {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Find the closest shoe size entry for a given foot length in mm.
 */
export function findShoeSizeEntry(footLengthMm: number): ShoeSizeEntry | null {
  if (footLengthMm <= 0 || SHOE_SIZE_TABLE.length === 0) return null;
  const minimumLength = SHOE_SIZE_TABLE[0].lengthMm;
  const maximumLength = SHOE_SIZE_TABLE[SHOE_SIZE_TABLE.length - 1].lengthMm;
  if (footLengthMm < minimumLength || footLengthMm > maximumLength) return null;

  let closest: ShoeSizeEntry = SHOE_SIZE_TABLE[0];
  let minDiff = Math.abs(footLengthMm - SHOE_SIZE_TABLE[0].lengthMm);

  for (const entry of SHOE_SIZE_TABLE) {
    const diff = Math.abs(footLengthMm - entry.lengthMm);
    if (diff < minDiff) {
      minDiff = diff;
      closest = entry;
    }
  }
  return closest;
}

/**
 * Determine fit description based on foot length vs table entry.
 */
function determineFit(footLengthMm: number, entryLengthMm: number): ShoeSizeRecommendation['fit'] {
  const diff = footLengthMm - entryLengthMm;
  if (Math.abs(diff) <= 2.5) return 'exact';
  if (diff > 0) return 'half-size-up';
  return 'half-size-down';
}

/**
 * Build a full MeasurementResult from pixel dimensions and PPM.
 */
export function buildMeasurementResult(
  lengthPx: number,
  widthPx: number,
  ppm: number
): MeasurementResult | null {
  if (ppm <= 0 || lengthPx <= 0 || widthPx <= 0) return null;

  return buildMeasurementResultFromMillimeters(lengthPx / ppm, widthPx / ppm, ppm, {
    confidence: 0.4,
    quality: 'low',
    calibrationMode: 'ppm-fallback',
    homographyValid: false,
    warnings: ['ใช้สเกล PPM สำรอง จึงยังบันทึกผลหรือแนะนำไซซ์รองเท้าไม่ได้'],
  });
}

export interface MeasurementQualityInput {
  confidence: number;
  quality: QualityStatus;
  calibrationMode: CalibrationMode;
  homographyValid: boolean;
  warnings?: string[];
}

/** Build a result from temporally aggregated, already-calibrated dimensions. */
export function buildMeasurementResultFromMillimeters(
  footLengthMm: number,
  footWidthMm: number,
  ppm: number,
  qualityInput: MeasurementQualityInput = {
    confidence: 0.4,
    quality: 'low',
    calibrationMode: 'ppm-fallback',
    homographyValid: false,
    warnings: ['ยังไม่มี homography ที่ผ่านเกณฑ์'],
  }
): MeasurementResult | null {
  if (ppm <= 0 || footLengthMm <= 0 || footWidthMm <= 0) return null;

  const footLengthCm = footLengthMm / 10;
  const footWidthCm = footWidthMm / 10;

  const entry = findShoeSizeEntry(footLengthMm);

  const shoeSize: ShoeSizeRecommendation | null =
    qualityInput.quality === 'low' || !qualityInput.homographyValid || !entry
      ? null
      : {
    eu: entry.eu,
    usMen: entry.usMen,
    usWomen: entry.usWomen,
    uk: entry.uk,
    fit: determineFit(footLengthMm, entry.lengthMm),
  };

  return {
    footLengthCm: Math.round(footLengthCm * 10) / 10,
    footWidthCm: Math.round(footWidthCm * 10) / 10,
    ppm,
    timestamp: Date.now(),
    confidence: Math.max(0, Math.min(1, qualityInput.confidence)),
    quality: qualityInput.quality,
    calibrationMode: qualityInput.calibrationMode,
    homographyValid: qualityInput.homographyValid,
    warnings: [
      ...(qualityInput.warnings ?? []),
      ...(!entry ? ['ความยาวอยู่นอกช่วงตารางไซซ์รองเท้าที่รองรับ'] : []),
    ],
    shoeSize,
  };
}

/**
 * Validate that a PPM value is within a realistic range.
 * At typical phone camera distances (30–60 cm), a credit card width of 85.6 mm
 * should produce between 3 and 50 pixels/mm.
 */
export function isValidPPM(ppm: number): boolean {
  return ppm >= 1.5 && ppm <= 80;
}

/**
 * Round a number to N decimal places.
 */
export function round(value: number, decimals = 1): number {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}
