import { PoseLandmark, FootROI } from '@/types';
import { POSE_LANDMARKS, FOOT_ROI_PADDING } from '@/constants';

/**
 * Extract foot landmarks from MediaPipe Pose result.
 * Returns left foot landmarks by default.
 */
export function extractFootLandmarks(
  landmarks: PoseLandmark[],
  side: 'left' | 'right' = 'left'
): {
  ankle: PoseLandmark | null;
  heel: PoseLandmark | null;
  footIndex: PoseLandmark | null;
} {
  const isLeft = side === 'left';
  const ankleIdx = isLeft ? POSE_LANDMARKS.LEFT_ANKLE : POSE_LANDMARKS.RIGHT_ANKLE;
  const heelIdx = isLeft ? POSE_LANDMARKS.LEFT_HEEL : POSE_LANDMARKS.RIGHT_HEEL;
  const footIndexIdx = isLeft ? POSE_LANDMARKS.LEFT_FOOT_INDEX : POSE_LANDMARKS.RIGHT_FOOT_INDEX;

  const ankle = landmarks[ankleIdx] ?? null;
  const heel = landmarks[heelIdx] ?? null;
  const footIndex = landmarks[footIndexIdx] ?? null;

  return { ankle, heel, footIndex };
}

/**
 * Determine which foot is more visible based on landmark visibility scores.
 */
export function determineBetterFoot(landmarks: PoseLandmark[]): 'left' | 'right' {
  const leftVisibility =
    (landmarks[POSE_LANDMARKS.LEFT_HEEL]?.visibility ?? 0) +
    (landmarks[POSE_LANDMARKS.LEFT_FOOT_INDEX]?.visibility ?? 0);
  const rightVisibility =
    (landmarks[POSE_LANDMARKS.RIGHT_HEEL]?.visibility ?? 0) +
    (landmarks[POSE_LANDMARKS.RIGHT_FOOT_INDEX]?.visibility ?? 0);

  return leftVisibility >= rightVisibility ? 'left' : 'right';
}

/**
 * Calculate a padded Region of Interest around foot landmarks.
 * Landmarks are normalized (0–1), so we scale by frame dimensions.
 */
export function calculateFootROI(
  landmarks: PoseLandmark[],
  frameWidth: number,
  frameHeight: number,
  side: 'left' | 'right' = 'left'
): FootROI | null {
  const { ankle, heel, footIndex } = extractFootLandmarks(landmarks, side);

  if (!ankle || !heel || !footIndex) return null;

  // Check minimum visibility threshold
  const minVis = 0.4;
  if (
    (ankle.visibility ?? 0) < minVis ||
    (heel.visibility ?? 0) < minVis ||
    (footIndex.visibility ?? 0) < minVis
  ) {
    return null;
  }

  // Convert normalized coords to pixel coords
  const points = [ankle, heel, footIndex].map((lm) => ({
    x: lm.x * frameWidth,
    y: lm.y * frameHeight,
  }));

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const padX = FOOT_ROI_PADDING * frameWidth;
  const padY = FOOT_ROI_PADDING * frameHeight;

  // Expand ROI with extra space for the full foot
  const expansionFactor = 1.5;
  const rawW = (maxX - minX) * expansionFactor;
  const rawH = (maxY - minY) * expansionFactor;

  return {
    x: Math.max(0, minX - padX - rawW * 0.25),
    y: Math.max(0, minY - padY - rawH * 0.5),
    width: Math.min(frameWidth, rawW + padX * 2),
    height: Math.min(frameHeight, rawH + padY * 2),
  };
}

/**
 * Check if foot landmarks have sufficient confidence for processing.
 */
export function isFootVisible(landmarks: PoseLandmark[], side: 'left' | 'right'): boolean {
  const { ankle, heel, footIndex } = extractFootLandmarks(landmarks, side);
  if (!ankle || !heel || !footIndex) return false;
  return (
    (ankle.visibility ?? 0) > 0.5 &&
    (heel.visibility ?? 0) > 0.5 &&
    (footIndex.visibility ?? 0) > 0.5
  );
}

/**
 * Guide-aligned ROI used when Pose Landmarker cannot see an isolated top-down
 * foot. Values mirror the on-camera placement guide while leaving the card
 * area on the right mostly outside the search region.
 */
export function calculateFallbackFootROI(
  frameWidth: number,
  frameHeight: number
): FootROI {
  return {
    x: frameWidth * 0.08,
    y: frameHeight * 0.04,
    width: frameWidth * 0.68,
    height: frameHeight * 0.91,
  };
}
