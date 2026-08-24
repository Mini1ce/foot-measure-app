import type { FootROI, LetterboxPlacement } from '@/types';
import { FOOT_SEGMENTER_IMAGE_SIZE } from '@/constants';

export interface PreparedFootInput {
  canvas: HTMLCanvasElement;
  roi: FootROI;
  placement: LetterboxPlacement;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

/** Match TensorFlow resize-with-pad dimensions without stretching the foot. */
export function calculateLetterboxPlacement(
  sourceWidth: number,
  sourceHeight: number,
  targetSize = FOOT_SEGMENTER_IMAGE_SIZE
): LetterboxPlacement | null {
  if (
    !Number.isFinite(sourceWidth) ||
    !Number.isFinite(sourceHeight) ||
    !Number.isFinite(targetSize) ||
    sourceWidth <= 0 ||
    sourceHeight <= 0 ||
    targetSize <= 0
  ) {
    return null;
  }

  const scale = Math.min(targetSize / sourceWidth, targetSize / sourceHeight);
  const width = Math.max(1, Math.floor(sourceWidth * scale));
  const height = Math.max(1, Math.floor(sourceHeight * scale));
  return {
    x: Math.floor((targetSize - width) / 2),
    y: Math.floor((targetSize - height) / 2),
    width,
    height,
    targetSize,
  };
}

/** Crop the camera ROI and place it on the exact square canvas expected by the model. */
export function prepareFootSegmenterInput(
  source: CanvasImageSource,
  reusableCanvas: HTMLCanvasElement,
  roi: FootROI,
  frameWidth: number,
  frameHeight: number
): PreparedFootInput | null {
  const roiX = clamp(Math.floor(roi.x), 0, frameWidth);
  const roiY = clamp(Math.floor(roi.y), 0, frameHeight);
  const roiWidth = clamp(Math.ceil(roi.width), 0, frameWidth - roiX);
  const roiHeight = clamp(Math.ceil(roi.height), 0, frameHeight - roiY);
  const placement = calculateLetterboxPlacement(roiWidth, roiHeight);
  if (!placement || roiWidth <= 0 || roiHeight <= 0) return null;

  if (reusableCanvas.width !== placement.targetSize) {
    reusableCanvas.width = placement.targetSize;
  }
  if (reusableCanvas.height !== placement.targetSize) {
    reusableCanvas.height = placement.targetSize;
  }
  const context = reusableCanvas.getContext('2d', { willReadFrequently: false });
  if (!context) return null;

  context.save();
  context.fillStyle = 'rgb(0, 0, 0)';
  context.fillRect(0, 0, placement.targetSize, placement.targetSize);
  context.drawImage(
    source,
    roiX,
    roiY,
    roiWidth,
    roiHeight,
    placement.x,
    placement.y,
    placement.width,
    placement.height
  );
  context.restore();

  return {
    canvas: reusableCanvas,
    roi: { x: roiX, y: roiY, width: roiWidth, height: roiHeight },
    placement,
  };
}
