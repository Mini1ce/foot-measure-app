import type { FootROI, Point } from '@/types';
import {
  CARD_ASPECT_RATIO,
  CARD_GUIDE_LAYOUT,
  FOOT_GUIDE_RECT,
} from '@/constants';

export interface ObjectCoverGeometry {
  scale: number;
  offsetX: number;
  offsetY: number;
  sourceWidth: number;
  sourceHeight: number;
  displayWidth: number;
  displayHeight: number;
  visibleSource: FootROI;
}

export interface DisplayGuideRects {
  foot: FootROI;
  card: FootROI;
  cardLandscape: FootROI;
  cardPortrait: FootROI;
}

/**
 * Normalized display-space guides. The card height is derived from display
 * pixels so its visible outline always keeps the physical ID-1 aspect ratio.
 */
export function calculateDisplayGuideRects(
  displayWidth: number,
  displayHeight: number
): DisplayGuideRects | null {
  if (
    !Number.isFinite(displayWidth) ||
    !Number.isFinite(displayHeight) ||
    displayWidth <= 0 ||
    displayHeight <= 0
  ) return null;

  const cardWidth = CARD_GUIDE_LAYOUT.width;
  const squareHeight = cardWidth * displayWidth / displayHeight;
  const landscapeHeight = squareHeight / CARD_ASPECT_RATIO;
  const portraitWidth = cardWidth / CARD_ASPECT_RATIO;
  const squareY = 1 - CARD_GUIDE_LAYOUT.bottom - squareHeight;
  if (squareHeight <= 0 || squareY < 0) return null;
  return {
    foot: { ...FOOT_GUIDE_RECT },
    card: {
      x: CARD_GUIDE_LAYOUT.x,
      y: squareY,
      width: cardWidth,
      height: squareHeight,
    },
    cardLandscape: {
      x: CARD_GUIDE_LAYOUT.x,
      y: 1 - CARD_GUIDE_LAYOUT.bottom - landscapeHeight,
      width: cardWidth,
      height: landscapeHeight,
    },
    cardPortrait: {
      x: CARD_GUIDE_LAYOUT.x + cardWidth - portraitWidth,
      y: squareY,
      width: portraitWidth,
      height: squareHeight,
    },
  };
}

export function calculateObjectCoverGeometry(
  sourceWidth: number,
  sourceHeight: number,
  displayWidth: number,
  displayHeight: number
): ObjectCoverGeometry | null {
  if (
    ![sourceWidth, sourceHeight, displayWidth, displayHeight].every(
      (value) => Number.isFinite(value) && value > 0
    )
  ) return null;

  const scale = Math.max(displayWidth / sourceWidth, displayHeight / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const offsetX = (displayWidth - renderedWidth) / 2;
  const offsetY = (displayHeight - renderedHeight) / 2;
  return {
    scale,
    offsetX,
    offsetY,
    sourceWidth,
    sourceHeight,
    displayWidth,
    displayHeight,
    visibleSource: {
      x: -offsetX / scale,
      y: -offsetY / scale,
      width: displayWidth / scale,
      height: displayHeight / scale,
    },
  };
}

export function sourcePointToDisplay(
  geometry: ObjectCoverGeometry,
  point: Point
): Point {
  return {
    x: point.x * geometry.scale + geometry.offsetX,
    y: point.y * geometry.scale + geometry.offsetY,
  };
}

/** Map a normalized rectangle drawn over the visible video back to source pixels. */
export function displayRectToSource(
  geometry: ObjectCoverGeometry,
  normalizedRect: FootROI
): FootROI {
  const displayX = normalizedRect.x * geometry.displayWidth;
  const displayY = normalizedRect.y * geometry.displayHeight;
  const displayWidth = normalizedRect.width * geometry.displayWidth;
  const displayHeight = normalizedRect.height * geometry.displayHeight;
  const x = Math.max(0, (displayX - geometry.offsetX) / geometry.scale);
  const y = Math.max(0, (displayY - geometry.offsetY) / geometry.scale);
  const right = Math.min(
    geometry.sourceWidth,
    (displayX + displayWidth - geometry.offsetX) / geometry.scale
  );
  const bottom = Math.min(
    geometry.sourceHeight,
    (displayY + displayHeight - geometry.offsetY) / geometry.scale
  );
  return { x, y, width: Math.max(0, right - x), height: Math.max(0, bottom - y) };
}
