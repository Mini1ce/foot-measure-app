import type { FootContourResult, HomographyTransform, Point } from '@/types';
import { invertHomography, transformPoint } from './homography';

interface ProjectedPoint {
  image: Point;
  plane: Point;
  major: number;
  minor: number;
}

interface WidthSlice {
  centerMajor: number;
  normalizedPosition: number;
  width: number;
  minimumMinor: number;
  maximumMinor: number;
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

function percentile(values: number[], amount: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = clamp01(amount) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function distance(first: Point, second: Point): number {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function emptyResult(): FootContourResult {
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

function covarianceAxis(points: Point[]): { center: Point; majorAxis: Point; minorAxis: Point } {
  const center = points.reduce(
    (sum, point) => ({
      x: sum.x + point.x / points.length,
      y: sum.y + point.y / points.length,
    }),
    { x: 0, y: 0 }
  );
  let covarianceXX = 0;
  let covarianceYY = 0;
  let covarianceXY = 0;
  for (const point of points) {
    const x = point.x - center.x;
    const y = point.y - center.y;
    covarianceXX += x * x;
    covarianceYY += y * y;
    covarianceXY += x * y;
  }
  const angle = 0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY);
  const majorAxis = { x: Math.cos(angle), y: Math.sin(angle) };
  return {
    center,
    majorAxis,
    minorAxis: { x: -majorAxis.y, y: majorAxis.x },
  };
}

function buildWidthSlices(
  projectedContour: ProjectedPoint[],
  minimumMajor: number,
  length: number,
  binCount: number
): WidthSlice[] {
  const slices: WidthSlice[] = [];
  for (let bin = 0; bin < binCount; bin++) {
    const centerMajor = minimumMajor + (length * (bin + 0.5)) / binCount;
    const intersections: number[] = [];
    for (let index = 0; index < projectedContour.length; index++) {
      const first = projectedContour[index];
      const second = projectedContour[(index + 1) % projectedContour.length];
      const crosses =
        (first.major <= centerMajor && second.major > centerMajor) ||
        (second.major <= centerMajor && first.major > centerMajor);
      if (!crosses) continue;
      const amount = (centerMajor - first.major) / (second.major - first.major);
      intersections.push(first.minor + (second.minor - first.minor) * amount);
    }
    if (intersections.length < 2) continue;
    const minimumMinor = Math.min(...intersections);
    const maximumMinor = Math.max(...intersections);
    const width = maximumMinor - minimumMinor;
    if (width <= 0) continue;
    slices.push({
      centerMajor,
      normalizedPosition: (bin + 0.5) / binCount,
      width,
      minimumMinor,
      maximumMinor,
    });
  }
  return slices;
}

function endWidth(slices: WidthSlice[], start: number, end: number): number {
  return median(
    slices
      .filter((slice) => slice.normalizedPosition >= start && slice.normalizedPosition <= end)
      .map((slice) => slice.width)
  );
}

/**
 * Measure a contour in either image space or the card's millimetre plane.
 * Width is the upper quartile of sustained perpendicular cross-sections in
 * the 32–82% heel-relative region. This excludes heel/toe end effects and is
 * deliberately not the single largest mask slice.
 */
export function measureFootGeometry(
  imagePoints: Point[],
  homography: HomographyTransform | null = null
): FootContourResult {
  if (imagePoints.length < 12) return emptyResult();

  const analysisPoints: { image: Point; plane: Point }[] = [];
  for (const image of imagePoints) {
    const plane = homography ? transformPoint(homography, image) : image;
    if (!plane || !Number.isFinite(plane.x) || !Number.isFinite(plane.y)) {
      return emptyResult();
    }
    analysisPoints.push({ image, plane });
  }

  const axes = covarianceAxis(analysisPoints.map((item) => item.plane));
  const projectedContour: ProjectedPoint[] = analysisPoints.map(({ image, plane }) => {
    const relativeX = plane.x - axes.center.x;
    const relativeY = plane.y - axes.center.y;
    return {
      image,
      plane,
      major: relativeX * axes.majorAxis.x + relativeY * axes.majorAxis.y,
      minor: relativeX * axes.minorAxis.x + relativeY * axes.minorAxis.y,
    };
  });
  const projected = [...projectedContour].sort((first, second) => first.major - second.major);

  const endpointTrim = projected.length >= 100 ? Math.floor(projected.length * 0.005) : 0;
  const firstEnd = projected[endpointTrim];
  const secondEnd = projected[projected.length - 1 - endpointTrim];
  const planeLength = secondEnd.major - firstEnd.major;
  if (!Number.isFinite(planeLength) || planeLength <= 0) return emptyResult();

  const binCount = 36;
  const slices = buildWidthSlices(projectedContour, firstEnd.major, planeLength, binCount);
  if (slices.length < 12) return emptyResult();

  // The forefoot is normally wider than the heel. If both ends are nearly
  // symmetric, either direction yields the same robust central-width result.
  const firstSideWidth = endWidth(slices, 0.08, 0.28);
  const secondSideWidth = endWidth(slices, 0.72, 0.92);
  const toeAtFirstEnd = firstSideWidth > secondSideWidth;
  const heelEnd = toeAtFirstEnd ? secondEnd : firstEnd;
  const toeEnd = toeAtFirstEnd ? firstEnd : secondEnd;
  const planeLengthPoints: [Point, Point] = [heelEnd.major, toeEnd.major]
    .map((major) => ({
      x: axes.center.x + axes.majorAxis.x * major,
      y: axes.center.y + axes.majorAxis.y * major,
    })) as [Point, Point];

  const candidateSlices = slices.filter((slice) => {
    const heelRelative = toeAtFirstEnd
      ? 1 - slice.normalizedPosition
      : slice.normalizedPosition;
    return heelRelative >= 0.32 && heelRelative <= 0.82;
  });
  if (candidateSlices.length < 8) return emptyResult();

  const candidateWidths = candidateSlices.map((slice) => slice.width);
  const representativeWidth = percentile(candidateWidths, 0.75);
  const selectedSlice = candidateSlices.reduce((best, slice) =>
    Math.abs(slice.width - representativeWidth) < Math.abs(best.width - representativeWidth)
      ? slice
      : best
  );
  const planeWidthPoints: [Point, Point] = [selectedSlice.minimumMinor, selectedSlice.maximumMinor]
    .map((minor) => ({
      x: axes.center.x + axes.majorAxis.x * selectedSlice.centerMajor + axes.minorAxis.x * minor,
      y: axes.center.y + axes.majorAxis.y * selectedSlice.centerMajor + axes.minorAxis.y * minor,
    })) as [Point, Point];
  let widthImagePoints = planeWidthPoints;
  let lengthImagePoints = planeLengthPoints;
  if (homography) {
    const inverse = invertHomography(homography);
    if (!inverse) return emptyResult();
    const first = transformPoint(inverse, planeWidthPoints[0]);
    const second = transformPoint(inverse, planeWidthPoints[1]);
    if (!first || !second) return emptyResult();
    widthImagePoints = [first, second];
    const heelImage = transformPoint(inverse, planeLengthPoints[0]);
    const toeImage = transformPoint(inverse, planeLengthPoints[1]);
    if (!heelImage || !toeImage) return emptyResult();
    lengthImagePoints = [heelImage, toeImage];
  }
  const widthToLengthRatio = representativeWidth / planeLength;
  if (widthToLengthRatio < 0.16 || widthToLengthRatio > 0.62) return emptyResult();

  const medianWidth = median(candidateWidths);
  const medianAbsoluteDeviation = median(
    candidateWidths.map((value) => Math.abs(value - medianWidth))
  );
  const profileConsistency = medianWidth > 0
    ? clamp01(1 - medianAbsoluteDeviation / (medianWidth * 0.28))
    : 0;
  const sliceCoverage = clamp01(candidateSlices.length / 18);
  const geometryConfidence = clamp01(profileConsistency * 0.72 + sliceCoverage * 0.28);

  const signature: number[] = [];
  for (let segment = 0; segment < 10; segment++) {
    const start = segment / 10;
    const end = (segment + 1) / 10;
    const values = slices
      .filter((slice) => slice.normalizedPosition >= start && slice.normalizedPosition < end)
      .map((slice) => slice.width / planeLength);
    signature.push(values.length > 0 ? median(values) : 0);
  }
  if (toeAtFirstEnd) signature.reverse();

  const sampleStep = Math.max(1, Math.ceil(imagePoints.length / 500));
  const contourPoints = imagePoints.filter((_, index) => index % sampleStep === 0);
  return {
    detected: true,
    lengthPx: distance(lengthImagePoints[0], lengthImagePoints[1]),
    widthPx: distance(widthImagePoints[0], widthImagePoints[1]),
    contourPoints,
    heelPoint: lengthImagePoints[0],
    toePoint: lengthImagePoints[1],
    widestPoints: widthImagePoints,
    geometryConfidence,
    shapeSignature: signature,
  };
}
