import { CARD_HEIGHT_MM, CARD_WIDTH_MM } from '@/constants';
import type { HomographyTransform, Point } from '@/types';

const MATRIX_SIZE = 8;
const EPSILON = 1e-9;

function polygonArea(points: Point[]): number {
  let twiceArea = 0;
  for (let index = 0; index < points.length; index++) {
    const next = points[(index + 1) % points.length];
    twiceArea += points[index].x * next.y - next.x * points[index].y;
  }
  return Math.abs(twiceArea) / 2;
}

function distance(first: Point, second: Point): number {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

/** Solve an 8x8 system with partial pivoting. */
function solveLinearSystem(coefficients: number[][], values: number[]): number[] | null {
  const augmented = coefficients.map((row, index) => [...row, values[index]]);

  for (let column = 0; column < MATRIX_SIZE; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < MATRIX_SIZE; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivotRow][column])) {
        pivotRow = row;
      }
    }

    const pivot = augmented[pivotRow][column];
    if (!Number.isFinite(pivot) || Math.abs(pivot) < EPSILON) return null;
    [augmented[column], augmented[pivotRow]] = [augmented[pivotRow], augmented[column]];

    for (let valueIndex = column; valueIndex <= MATRIX_SIZE; valueIndex++) {
      augmented[column][valueIndex] /= pivot;
    }

    for (let row = 0; row < MATRIX_SIZE; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      if (Math.abs(factor) < EPSILON) continue;
      for (let valueIndex = column; valueIndex <= MATRIX_SIZE; valueIndex++) {
        augmented[row][valueIndex] -= factor * augmented[column][valueIndex];
      }
    }
  }

  const solution = augmented.map((row) => row[MATRIX_SIZE]);
  return solution.every(Number.isFinite) ? solution : null;
}

function targetRectangle(corners: Point[]): {
  points: Point[];
  orientation: HomographyTransform['cardOrientation'];
  widthMm: number;
  heightMm: number;
} {
  const horizontalLength = (distance(corners[0], corners[1]) + distance(corners[3], corners[2])) / 2;
  const verticalLength = (distance(corners[0], corners[3]) + distance(corners[1], corners[2])) / 2;
  const landscape = horizontalLength >= verticalLength;
  const widthMm = landscape ? CARD_WIDTH_MM : CARD_HEIGHT_MM;
  const heightMm = landscape ? CARD_HEIGHT_MM : CARD_WIDTH_MM;
  return {
    orientation: landscape ? 'landscape' : 'portrait',
    widthMm,
    heightMm,
    points: [
      { x: 0, y: 0 },
      { x: widthMm, y: 0 },
      { x: widthMm, y: heightMm },
      { x: 0, y: heightMm },
    ],
  };
}

/** Apply an image-to-plane homography. Returns null near the projective horizon. */
export function transformPoint(
  transform: HomographyTransform,
  point: Point
): Point | null {
  const [h11, h12, h13, h21, h22, h23, h31, h32, h33] = transform.matrix;
  const denominator = h31 * point.x + h32 * point.y + h33;
  if (!Number.isFinite(denominator) || Math.abs(denominator) < EPSILON) return null;
  const x = (h11 * point.x + h12 * point.y + h13) / denominator;
  const y = (h21 * point.x + h22 * point.y + h23) / denominator;
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

/**
 * Build an image-pixel to millimetre transform from ordered card corners.
 * Corners must be TL, TR, BR, BL. Long/short physical axes are selected for
 * both landscape and portrait cards; rotation of the output plane does not
 * affect Euclidean distances.
 */
export function createCardHomography(corners: Point[] | null): HomographyTransform | null {
  if (
    !corners ||
    corners.length !== 4 ||
    corners.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y)) ||
    polygonArea(corners) < 16 ||
    corners.some((point, index) => distance(point, corners[(index + 1) % 4]) < 2)
  ) {
    return null;
  }

  const target = targetRectangle(corners);
  const coefficients: number[][] = [];
  const values: number[] = [];
  for (let index = 0; index < 4; index++) {
    const { x, y } = corners[index];
    const { x: u, y: v } = target.points[index];
    coefficients.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    values.push(u);
    coefficients.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    values.push(v);
  }

  const solution = solveLinearSystem(coefficients, values);
  if (!solution) return null;
  const matrix: HomographyTransform['matrix'] = [
    solution[0], solution[1], solution[2],
    solution[3], solution[4], solution[5],
    solution[6], solution[7], 1,
  ];
  const provisional: HomographyTransform = {
    matrix,
    cardOrientation: target.orientation,
    destinationWidthMm: target.widthMm,
    destinationHeightMm: target.heightMm,
    reprojectionErrorMm: Number.POSITIVE_INFINITY,
  };

  let squaredError = 0;
  for (let index = 0; index < 4; index++) {
    const projected = transformPoint(provisional, corners[index]);
    if (!projected) return null;
    const error = distance(projected, target.points[index]);
    squaredError += error * error;
  }
  const reprojectionErrorMm = Math.sqrt(squaredError / 4);
  if (!Number.isFinite(reprojectionErrorMm) || reprojectionErrorMm > 0.1) return null;

  return { ...provisional, reprojectionErrorMm };
}

export function distanceOnPlaneMm(
  transform: HomographyTransform,
  first: Point,
  second: Point
): number | null {
  const transformedFirst = transformPoint(transform, first);
  const transformedSecond = transformPoint(transform, second);
  if (!transformedFirst || !transformedSecond) return null;
  const value = distance(transformedFirst, transformedSecond);
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function invertHomography(
  transform: HomographyTransform
): HomographyTransform | null {
  const [a, b, c, d, e, f, g, h, i] = transform.matrix;
  const cofactor00 = e * i - f * h;
  const cofactor01 = c * h - b * i;
  const cofactor02 = b * f - c * e;
  const cofactor10 = f * g - d * i;
  const cofactor11 = a * i - c * g;
  const cofactor12 = c * d - a * f;
  const cofactor20 = d * h - e * g;
  const cofactor21 = b * g - a * h;
  const cofactor22 = a * e - b * d;
  const determinant = a * cofactor00 + b * cofactor10 + c * cofactor20;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < EPSILON) return null;
  const inverse = [
    cofactor00, cofactor01, cofactor02,
    cofactor10, cofactor11, cofactor12,
    cofactor20, cofactor21, cofactor22,
  ].map((value) => value / determinant) as HomographyTransform['matrix'];
  return { ...transform, matrix: inverse };
}

export function cardPlaneCenter(transform: HomographyTransform): Point {
  return {
    x: transform.destinationWidthMm / 2,
    y: transform.destinationHeightMm / 2,
  };
}
