/* eslint-disable @typescript-eslint/no-explicit-any -- OpenCV.js exposes an untyped CDN runtime. */
import { CardDetectionResult, CardIssue, FootROI, Point, QualityStatus } from '@/types';
import {
  CARD_ASPECT_RATIO,
  CARD_CONFIDENCE_THRESHOLD,
  CARD_HEIGHT_MM,
  CARD_MAX_MEASUREMENT_AREA_RATIO,
  CARD_MAX_AREA_RATIO,
  CARD_MIN_EDGE_CONSISTENCY,
  CARD_MIN_AREA_RATIO,
  CARD_MIN_MEASUREMENT_AREA_RATIO,
  CARD_MIN_PERSPECTIVE_SCORE,
  CARD_WIDTH_MM,
} from '@/constants';
import { createCardHomography } from '@/lib/calculations/homography';

declare const cv: any;

interface CardCandidate {
  confidence: number;
  widthPx: number;
  heightPx: number;
  corners: Point[];
  ppm: number;
  homography: CardDetectionResult['homography'];
  measurementReady: boolean;
  quality: QualityStatus;
  issue: CardIssue;
  edgeConsistency: number;
  perspectiveScore: number;
  areaRatio: number;
}

export interface CardDetectionOptions {
  guideRect?: FootROI | null;
}

const EMPTY_RESULT: CardDetectionResult = {
  detected: false,
  confidence: 0,
  widthPx: 0,
  heightPx: 0,
  corners: null,
  ppm: 0,
  homography: null,
  measurementReady: false,
  quality: 'low',
  issue: 'not-found',
  edgeConsistency: 0,
  perspectiveScore: 0,
  areaRatio: 0,
};

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * Order a convex quad clockwise as [top-left, top-right, bottom-right,
 * bottom-left]. Sorting around the centroid remains stable when the card is
 * rotated; splitting points into "top" and "bottom" does not.
 */
function orderCorners(points: Point[]): Point[] {
  const center = points.reduce(
    (sum, point) => ({ x: sum.x + point.x / points.length, y: sum.y + point.y / points.length }),
    { x: 0, y: 0 }
  );

  const clockwise = [...points].sort(
    (a, b) =>
      Math.atan2(a.y - center.y, a.x - center.x) -
      Math.atan2(b.y - center.y, b.x - center.x)
  );
  const topLeftIndex = clockwise.reduce(
    (best, point, index) =>
      point.x + point.y < clockwise[best].x + clockwise[best].y ? index : best,
    0
  );
  const ordered = [
    ...clockwise.slice(topLeftIndex),
    ...clockwise.slice(0, topLeftIndex),
  ];

  // In image coordinates a TL → TR → BR → BL polygon has positive area.
  if (signedPolygonArea(ordered) < 0) {
    return [ordered[0], ordered[3], ordered[2], ordered[1]];
  }
  return ordered;
}

function signedPolygonArea(points: Point[]): number {
  let twiceArea = 0;
  for (let i = 0; i < points.length; i++) {
    const next = points[(i + 1) % points.length];
    twiceArea += points[i].x * next.y - next.x * points[i].y;
  }
  return twiceArea / 2;
}

function isConvexQuad(points: Point[]): boolean {
  let direction = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const c = points[(i + 2) % points.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) < 1e-6) return false;
    const currentDirection = Math.sign(cross);
    if (direction !== 0 && currentDirection !== direction) return false;
    direction = currentDirection;
  }
  return true;
}

function edgeLengths(corners: Point[]): {
  top: number;
  right: number;
  bottom: number;
  left: number;
  width: number;
  height: number;
} {
  const top = Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y);
  const right = Math.hypot(corners[2].x - corners[1].x, corners[2].y - corners[1].y);
  const bottom = Math.hypot(corners[2].x - corners[3].x, corners[2].y - corners[3].y);
  const left = Math.hypot(corners[3].x - corners[0].x, corners[3].y - corners[0].y);
  return { top, right, bottom, left, width: (top + bottom) / 2, height: (left + right) / 2 };
}

/** Score how close the quad's corners are to right angles. */
function rightAngleScore(corners: Point[]): { score: number; maxCosine: number } {
  let cosineSum = 0;
  let maxCosine = 0;

  for (let i = 0; i < corners.length; i++) {
    const previous = corners[(i + corners.length - 1) % corners.length];
    const current = corners[i];
    const next = corners[(i + 1) % corners.length];
    const ax = previous.x - current.x;
    const ay = previous.y - current.y;
    const bx = next.x - current.x;
    const by = next.y - current.y;
    const denominator = Math.hypot(ax, ay) * Math.hypot(bx, by);
    if (denominator === 0) return { score: 0, maxCosine: 1 };
    const cosine = Math.abs((ax * bx + ay * by) / denominator);
    cosineSum += cosine;
    maxCosine = Math.max(maxCosine, cosine);
  }

  return {
    score: clamp01(1 - cosineSum / corners.length / 0.55),
    maxCosine,
  };
}

/**
 * Use both physical card axes when calculating scale. This works for landscape
 * and portrait cards and is less sensitive to perspective on one edge.
 */
function calculateCardPPM(widthPx: number, heightPx: number): number {
  const longEdgePx = Math.max(widthPx, heightPx);
  const shortEdgePx = Math.min(widthPx, heightPx);
  return (longEdgePx / CARD_WIDTH_MM + shortEdgePx / CARD_HEIGHT_MM) / 2;
}

function diagonalConsistency(corners: Point[]): number {
  const first = Math.hypot(corners[2].x - corners[0].x, corners[2].y - corners[0].y);
  const second = Math.hypot(corners[3].x - corners[1].x, corners[3].y - corners[1].y);
  return Math.min(first, second) / Math.max(first, second);
}

/** Require each physical edge to be continuously supported by Canny evidence. */
function edgeSupportScore(edgeEvidence: any, corners: Point[]): number {
  const perEdge: number[] = [];
  const data = edgeEvidence.data as Uint8Array;
  for (let edge = 0; edge < 4; edge++) {
    const start = corners[edge];
    const end = corners[(edge + 1) % 4];
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    const samples = Math.max(12, Math.ceil(length / 4));
    let supported = 0;
    for (let sample = 0; sample <= samples; sample++) {
      const amount = sample / samples;
      const x = Math.round(start.x + (end.x - start.x) * amount);
      const y = Math.round(start.y + (end.y - start.y) * amount);
      let found = false;
      for (let offsetY = -3; offsetY <= 3 && !found; offsetY++) {
        for (let offsetX = -3; offsetX <= 3; offsetX++) {
          const sampleX = x + offsetX;
          const sampleY = y + offsetY;
          if (
            sampleX >= 0 &&
            sampleX < edgeEvidence.cols &&
            sampleY >= 0 &&
            sampleY < edgeEvidence.rows &&
            data[sampleY * edgeEvidence.cols + sampleX] > 0
          ) {
            found = true;
            break;
          }
        }
      }
      if (found) supported++;
    }
    perEdge.push(supported / (samples + 1));
  }
  return Math.min(...perEdge);
}

function isInsideGuide(corners: Point[], guideRect?: FootROI | null): boolean {
  if (!guideRect) return true;
  const marginX = guideRect.width * 0.16;
  const marginY = guideRect.height * 0.16;
  return corners.every((point) => (
    point.x >= guideRect.x - marginX &&
    point.x <= guideRect.x + guideRect.width + marginX &&
    point.y >= guideRect.y - marginY &&
    point.y <= guideRect.y + guideRect.height + marginY
  ));
}

function candidateFromQuad(
  rawPoints: Point[],
  contourArea: number,
  frameArea: number,
  edgeEvidence: any,
  guideRect?: FootROI | null
): CardCandidate | null {
  const corners = orderCorners(rawPoints);
  if (!isConvexQuad(corners)) return null;

  const quadArea = Math.abs(signedPolygonArea(corners));
  if (quadArea <= 0) return null;

  const { top, right, bottom, left, width, height } = edgeLengths(corners);
  const shortEdge = Math.min(width, height);
  if (shortEdge <= 0) return null;

  const measuredAspect = Math.max(width, height) / shortEdge;
  const aspectError = Math.abs(measuredAspect - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO;
  // A top-down capture may have mild perspective, but a square or long strip is
  // not an ID-1 card.
  if (aspectError > 0.32) return null;

  const angles = rightAngleScore(corners);
  if (angles.maxCosine > 0.7) return null;

  const areaAgreement = Math.min(contourArea, quadArea) / Math.max(contourArea, quadArea);
  if (areaAgreement < 0.58) return null;

  const oppositeEdgeConsistency =
    (Math.min(top, bottom) / Math.max(top, bottom) +
      Math.min(left, right) / Math.max(left, right)) /
    2;
  if (oppositeEdgeConsistency < 0.5) return null;

  const edgeConsistency = edgeSupportScore(edgeEvidence, corners);
  const diagonalScore = diagonalConsistency(corners);
  const perspectiveScore = clamp01(
    oppositeEdgeConsistency * 0.5 + angles.score * 0.3 + diagonalScore * 0.2
  );

  const aspectScore = clamp01(1 - aspectError / 0.28);
  const areaRatio = contourArea / frameArea;
  const areaScore =
    clamp01(areaRatio / 0.035) * (areaRatio > 0.22 ? clamp01(1 - (areaRatio - 0.22) / 0.3) : 1);
  const guideOk = isInsideGuide(corners, guideRect);
  const confidence = clamp01(
    aspectScore * 0.3 +
      angles.score * 0.14 +
      areaAgreement * 0.12 +
      oppositeEdgeConsistency * 0.12 +
      edgeConsistency * 0.22 +
      areaScore * 0.06 +
      (guideOk ? 1 : 0.25) * 0.04
  );
  const homography = createCardHomography(corners);
  let issue: CardIssue = null;
  if (!guideOk) issue = 'outside-guide';
  else if (areaRatio < CARD_MIN_MEASUREMENT_AREA_RATIO) issue = 'too-small';
  else if (areaRatio > CARD_MAX_MEASUREMENT_AREA_RATIO) issue = 'too-large';
  else if (edgeConsistency < CARD_MIN_EDGE_CONSISTENCY) issue = 'partially-occluded';
  else if (aspectError > 0.1) issue = 'wrong-shape';
  else if (
    angles.maxCosine > 0.45 ||
    oppositeEdgeConsistency < 0.67 ||
    perspectiveScore < CARD_MIN_PERSPECTIVE_SCORE
  ) issue = 'too-tilted';
  else if (!homography) issue = 'homography-failed';
  else if (confidence < CARD_CONFIDENCE_THRESHOLD) issue = 'wrong-shape';
  const measurementReady = issue === null;
  const quality: QualityStatus = measurementReady
    ? confidence >= 0.8 && perspectiveScore >= 0.78 && edgeConsistency >= 0.8
      ? 'high'
      : 'medium'
    : 'low';

  return {
    confidence,
    widthPx: width,
    heightPx: height,
    corners,
    ppm: calculateCardPPM(width, height),
    homography,
    measurementReady,
    quality,
    issue,
    edgeConsistency,
    perspectiveScore,
    areaRatio,
  };
}

/** Search one binary image for the strongest rectangular card candidate. */
function findBestCandidate(
  binary: any,
  edgeEvidence: any,
  frameArea: number,
  retrievalMode: number,
  guideRect?: FootROI | null
): CardCandidate | null {
  let contours: any = null;
  let hierarchy: any = null;
  let best: CardCandidate | null = null;

  try {
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(binary, contours, hierarchy, retrievalMode, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      try {
        const area = Math.abs(cv.contourArea(contour));
        if (
          area < frameArea * CARD_MIN_AREA_RATIO ||
          area > frameArea * CARD_MAX_AREA_RATIO
        ) {
          continue;
        }

        const perimeter = cv.arcLength(contour, true);
        if (perimeter <= 0) continue;

        let quad: any = null;
        for (const epsilon of [0.015, 0.025, 0.04, 0.06]) {
          const approximation = new cv.Mat();
          let selected = false;
          try {
            cv.approxPolyDP(contour, approximation, epsilon * perimeter, true);
            if (approximation.rows === 4) {
              quad = approximation;
              selected = true;
              break;
            }
          } finally {
            if (!selected) approximation.delete();
          }
        }

        if (!quad) continue;
        try {
          const points: Point[] = [];
          for (let pointIndex = 0; pointIndex < 4; pointIndex++) {
            points.push({
              x: quad.data32S[pointIndex * 2],
              y: quad.data32S[pointIndex * 2 + 1],
            });
          }

          const candidate = candidateFromQuad(points, area, frameArea, edgeEvidence, guideRect);
          if (candidate && (!best || candidate.confidence > best.confidence)) {
            best = candidate;
          }
        } finally {
          quad.delete();
        }
      } finally {
        contour.delete();
      }
    }
  } finally {
    contours?.delete();
    hierarchy?.delete();
  }

  return best;
}

/**
 * Detect a standard ID-1 card from a single RGBA OpenCV frame.
 *
 * The primary pass follows closed Canny edges. If that pass is weak, an
 * adaptive-threshold pass recovers low-contrast cards and cards with broken
 * borders. RETR_LIST is intentional: a card can be nested inside the outline
 * of a foot/shadow and therefore may not be an external contour.
 */
export function detectCard(
  src: any,
  frameWidth: number,
  frameHeight: number,
  options: CardDetectionOptions = {}
): CardDetectionResult {
  const frameArea = frameWidth * frameHeight;
  let gray: any = null;
  let blurred: any = null;
  let edges: any = null;
  let closedEdges: any = null;
  let edgeKernel: any = null;

  try {
    gray = new cv.Mat();
    blurred = new cv.Mat();
    edges = new cv.Mat();
    closedEdges = new cv.Mat();
    edgeKernel = cv.Mat.ones(5, 5, cv.CV_8U);
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 35, 125);
    cv.morphologyEx(edges, closedEdges, cv.MORPH_CLOSE, edgeKernel);
    cv.dilate(closedEdges, closedEdges, edgeKernel, new cv.Point(-1, -1), 1);

    let best = findBestCandidate(
      closedEdges,
      closedEdges,
      frameArea,
      cv.RETR_LIST,
      options.guideRect
    );

    let adaptive: any = null;
    let adaptiveClosed: any = null;
    try {
      adaptive = new cv.Mat();
      adaptiveClosed = new cv.Mat();
      cv.adaptiveThreshold(
        blurred,
        adaptive,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        31,
        7
      );
      cv.morphologyEx(adaptive, adaptiveClosed, cv.MORPH_CLOSE, edgeKernel);
      const adaptiveBest = findBestCandidate(
        adaptiveClosed,
        closedEdges,
        frameArea,
        cv.RETR_LIST,
        options.guideRect
      );
      if (adaptiveBest && (!best || adaptiveBest.confidence > best.confidence)) {
        best = adaptiveBest;
      }
    } finally {
      adaptive?.delete();
      adaptiveClosed?.delete();
    }

    if (!best) return { ...EMPTY_RESULT };
    return {
      detected: true,
      confidence: best.confidence,
      widthPx: best.widthPx,
      heightPx: best.heightPx,
      corners: best.corners,
      ppm: best.ppm,
      homography: best.homography,
      measurementReady: best.measurementReady,
      quality: best.quality,
      issue: best.issue,
      edgeConsistency: best.edgeConsistency,
      perspectiveScore: best.perspectiveScore,
      areaRatio: best.areaRatio,
    };
  } finally {
    gray?.delete();
    blurred?.delete();
    edges?.delete();
    closedEdges?.delete();
    edgeKernel?.delete();
  }
}

/** Draw the card outline and scale estimate on the transparent overlay. */
export function drawCardOverlay(
  ctx: CanvasRenderingContext2D,
  result: CardDetectionResult,
  scaleX: number,
  scaleY: number
): void {
  if (!result.detected || !result.corners) return;

  const corners = result.corners.map((point) => ({
    x: point.x * scaleX,
    y: point.y * scaleY,
  }));
  const alpha = Math.max(0.55, result.confidence);
  const color =
    result.confidence > 0.7
      ? `rgba(34,197,94,${alpha})`
      : `rgba(234,179,8,${alpha})`;

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
  ctx.closePath();
  ctx.stroke();

  for (const point of corners) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  const labelY = Math.max(18, Math.min(...corners.map((point) => point.y)) - 10);
  const labelX = Math.max(6, Math.min(...corners.map((point) => point.x)));
  ctx.fillStyle = color;
  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.fillText(
    result.measurementReady ? 'ID-1 reference ready' : 'Adjust ID-1 reference',
    labelX,
    labelY
  );
}
