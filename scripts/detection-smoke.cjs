/* eslint-disable @typescript-eslint/no-require-imports -- This smoke test installs a CommonJS TypeScript loader. */
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const Module = require('node:module');
const path = require('node:path');
const ts = require('typescript');

const projectRoot = path.resolve(__dirname, '..');
const originalResolveFilename = Module._resolveFilename;

// Load the real TypeScript detector modules without adding a test-only runner.
Module._resolveFilename = function resolveAlias(request, parent, isMain, options) {
  const resolvedRequest = request.startsWith('@/')
    ? path.join(projectRoot, 'src', request.slice(2))
    : request;
  return originalResolveFilename.call(this, resolvedRequest, parent, isMain, options);
};

require.extensions['.ts'] = function transpileTypeScript(module, filename) {
  const source = fs.readFileSync(filename, 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      esModuleInterop: true,
    },
    fileName: filename,
  }).outputText;
  module._compile(output, filename);
};

global.cv = require('opencv.js');

const { detectCard } = require('../src/lib/opencv/cardDetector.ts');
const {
  detectFootContour,
  detectFootContourFromModelMask,
} = require('../src/lib/opencv/footContour.ts');
const {
  calculateLetterboxPlacement,
} = require('../src/lib/mediapipe/footSegmenter.ts');
const {
  cardTrackingReference,
  FootTracker,
} = require('../src/lib/tracking/footTracker.ts');
const { CardTracker } = require('../src/lib/tracking/cardTracker.ts');
const {
  createCardHomography,
  distanceOnPlaneMm,
  transformPoint,
} = require('../src/lib/calculations/homography.ts');
const { measureFootGeometry } = require('../src/lib/calculations/footGeometry.ts');
const {
  buildMeasurementResult,
  buildMeasurementResultFromMillimeters,
} = require('../src/lib/calculations/dimensions.ts');
const { canCaptureMeasurement } = require('../src/lib/calculations/measurementPolicy.ts');
const {
  calculateDisplayGuideRects,
  calculateObjectCoverGeometry,
  displayRectToSource,
  sourcePointToDisplay,
} = require('../src/lib/camera/objectCover.ts');
const {
  CameraSetupError,
  classifyCameraError,
  stopMediaStream,
} = require('../src/lib/camera/mediaStream.ts');
const { copyFootMaskAndClose } = require('../src/lib/mediapipe/segmenterResult.ts');
const { FrameProcessingGate } = require('../src/lib/performance/frameProcessingGate.ts');

const WIDTH = 640;
const HEIGHT = 480;
const FALLBACK_ROI = {
  x: WIDTH * 0.08,
  y: HEIGHT * 0.04,
  width: WIDTH * 0.68,
  height: HEIGHT * 0.91,
};
const DISPLAY_GUIDES = calculateDisplayGuideRects(WIDTH, HEIGHT);
assert.ok(DISPLAY_GUIDES);
const CARD_TEST_GUIDE = {
  x: DISPLAY_GUIDES.card.x * WIDTH,
  y: DISPLAY_GUIDES.card.y * HEIGHT,
  width: DISPLAY_GUIDES.card.width * WIDTH,
  height: DISPLAY_GUIDES.card.height * HEIGHT,
};

function createFrame(red = 214, green = 214, blue = 214) {
  const frame = new cv.Mat(HEIGHT, WIDTH, cv.CV_8UC4);
  for (let index = 0; index < frame.data.length; index += 4) {
    frame.data[index] = red;
    frame.data[index + 1] = green;
    frame.data[index + 2] = blue;
    frame.data[index + 3] = 255;
  }
  return frame;
}

function setPixel(frame, x, y, colour) {
  const index = (y * WIDTH + x) * 4;
  frame.data[index] = colour[0];
  frame.data[index + 1] = colour[1];
  frame.data[index + 2] = colour[2];
  frame.data[index + 3] = 255;
}

function paintCard(frame, left, top, width, height) {
  const border = 4;
  for (let y = top; y < top + height; y++) {
    for (let x = left; x < left + width; x++) {
      const isBorder =
        x < left + border ||
        x >= left + width - border ||
        y < top + border ||
        y >= top + height - border;
      setPixel(frame, x, y, isBorder ? [35, 35, 35] : [248, 248, 248]);
    }
  }
}

function distanceToSegment(x, y, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const amount = lengthSquared === 0
    ? 0
    : Math.max(0, Math.min(1, ((x - start.x) * dx + (y - start.y) * dy) / lengthSquared));
  return Math.hypot(x - (start.x + amount * dx), y - (start.y + amount * dy));
}

function paintPolygonCard(frame, points) {
  const minimumX = Math.floor(Math.min(...points.map((point) => point.x)));
  const maximumX = Math.ceil(Math.max(...points.map((point) => point.x)));
  const minimumY = Math.floor(Math.min(...points.map((point) => point.y)));
  const maximumY = Math.ceil(Math.max(...points.map((point) => point.y)));

  for (let y = minimumY; y <= maximumY; y++) {
    for (let x = minimumX; x <= maximumX; x++) {
      const crosses = points.map((point, index) => {
        const next = points[(index + 1) % points.length];
        return (next.x - point.x) * (y - point.y) - (next.y - point.y) * (x - point.x);
      });
      const inside = crosses.every((value) => value >= 0) || crosses.every((value) => value <= 0);
      if (!inside) continue;
      const borderDistance = Math.min(
        ...points.map((point, index) =>
          distanceToSegment(x, y, point, points[(index + 1) % points.length])
        )
      );
      setPixel(frame, x, y, borderDistance <= 4 ? [35, 35, 35] : [248, 248, 248]);
    }
  }
}

function assertNear(actual, expected, tolerance, message) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${message}: expected ${expected} ± ${tolerance}, received ${actual}`
  );
}

function invertHomographyPoint(transform, point) {
  const [a, b, c, d, e, f, g, h, i] = transform.matrix;
  const determinant =
    a * (e * i - f * h) -
    b * (d * i - f * g) +
    c * (d * h - e * g);
  assert.ok(Math.abs(determinant) > 1e-12, 'fixture homography must be invertible');
  const inverse = [
    (e * i - f * h) / determinant,
    (c * h - b * i) / determinant,
    (b * f - c * e) / determinant,
    (f * g - d * i) / determinant,
    (a * i - c * g) / determinant,
    (c * d - a * f) / determinant,
    (d * h - e * g) / determinant,
    (b * g - a * h) / determinant,
    (a * e - b * d) / determinant,
  ];
  const denominator = inverse[6] * point.x + inverse[7] * point.y + inverse[8];
  assert.ok(Math.abs(denominator) > 1e-12, 'fixture point must not lie on projective horizon');
  return {
    x: (inverse[0] * point.x + inverse[1] * point.y + inverse[2]) / denominator,
    y: (inverse[3] * point.x + inverse[4] * point.y + inverse[5]) / denominator,
  };
}

// Independent plane-to-image fixture transform. Production code only solves
// the reverse image-to-plane mapping and does not use these coefficients.
function projectKnownPlanePoint(point) {
  const denominator = 1 + 0.0008 * point.x + 0.0005 * point.y;
  return {
    x: (2.7 * point.x + 0.25 * point.y + 800) / denominator,
    y: (0.12 * point.x + 2.5 * point.y + 300) / denominator,
  };
}

function rotateAndTranslate(point, angleRad, center) {
  const cosine = Math.cos(angleRad);
  const sine = Math.sin(angleRad);
  return {
    x: center.x + point.x * cosine - point.y * sine,
    y: center.y + point.x * sine + point.y * cosine,
  };
}

function syntheticFootWidth(normalizedPosition, widthMm) {
  if (normalizedPosition <= 0.08) return widthMm * (0.38 + normalizedPosition * 3.5);
  if (normalizedPosition <= 0.32) {
    return widthMm * (0.66 + ((normalizedPosition - 0.08) / 0.24) * 0.34);
  }
  if (normalizedPosition <= 0.82) return widthMm;
  return widthMm * Math.max(0, (1 - normalizedPosition) / 0.18);
}

function createSyntheticFootPoints({
  lengthMm = 260,
  widthMm = 100,
  angleDeg = 0,
  center = { x: 280, y: 220 },
  project = (point) => point,
  addEndOutliers = false,
  asymmetryMm = 0,
} = {}) {
  const angleRad = (angleDeg * Math.PI) / 180;
  const top = [];
  const bottom = [];
  const sampleCount = 360;
  for (let index = 0; index <= sampleCount; index++) {
    const amount = index / sampleCount;
    const major = -lengthMm / 2 + amount * lengthMm;
    const isEndOutlier = addEndOutliers && [0.025, 0.05, 0.95, 0.975]
      .some((target) => Math.abs(amount - target) < 1 / sampleCount / 2);
    const halfWidth = isEndOutlier
      ? widthMm * 0.9
      : syntheticFootWidth(amount, widthMm) / 2;
    const centerOffset = asymmetryMm * Math.sin(amount * Math.PI * 2);
    top.push(project(rotateAndTranslate({ x: major, y: centerOffset + halfWidth }, angleRad, center)));
    bottom.push(project(rotateAndTranslate({ x: major, y: centerOffset - halfWidth }, angleRad, center)));
  }

  return [...top, ...bottom.reverse()];
}

function paintFoot(frame) {
  for (let y = 70; y < 430; y++) {
    for (let x = 150; x < 370; x++) {
      const body = ((x - 260) / 58) ** 2 + ((y - 270) / 140) ** 2 <= 1;
      const forefoot = ((x - 258) / 70) ** 2 + ((y - 145) / 62) ** 2 <= 1;
      const heel = ((x - 260) / 42) ** 2 + ((y - 390) / 45) ** 2 <= 1;
      if (body || forefoot || heel) setPixel(frame, x, y, [188, 126, 94]);
    }
  }
}

function testLandscapeCardAndFoot() {
  const frame = createFrame();
  try {
    paintFoot(frame);
    paintCard(frame, 495, 350, 110, 69);
    const card = detectCard(frame, WIDTH, HEIGHT, { guideRect: CARD_TEST_GUIDE });
    assert.equal(card.detected, true, 'landscape card should be detected');
    assert.ok(card.confidence >= 0.5, `card confidence was ${card.confidence}`);
    assert.equal(card.measurementReady, true, `landscape card issue was ${card.issue}`);

    const foot = detectFootContour(frame, FALLBACK_ROI, WIDTH, HEIGHT, card.corners);
    assert.equal(foot.detected, true, 'foot silhouette should be detected');
    assert.ok(foot.lengthPx > 220 && foot.lengthPx < 380, `foot length was ${foot.lengthPx}`);
    assert.ok(foot.widthPx > 80 && foot.widthPx < 190, `foot width was ${foot.widthPx}`);
  } finally {
    frame.delete();
  }
}

function testPortraitCardScale() {
  const frame = createFrame(205, 205, 205);
  try {
    paintCard(frame, 505, 280, 64, 102);
    const card = detectCard(frame, WIDTH, HEIGHT, { guideRect: CARD_TEST_GUIDE });
    assert.equal(card.detected, true, 'portrait card should be detected');
    assert.ok(card.ppm > 0.7 && card.ppm < 2, `portrait card PPM was ${card.ppm}`);
    assert.equal(card.measurementReady, true, `portrait card issue was ${card.issue}`);
  } finally {
    frame.delete();
  }
}

function testRotatedCard() {
  const frame = createFrame(202, 202, 202);
  try {
    paintPolygonCard(frame, [
      { x: 490, y: 305 },
      { x: 594, y: 335 },
      { x: 575, y: 400 },
      { x: 471, y: 370 },
    ]);
    const card = detectCard(frame, WIDTH, HEIGHT, { guideRect: CARD_TEST_GUIDE });
    assert.equal(card.detected, true, 'rotated card should be detected');
    assert.ok(card.confidence >= 0.5, `rotated card confidence was ${card.confidence}`);
  } finally {
    frame.delete();
  }
}

function testPerspectiveCardDetection() {
  const frame = createFrame(202, 202, 202);
  try {
    paintPolygonCard(frame, [
      { x: 478, y: 304 },
      { x: 597, y: 326 },
      { x: 574, y: 397 },
      { x: 469, y: 372 },
    ]);
    const card = detectCard(frame, WIDTH, HEIGHT, { guideRect: CARD_TEST_GUIDE });
    assert.equal(card.detected, true, 'mild perspective card should be detected');
    assert.ok(card.homography, 'detected perspective card should expose a homography');
    assert.ok(card.perspectiveScore >= 0.62, `perspective score was ${card.perspectiveScore}`);
    assert.equal(card.measurementReady, true, `perspective card issue was ${card.issue}`);
  } finally {
    frame.delete();
  }
}

function testWrongAspectRectangleRejected() {
  const squareFrame = createFrame(208, 208, 208);
  const stripFrame = createFrame(208, 208, 208);
  try {
    paintCard(squareFrame, 485, 320, 92, 92);
    paintCard(stripFrame, 460, 340, 155, 42);
    assert.equal(
      detectCard(squareFrame, WIDTH, HEIGHT).detected,
      false,
      'a square object must not be accepted as an ID-1 card'
    );
    assert.equal(
      detectCard(stripFrame, WIDTH, HEIGHT).detected,
      false,
      'a long rectangular object must not be accepted as an ID-1 card'
    );
  } finally {
    squareFrame.delete();
    stripFrame.delete();
  }
}

function testReferenceCardMeasurementGates() {
  const outsideFrame = createFrame(208, 208, 208);
  const smallFrame = createFrame(208, 208, 208);
  const largeFrame = createFrame(208, 208, 208);
  try {
    paintCard(outsideFrame, 100, 340, 110, 69);
    const outside = detectCard(outsideFrame, WIDTH, HEIGHT, { guideRect: CARD_TEST_GUIDE });
    assert.equal(outside.detected, true, 'an ID-1-shaped non-card rectangle may be located geometrically');
    assert.equal(outside.measurementReady, false, 'an object outside the visible card zone must not measure');
    assert.equal(outside.issue, 'outside-guide');

    paintCard(smallFrame, 515, 360, 60, 38);
    const small = detectCard(smallFrame, WIDTH, HEIGHT, { guideRect: CARD_TEST_GUIDE });
    assert.equal(small.detected, true, 'a small reference candidate should still provide guidance');
    assert.equal(small.measurementReady, false);
    assert.equal(small.issue, 'too-small');

    paintCard(largeFrame, 120, 100, 300, 189);
    const large = detectCard(largeFrame, WIDTH, HEIGHT);
    assert.equal(large.detected, true, 'a large reference candidate should still provide guidance');
    assert.equal(large.measurementReady, false);
    assert.equal(large.issue, 'too-large');
  } finally {
    outsideFrame.delete();
    smallFrame.delete();
    largeFrame.delete();
  }
}

function testLandscapePortraitAndPerspectiveHomography() {
  const landscapeCorners = [
    { x: 100, y: 80 },
    { x: 271.2, y: 80 },
    { x: 271.2, y: 187.96 },
    { x: 100, y: 187.96 },
  ];
  const landscape = createCardHomography(landscapeCorners);
  assert.ok(landscape, 'landscape card should create a homography');
  assert.equal(landscape.cardOrientation, 'landscape');
  assertNear(distanceOnPlaneMm(landscape, landscapeCorners[0], landscapeCorners[1]), 85.6, 1e-6, 'landscape long edge');
  assertNear(distanceOnPlaneMm(landscape, landscapeCorners[1], landscapeCorners[2]), 53.98, 1e-6, 'landscape short edge');

  const portraitCorners = [
    { x: 320, y: 60 },
    { x: 427.96, y: 60 },
    { x: 427.96, y: 231.2 },
    { x: 320, y: 231.2 },
  ];
  const portrait = createCardHomography(portraitCorners);
  assert.ok(portrait, 'portrait card should create a homography');
  assert.equal(portrait.cardOrientation, 'portrait');
  assertNear(distanceOnPlaneMm(portrait, portraitCorners[0], portraitCorners[1]), 53.98, 1e-6, 'portrait short edge');
  assertNear(distanceOnPlaneMm(portrait, portraitCorners[1], portraitCorners[2]), 85.6, 1e-6, 'portrait long edge');

  const perspectiveCorners = [
    { x: 0, y: 0 },
    { x: 85.6, y: 0 },
    { x: 85.6, y: 53.98 },
    { x: 0, y: 53.98 },
  ].map(projectKnownPlanePoint);
  const perspective = createCardHomography(perspectiveCorners);
  assert.ok(perspective, 'perspective card should create a homography');
  const firstPlanePoint = { x: -180, y: 24 };
  const secondPlanePoint = { x: 80, y: 24 };
  const firstImagePoint = projectKnownPlanePoint(firstPlanePoint);
  const secondImagePoint = projectKnownPlanePoint(secondPlanePoint);
  assertNear(
    distanceOnPlaneMm(perspective, firstImagePoint, secondImagePoint),
    260,
    1e-6,
    'perspective-corrected plane distance'
  );
  const reprojected = transformPoint(perspective, firstImagePoint);
  assert.ok(reprojected);
  assertNear(reprojected.x, firstPlanePoint.x, 1e-6, 'perspective round-trip x');
  assertNear(reprojected.y, firstPlanePoint.y, 1e-6, 'perspective round-trip y');

  assert.equal(createCardHomography(null), null);
  assert.equal(createCardHomography([{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 3, y: 0 }]), null);
  assert.equal(createCardHomography([{ x: 0, y: 0 }, { x: Number.NaN, y: 0 }, { x: 2, y: 2 }, { x: 0, y: 2 }]), null);
}

const syntheticLengthErrors = [];
const syntheticWidthErrors = [];

function recordGeometryError(result, expectedLengthMm, expectedWidthMm, homography = null) {
  assert.equal(result.detected, true, 'synthetic foot should produce valid geometry');
  assert.ok(result.heelPoint && result.toePoint && result.widestPoints);
  const lengthMm = homography
    ? distanceOnPlaneMm(homography, result.heelPoint, result.toePoint)
    : result.lengthPx;
  const widthMm = homography
    ? distanceOnPlaneMm(homography, result.widestPoints[0], result.widestPoints[1])
    : result.widthPx;
  assert.ok(lengthMm && widthMm);
  syntheticLengthErrors.push(Math.abs(lengthMm - expectedLengthMm));
  syntheticWidthErrors.push(Math.abs(widthMm - expectedWidthMm));
  return { lengthMm, widthMm };
}

function testRotatedFootAndPerpendicularWidth() {
  for (const angleDeg of [0, 25, 57, 90, 133]) {
    const result = measureFootGeometry(createSyntheticFootPoints({ angleDeg }));
    const measured = recordGeometryError(result, 260, 100);
    assertNear(measured.lengthMm, 260, 2, `rotated foot length at ${angleDeg}°`);
    assertNear(measured.widthMm, 100, 3, `rotated foot width at ${angleDeg}°`);

    const lengthVector = {
      x: result.toePoint.x - result.heelPoint.x,
      y: result.toePoint.y - result.heelPoint.y,
    };
    const widthVector = {
      x: result.widestPoints[1].x - result.widestPoints[0].x,
      y: result.widestPoints[1].y - result.widestPoints[0].y,
    };
    const cosine = Math.abs(
      (lengthVector.x * widthVector.x + lengthVector.y * widthVector.y) /
      (Math.hypot(lengthVector.x, lengthVector.y) * Math.hypot(widthVector.x, widthVector.y))
    );
    assert.ok(cosine < 0.08, `width should be perpendicular at ${angleDeg}°; |cos|=${cosine}`);
  }

  const robustResult = measureFootGeometry(createSyntheticFootPoints({
    angleDeg: 41,
    addEndOutliers: true,
  }));
  const robust = recordGeometryError(robustResult, 260, 100);
  assertNear(robust.widthMm, 100, 4, 'toe/heel outliers must not inflate representative width');

  const curvedResult = measureFootGeometry(createSyntheticFootPoints({
    angleDeg: 33,
    asymmetryMm: 12,
  }));
  const curved = recordGeometryError(curvedResult, 260, 100);
  assertNear(curved.widthMm, 100, 4, 'curved asymmetric sides should use one perpendicular section');

  const unnatural = measureFootGeometry(createSyntheticFootPoints({ widthMm: 180 }));
  assert.equal(unnatural.detected, false, 'an implausibly wide contour must fail geometry validation');
}

function testPerspectiveCorrectedFootFixture() {
  const cardCorners = [
    { x: 0, y: 0 },
    { x: 85.6, y: 0 },
    { x: 85.6, y: 53.98 },
    { x: 0, y: 53.98 },
  ].map(projectKnownPlanePoint);
  const homography = createCardHomography(cardCorners);
  assert.ok(homography);
  const imagePoints = createSyntheticFootPoints({
    lengthMm: 265,
    widthMm: 102,
    angleDeg: 18,
    center: { x: -115, y: 80 },
    project: projectKnownPlanePoint,
  });
  const result = measureFootGeometry(imagePoints, homography);
  const measured = recordGeometryError(result, 265, 102, homography);
  assertNear(measured.lengthMm, 265, 2, 'homography-corrected foot length');
  assertNear(measured.widthMm, 102, 3, 'homography-corrected foot width');

  const cornerNoise = [
    { x: 1.1, y: -0.7 },
    { x: -0.8, y: 0.9 },
    { x: 0.7, y: 1.0 },
    { x: -1.0, y: -0.6 },
  ];
  const noisyHomography = createCardHomography(cardCorners.map((point, index) => ({
    x: point.x + cornerNoise[index].x,
    y: point.y + cornerNoise[index].y,
  })));
  assert.ok(noisyHomography);
  const noisyResult = measureFootGeometry(imagePoints, noisyHomography);
  const noisyMeasured = recordGeometryError(noisyResult, 265, 102, noisyHomography);
  assertNear(noisyMeasured.lengthMm, 265, 5, 'small card-corner noise length');
  assertNear(noisyMeasured.widthMm, 102, 6, 'small card-corner noise width');
}

function testBlankFrame() {
  const frame = createFrame();
  try {
    const card = detectCard(frame, WIDTH, HEIGHT);
    const foot = detectFootContour(frame, FALLBACK_ROI, WIDTH, HEIGHT);
    assert.equal(card.detected, false, 'blank frame must not contain a card');
    assert.equal(foot.detected, false, 'blank frame must not contain a foot');
  } finally {
    frame.delete();
  }
}

function testLetterboxGeometryAndModelMask() {
  assert.deepEqual(calculateLetterboxPlacement(200, 400, 384), {
    x: 96,
    y: 0,
    width: 192,
    height: 384,
    targetSize: 384,
  });
  assert.deepEqual(calculateLetterboxPlacement(400, 200, 384), {
    x: 0,
    y: 96,
    width: 384,
    height: 192,
    targetSize: 384,
  });

  const placement = calculateLetterboxPlacement(FALLBACK_ROI.width, FALLBACK_ROI.height, 384);
  for (const angleDeg of [0, 45, 90]) {
    const data = new Float32Array(384 * 384);
    const angle = (angleDeg * Math.PI) / 180;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    for (let maskY = placement.y; maskY < placement.y + placement.height; maskY++) {
      for (let maskX = placement.x; maskX < placement.x + placement.width; maskX++) {
        const rotatedX = ((maskX - placement.x) / placement.width) * FALLBACK_ROI.width;
        const rotatedY = ((maskY - placement.y) / placement.height) * FALLBACK_ROI.height;
        const deltaX = rotatedX - 215;
        const deltaY = rotatedY - 230;
        const localX = 215 + deltaX * cosine + deltaY * sine;
        const localY = 230 - deltaX * sine + deltaY * cosine;
        const body = ((localX - 215) / 58) ** 2 + ((localY - 230) / 140) ** 2 <= 1;
        const forefoot = ((localX - 215) / 70) ** 2 + ((localY - 105) / 58) ** 2 <= 1;
        const heel = ((localX - 215) / 42) ** 2 + ((localY - 365) / 42) ** 2 <= 1;
        if (body || forefoot || heel) data[maskY * 384 + maskX] = 0.98;
      }
    }

    const result = detectFootContourFromModelMask(
      { data, width: 384, height: 384 },
      FALLBACK_ROI,
      placement,
      WIDTH,
      HEIGHT,
      0.5
    );
    assert.equal(result.detected, true, `model mask rotated ${angleDeg}° should produce a foot contour`);
    assert.ok(result.lengthPx > 240 && result.lengthPx < 380, `model length at ${angleDeg}° was ${result.lengthPx}`);
    assert.ok(result.widthPx > 80 && result.widthPx < 190, `model width at ${angleDeg}° was ${result.widthPx}`);
  }
}

function testBundledFootModelAssets() {
  const modelPath = path.join(projectRoot, 'public', 'models', 'foot_segmenter_fp16_metadata.tflite');
  const model = fs.readFileSync(modelPath);
  assert.equal(
    crypto.createHash('sha256').update(model).digest('hex'),
    '480a97381c7b59c32b6e29b9265ed4073c169eb4fc4bf3b7c71e67380952a130'
  );
  assert.ok(model.includes(Buffer.from('TFLITE_METADATA')), 'model must contain MediaPipe metadata');
  for (const asset of [
    'vision_wasm_internal.js',
    'vision_wasm_internal.wasm',
    'vision_wasm_module_internal.js',
    'vision_wasm_module_internal.wasm',
    'vision_wasm_nosimd_internal.js',
    'vision_wasm_nosimd_internal.wasm',
  ]) {
    assert.equal(
      fs.existsSync(path.join(projectRoot, 'public', 'mediapipe', 'wasm', asset)),
      true,
      `${asset} should be self-hosted`
    );
  }
}

const TRACKING_CARD_CORNERS = [
  { x: 445, y: 330 },
  { x: 616.2, y: 330 },
  { x: 616.2, y: 437.96 },
  { x: 445, y: 437.96 },
];
const TRACKING_REFERENCE = cardTrackingReference(TRACKING_CARD_CORNERS);
const TRACKING_HOMOGRAPHY = createCardHomography(TRACKING_CARD_CORNERS);
assert.ok(TRACKING_REFERENCE && TRACKING_HOMOGRAPHY, 'tracking fixtures need valid card geometry');

const TRACKING_CALIBRATION = {
  mode: 'homography',
  ppm: 2,
  confidence: 0.9,
  homography: TRACKING_HOMOGRAPHY,
  fallbackReference: TRACKING_REFERENCE,
};

function createTrackedFoot({
  lengthMm = 260,
  widthMm = 100,
  relativeCenterMm = { x: -120, y: -20 },
  angleDeg = 90,
} = {}) {
  const angle = (angleDeg * Math.PI) / 180;
  const planeCenter = {
    x: TRACKING_HOMOGRAPHY.destinationWidthMm / 2 + relativeCenterMm.x,
    y: TRACKING_HOMOGRAPHY.destinationHeightMm / 2 + relativeCenterMm.y,
  };
  const longAxis = { x: Math.cos(angle), y: Math.sin(angle) };
  const widthAxis = { x: -longAxis.y, y: longAxis.x };
  const planeHeel = {
    x: planeCenter.x - (longAxis.x * lengthMm) / 2,
    y: planeCenter.y - (longAxis.y * lengthMm) / 2,
  };
  const planeToe = {
    x: planeCenter.x + (longAxis.x * lengthMm) / 2,
    y: planeCenter.y + (longAxis.y * lengthMm) / 2,
  };
  const planeWidest = [
    {
      x: planeCenter.x - (widthAxis.x * widthMm) / 2,
      y: planeCenter.y - (widthAxis.y * widthMm) / 2,
    },
    {
      x: planeCenter.x + (widthAxis.x * widthMm) / 2,
      y: planeCenter.y + (widthAxis.y * widthMm) / 2,
    },
  ];
  const heelPoint = invertHomographyPoint(TRACKING_HOMOGRAPHY, planeHeel);
  const toePoint = invertHomographyPoint(TRACKING_HOMOGRAPHY, planeToe);
  const widestPoints = planeWidest.map((point) => invertHomographyPoint(TRACKING_HOMOGRAPHY, point));
  const lengthPx = Math.hypot(toePoint.x - heelPoint.x, toePoint.y - heelPoint.y);
  const widthPx = Math.hypot(
    widestPoints[1].x - widestPoints[0].x,
    widestPoints[1].y - widestPoints[0].y
  );

  return {
    detected: true,
    lengthPx,
    widthPx,
    contourPoints: [heelPoint, widestPoints[0], toePoint, widestPoints[1]],
    heelPoint,
    toePoint,
    widestPoints,
    geometryConfidence: 0.9,
    shapeSignature: [0.22, 0.28, 0.34, 0.38, 0.39, 0.4, 0.4, 0.38, 0.3, 0.12],
  };
}

function addStableWindow(tracker, startTimestamp = 0) {
  const lengths = [260, 261, 259, 260.5, 259.5, 260.2, 260];
  const widths = [100, 100.8, 99.4, 100.5, 99.7, 100.2, 100];
  const scales = [2, 2.04, 1.96, 2.02, 1.98, 2.03, 1.97];
  const angles = [90, 91, 89, 90.5, 89.5, 90.2, 90];
  let update = null;

  for (let index = 0; index < 7; index++) {
    update = tracker.update(
      createTrackedFoot({
        lengthMm: lengths[index],
        widthMm: widths[index],
        angleDeg: angles[index],
      }),
      { ...TRACKING_CALIBRATION, ppm: scales[index] },
      startTimestamp + index * 80
    );
    assert.equal(update.status.sampleCount, index + 1, `sample ${index + 1} should be accepted`);
  }

  return update;
}

function testSevenFrameFootTracking() {
  const tracker = new FootTracker();
  const update = addStableWindow(tracker);

  assert.equal(update.status.isStable, true, 'seven consistent observations should be stable');
  assert.equal(update.status.phase, 'stable');
  assert.ok(update.measurement, 'stable tracking should produce a measurement');
  assert.ok(Math.abs(update.measurement.lengthMm - 260) < 0.01);
  assert.ok(Math.abs(update.measurement.widthMm - 100) < 0.01);
}

function testOutlierAndMissHandling() {
  const tracker = new FootTracker();
  for (let index = 0; index < 3; index++) {
    tracker.update(createTrackedFoot(), TRACKING_CALIBRATION, index * 100);
  }

  const outlier = tracker.update(
    createTrackedFoot({ lengthMm: 330, widthMm: 130, relativeCenterMm: { x: -35, y: 60 } }),
    TRACKING_CALIBRATION,
    300
  );
  assert.equal(outlier.sampleAccepted, false, 'one mismatching contour should be ignored');
  assert.equal(outlier.status.sampleCount, 3, 'an outlier must not replace the good window');

  const recovered = tracker.update(createTrackedFoot(), TRACKING_CALIBRATION, 400);
  assert.equal(recovered.status.sampleCount, 4, 'tracking should resume after one outlier');

  assert.equal(tracker.update(null, null, 500).status.sampleCount, 4);
  assert.equal(tracker.update(null, null, 600).status.sampleCount, 4);
  assert.equal(
    tracker.update(null, null, 700).status.sampleCount,
    0,
    'three processed misses should reset the track'
  );
}

function testCoherentMovementRebasesTrack() {
  const tracker = new FootTracker();
  const stable = addStableWindow(tracker);
  assert.equal(stable.status.isStable, true);

  const movedFoot = {
    lengthMm: 310,
    widthMm: 118,
    relativeCenterMm: { x: -40, y: 55 },
  };
  const first = tracker.update(createTrackedFoot(movedFoot), TRACKING_CALIBRATION, 600);
  assert.equal(first.status.isStable, false, 'movement must invalidate a previous stable result');
  assert.equal(first.status.sampleCount, 7, 'one mismatch should not discard the good history');

  const duplicate = tracker.update(createTrackedFoot(movedFoot), TRACKING_CALIBRATION, 600);
  assert.equal(duplicate.sampleAccepted, false, 'a duplicate camera frame must not advance rebase');
  assert.equal(duplicate.status.sampleCount, 7);

  const second = tracker.update(createTrackedFoot(movedFoot), TRACKING_CALIBRATION, 700);
  assert.equal(second.status.sampleCount, 2, 'two coherent mismatches should start a new track');
  assert.equal(second.status.isStable, false);
  assert.equal(second.measurement, null);
}

function testTimingGapAndManualReset() {
  const tracker = new FootTracker();
  tracker.update(createTrackedFoot(), TRACKING_CALIBRATION, 0);
  const tooSoon = tracker.update(createTrackedFoot(), TRACKING_CALIBRATION, 40);
  assert.equal(tooSoon.sampleAccepted, false, 'samples closer than 80 ms should be skipped');
  assert.equal(tooSoon.status.sampleCount, 1);

  const afterGap = tracker.update(createTrackedFoot(), TRACKING_CALIBRATION, 900);
  assert.equal(afterGap.status.sampleCount, 1, 'a long interruption should start a fresh track');

  tracker.reset();
  assert.equal(tracker.getStatus().sampleCount, 0, 'manual reset should clear all samples');
}

function testNoisyWindowWaitsForConsistency() {
  const tracker = new FootTracker();
  const noisyLengths = [240, 245, 250, 255, 260, 265, 270];
  let update = null;
  for (let index = 0; index < noisyLengths.length; index++) {
    update = tracker.update(
      createTrackedFoot({ lengthMm: noisyLengths[index] }),
      TRACKING_CALIBRATION,
      index * 80
    );
  }

  assert.equal(update.status.sampleCount, 7);
  assert.equal(update.status.phase, 'stabilizing');
  assert.equal(update.status.isStable, false, 'seven inconsistent samples must not be ready');
  assert.equal(update.measurement, null);

  for (let index = 0; index < 7; index++) {
    update = tracker.update(createTrackedFoot(), TRACKING_CALIBRATION, 560 + index * 80);
  }
  assert.equal(update.status.isStable, true, 'a clean rolling window should recover to stable');
}

function createCardResult(corners, overrides = {}) {
  const homography = createCardHomography(corners);
  const top = Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y);
  const left = Math.hypot(corners[3].x - corners[0].x, corners[3].y - corners[0].y);
  return {
    detected: true,
    confidence: 0.9,
    widthPx: top,
    heightPx: left,
    corners,
    ppm: (Math.max(top, left) / 85.6 + Math.min(top, left) / 53.98) / 2,
    homography,
    measurementReady: Boolean(homography),
    quality: 'high',
    issue: null,
    edgeConsistency: 0.95,
    perspectiveScore: 0.95,
    areaRatio: 0.06,
    ...overrides,
  };
}

function shiftPoints(points, x, y) {
  return points.map((point) => ({ x: point.x + x, y: point.y + y }));
}

function scalePoints(points, amount) {
  const center = points.reduce(
    (sum, point) => ({ x: sum.x + point.x / points.length, y: sum.y + point.y / points.length }),
    { x: 0, y: 0 }
  );
  return points.map((point) => ({
    x: center.x + (point.x - center.x) * amount,
    y: center.y + (point.y - center.y) * amount,
  }));
}

function testCardTrackingRejectsCameraMovement() {
  const tracker = new CardTracker();
  let status = null;
  for (let index = 0; index < 3; index++) {
    status = tracker.update(
      createCardResult(shiftPoints(TRACKING_CARD_CORNERS, index * 0.5, index * 0.25)),
      WIDTH,
      HEIGHT,
      index * 100
    );
  }
  assert.equal(status.isStable, true, 'three consistent card frames should be stable');

  status = tracker.update(
    createCardResult(shiftPoints(TRACKING_CARD_CORNERS, 55, -28)),
    WIDTH,
    HEIGHT,
    300
  );
  assert.equal(status.isStable, false, 'camera/card translation must invalidate card stability');
  assert.equal(status.sampleCount, 1);
  assert.equal(status.resetRequired, true);

  tracker.reset();
  tracker.update(createCardResult(TRACKING_CARD_CORNERS), WIDTH, HEIGHT, 0);
  tracker.update(createCardResult(TRACKING_CARD_CORNERS), WIDTH, HEIGHT, 100);
  tracker.update(createCardResult(TRACKING_CARD_CORNERS), WIDTH, HEIGHT, 200);
  status = tracker.update(
    createCardResult(scalePoints(TRACKING_CARD_CORNERS, 1.24)),
    WIDTH,
    HEIGHT,
    300
  );
  assert.equal(status.isStable, false, 'camera zoom/scale change must invalidate card stability');
  assert.equal(status.resetRequired, true);
  assert.equal(status.sampleCount, 1);

  tracker.reset();
  for (let index = 0; index < 3; index++) {
    status = tracker.update(
      createCardResult(shiftPoints(TRACKING_CARD_CORNERS, index * 10, 0)),
      WIDTH,
      HEIGHT,
      index * 100
    );
  }
  assert.equal(status.isStable, true, 'small motion within one window may stabilize');
  status = tracker.update(
    createCardResult(shiftPoints(TRACKING_CARD_CORNERS, 30, 0)),
    WIDTH,
    HEIGHT,
    300
  );
  assert.equal(status.resetRequired, true, 'cumulative camera drift must reset the anchor');
  assert.equal(status.sampleCount, 1);

  tracker.reset();
  status = tracker.update(createCardResult(TRACKING_CARD_CORNERS), WIDTH, HEIGHT, 0);
  status = tracker.update(createCardResult(TRACKING_CARD_CORNERS), WIDTH, HEIGHT, 0);
  assert.equal(status.sampleCount, 1, 'duplicate timestamps must not count as multiple card frames');

  tracker.reset();
  const invalidCard = createCardResult(TRACKING_CARD_CORNERS, {
    measurementReady: false,
    quality: 'low',
    issue: 'too-small',
  });
  tracker.update(invalidCard, WIDTH, HEIGHT, 0);
  tracker.update(invalidCard, WIDTH, HEIGHT, 100);
  status = tracker.update(invalidCard, WIDTH, HEIGHT, 200);
  assert.equal(status.isStable, true, 'a stable invalid card may support only a fallback preview');
  assert.equal(status.issue, 'too-small');
  status = tracker.update(createCardResult(TRACKING_CARD_CORNERS), WIDTH, HEIGHT, 300);
  assert.equal(status.resetRequired, true, 'invalid card frames must not pre-validate a good card');
  assert.equal(status.sampleCount, 1);
  assert.equal(status.isStable, false);

  assert.equal(tracker.update(null, WIDTH, HEIGHT, 400).sampleCount, 1);
  status = tracker.update(null, WIDTH, HEIGHT, 500);
  assert.equal(status.sampleCount, 0, 'two card misses should clear the track');
  assert.equal(status.resetRequired, true);
}

function testFallbackAndCaptureGate() {
  const malformedTracker = new FootTracker();
  const malformed = malformedTracker.update(createTrackedFoot(), {
    mode: 'homography',
    ppm: 2,
    confidence: 0.95,
    homography: null,
    fallbackReference: TRACKING_REFERENCE,
  }, 0);
  assert.equal(malformed.status.sampleCount, 0, 'a homography-labelled sample must contain a homography');

  const fallbackTracker = new FootTracker();
  const fallbackCalibration = {
    mode: 'ppm-fallback',
    ppm: 2,
    confidence: 0.95,
    homography: null,
    fallbackReference: TRACKING_REFERENCE,
  };
  let update = null;
  for (let index = 0; index < 7; index++) {
    update = fallbackTracker.update(createTrackedFoot(), fallbackCalibration, index * 80);
  }
  assert.equal(update.status.isStable, true, 'PPM fallback may stabilize for preview');
  assert.ok(update.measurement);
  assert.equal(update.measurement.calibrationMode, 'ppm-fallback');
  assert.equal(update.measurement.quality, 'low');
  assert.ok(update.measurement.confidence <= 0.45);

  const fallbackResult = buildMeasurementResult(520, 200, 2);
  assert.ok(fallbackResult);
  assert.equal(fallbackResult.homographyValid, false);
  assert.equal(fallbackResult.shoeSize, null, 'fallback must not show a shoe recommendation');
  assert.equal(canCaptureMeasurement({
    cameraReady: true,
    cardStable: true,
    footDetected: true,
    homographyValid: false,
    trackingReady: true,
    measurement: fallbackResult,
  }), false, 'PPM fallback must never pass the capture gate');

  const trusted = buildMeasurementResultFromMillimeters(260, 100, 2, {
    confidence: 0.88,
    quality: 'high',
    calibrationMode: 'homography',
    homographyValid: true,
    warnings: [],
  });
  assert.ok(trusted?.shoeSize);
  const outsideShoeTable = buildMeasurementResultFromMillimeters(180, 70, 2, {
    confidence: 0.88,
    quality: 'high',
    calibrationMode: 'homography',
    homographyValid: true,
    warnings: [],
  });
  assert.ok(outsideShoeTable);
  assert.equal(outsideShoeTable.shoeSize, null, 'out-of-range feet must not be clamped to a shoe size');
  assert.ok(outsideShoeTable.warnings.length > 0);
  const validGate = {
    cameraReady: true,
    cardStable: true,
    footDetected: true,
    homographyValid: true,
    trackingReady: true,
    measurement: trusted,
  };
  assert.equal(canCaptureMeasurement(validGate), true, 'trusted homography result should be capturable');
  assert.equal(canCaptureMeasurement({ ...validGate, trackingReady: false }), false);
  assert.equal(canCaptureMeasurement({ ...validGate, homographyValid: false }), false);
  assert.equal(canCaptureMeasurement({ ...validGate, cardStable: false }), false);
  assert.equal(canCaptureMeasurement({ ...validGate, cameraReady: false }), false);
}

function testObjectCoverMapping() {
  for (const [displayWidth, displayHeight] of [[390, 844], [844, 390]]) {
    const guides = calculateDisplayGuideRects(displayWidth, displayHeight);
    assert.ok(guides);
    const landscapeAspect =
      (guides.cardLandscape.width * displayWidth) /
      (guides.cardLandscape.height * displayHeight);
    const portraitAspect =
      (guides.cardPortrait.width * displayWidth) /
      (guides.cardPortrait.height * displayHeight);
    assertNear(landscapeAspect, 85.6 / 53.98, 1e-9, 'landscape guide aspect');
    assertNear(portraitAspect, 53.98 / 85.6, 1e-9, 'portrait guide aspect');
    assertNear(
      guides.card.width * displayWidth,
      guides.card.height * displayHeight,
      1e-9,
      'card orientation zone should be square in display pixels'
    );
  }

  const landscapeSource = calculateObjectCoverGeometry(1920, 1080, 400, 600);
  assert.ok(landscapeSource);
  assertNear(landscapeSource.scale, 600 / 1080, 1e-9, 'landscape object-cover scale');
  assertNear(landscapeSource.offsetX, -333.33333333333337, 1e-6, 'landscape crop x');
  assert.deepEqual(sourcePointToDisplay(landscapeSource, { x: 960, y: 540 }), { x: 200, y: 300 });
  const visible = displayRectToSource(landscapeSource, { x: 0, y: 0, width: 1, height: 1 });
  assertNear(visible.x, 600, 1e-6, 'visible source x');
  assertNear(visible.y, 0, 1e-6, 'visible source y');
  assertNear(visible.width, 720, 1e-6, 'visible source width');
  assertNear(visible.height, 1080, 1e-6, 'visible source height');

  const portraitSource = calculateObjectCoverGeometry(1080, 1920, 400, 600);
  assert.ok(portraitSource);
  const portraitVisible = displayRectToSource(portraitSource, { x: 0, y: 0, width: 1, height: 1 });
  assertNear(portraitVisible.x, 0, 1e-6, 'portrait visible x');
  assertNear(portraitVisible.y, 150, 1e-6, 'portrait crop y');
  assertNear(portraitVisible.width, 1080, 1e-6, 'portrait visible width');
  assertNear(portraitVisible.height, 1620, 1e-6, 'portrait visible height');
  assert.equal(calculateObjectCoverGeometry(0, 1080, 400, 600), null);
}

function testMediaStreamCleanupAndCameraErrors() {
  const stops = [0, 0, 0];
  stopMediaStream({
    getTracks: () => stops.map((_, index) => ({ stop: () => { stops[index]++; } })),
  });
  assert.deepEqual(stops, [1, 1, 1], 'every MediaStream track must be stopped');
  const resilientStops = [0, 0, 0];
  assert.doesNotThrow(() => stopMediaStream({
    getTracks: () => resilientStops.map((_, index) => ({
      stop: () => {
        resilientStops[index]++;
        if (index === 1) throw new Error('simulated track stop failure');
      },
    })),
  }));
  assert.deepEqual(resilientStops, [1, 1, 1], 'cleanup must continue after one track throws');
  assert.doesNotThrow(() => stopMediaStream(null));

  const cases = [
    ['NotAllowedError', 'permission-denied'],
    ['SecurityError', 'permission-denied'],
    ['NotFoundError', 'not-found'],
    ['OverconstrainedError', 'not-found'],
    ['NotReadableError', 'in-use'],
    ['TrackStartError', 'in-use'],
    ['AbortError', 'in-use'],
    ['TypeError', 'unsupported'],
    ['UnexpectedError', 'unknown'],
  ];
  for (const [name, expectedCode] of cases) {
    assert.equal(classifyCameraError({ name }).code, expectedCode, `${name} classification`);
  }
  assert.equal(classifyCameraError({ name: 'NotAllowedError' }, false).code, 'insecure-context');
  const metadataTimeout = new CameraSetupError('metadata-timeout', 'timeout');
  assert.deepEqual(classifyCameraError(metadataTimeout), {
    code: 'metadata-timeout',
    message: 'timeout',
  });
}

function testSegmenterResultCleanup() {
  let closeCount = 0;
  const wasmOwned = new Float32Array([0.1, 0.8, 0.2, 0.9]);
  const copied = copyFootMaskAndClose({
    confidenceMasks: [
      { width: 2, height: 2, getAsFloat32Array: () => new Float32Array(4) },
      { width: 2, height: 2, getAsFloat32Array: () => wasmOwned },
    ],
    close: () => { closeCount++; },
  });
  assert.ok(copied);
  wasmOwned[0] = 1;
  assertNear(copied.data[0], 0.1, 1e-6, 'segmenter result must be copied before close');
  assert.equal(closeCount, 1, 'successful result must close once');

  closeCount = 0;
  assert.equal(copyFootMaskAndClose({ confidenceMasks: [], close: () => { closeCount++; } }), null);
  assert.equal(closeCount, 1, 'missing foot mask must still close result');

  closeCount = 0;
  assert.throws(() => copyFootMaskAndClose({
    confidenceMasks: [
      { width: 1, height: 1, getAsFloat32Array: () => new Float32Array(1) },
      { width: 1, height: 1, getAsFloat32Array: () => { throw new Error('WASM read failed'); } },
    ],
    close: () => { closeCount++; },
  }), /WASM read failed/);
  assert.equal(closeCount, 1, 'throwing mask read must still close result');
}

function testFrameProcessingBackpressure() {
  const gate = new FrameProcessingGate();
  assert.equal(gate.tryStart(0), true);
  assert.equal(gate.isBusy(), true);
  assert.equal(gate.tryStart(100), false, 'processing lock must reject overlapping inference');
  gate.finish(200);
  assert.equal(gate.isBusy(), false);
  assert.equal(gate.getRecommendedIntervalMs(), 240, 'slow work should increase interval to cap');
  assert.equal(gate.tryStart(200), false, 'adaptive cadence should delay the next frame');
  assert.equal(gate.tryStart(240), true);
  gate.finish(20);
  assert.ok(gate.getRecommendedIntervalMs() > 80, 'slow EWMA should keep backpressure active');
  gate.reset();
  assert.equal(gate.getRecommendedIntervalMs(), 80);
  assert.equal(gate.isBusy(), false);
}

testLandscapeCardAndFoot();
testPortraitCardScale();
testRotatedCard();
testPerspectiveCardDetection();
testWrongAspectRectangleRejected();
testReferenceCardMeasurementGates();
testBlankFrame();
testLandscapePortraitAndPerspectiveHomography();
testRotatedFootAndPerpendicularWidth();
testPerspectiveCorrectedFootFixture();
testLetterboxGeometryAndModelMask();
testBundledFootModelAssets();
testSevenFrameFootTracking();
testOutlierAndMissHandling();
testCoherentMovementRebasesTrack();
testTimingGapAndManualReset();
testNoisyWindowWaitsForConsistency();
testCardTrackingRejectsCameraMovement();
testFallbackAndCaptureGate();
testObjectCoverMapping();
testMediaStreamCleanupAndCameraErrors();
testSegmenterResultCleanup();
testFrameProcessingBackpressure();

const syntheticLengthMae = syntheticLengthErrors.reduce((sum, value) => sum + value, 0) /
  syntheticLengthErrors.length;
const syntheticWidthMae = syntheticWidthErrors.reduce((sum, value) => sum + value, 0) /
  syntheticWidthErrors.length;
console.log(`Synthetic fixture length MAE: ${syntheticLengthMae.toFixed(3)} mm`);
console.log(`Synthetic fixture width MAE: ${syntheticWidthMae.toFixed(3)} mm`);
console.log('Synthetic detection, calibration, cleanup, and tracking smoke tests passed.');
