// ─── Core measurement types ────────────────────────────────────────────────

export type AppStep = 1 | 2 | 3;

export type QualityStatus = 'low' | 'medium' | 'high';
export type CalibrationMode = 'homography' | 'ppm-fallback';
export type CardIssue =
  | 'not-found'
  | 'too-small'
  | 'too-large'
  | 'too-tilted'
  | 'partially-occluded'
  | 'wrong-shape'
  | 'outside-guide'
  | 'homography-failed'
  | 'unstable'
  | null;

export interface HomographyTransform {
  matrix: [number, number, number, number, number, number, number, number, number];
  cardOrientation: 'landscape' | 'portrait';
  destinationWidthMm: number;
  destinationHeightMm: number;
  reprojectionErrorMm: number;
}

export interface CardDetectionResult {
  detected: boolean;
  confidence: number; // 0–1
  widthPx: number;
  heightPx: number;
  corners: Point[] | null;
  ppm: number; // pixels per millimeter
  homography: HomographyTransform | null;
  measurementReady: boolean;
  quality: QualityStatus;
  issue: CardIssue;
  edgeConsistency: number;
  perspectiveScore: number;
  areaRatio: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface FootContourResult {
  detected: boolean;
  lengthPx: number;
  widthPx: number;
  contourPoints: Point[];
  heelPoint: Point | null;
  toePoint: Point | null;
  widestPoints: [Point, Point] | null;
  geometryConfidence: number;
  shapeSignature: number[];
}

export type FootTrackingPhase = 'searching' | 'tracking' | 'stabilizing' | 'stable';

export interface FootTrackingStatus {
  sampleCount: number;
  targetSamples: number;
  isStable: boolean;
  consistency: number; // 0–1
  phase: FootTrackingPhase;
  confidence: number;
  quality: QualityStatus;
  calibrationMode: CalibrationMode | null;
}

export interface CardTrackingStatus {
  sampleCount: number;
  targetSamples: number;
  isStable: boolean;
  consistency: number;
  issue: CardIssue;
  resetRequired: boolean;
}

export interface MeasurementResult {
  footLengthCm: number;
  footWidthCm: number;
  ppm: number;
  timestamp: number;
  confidence: number;
  quality: QualityStatus;
  calibrationMode: CalibrationMode;
  homographyValid: boolean;
  warnings: string[];
  shoeSize: ShoeSizeRecommendation | null;
}

export type CameraErrorCode =
  | 'insecure-context'
  | 'permission-denied'
  | 'not-found'
  | 'in-use'
  | 'metadata-timeout'
  | 'playback-failed'
  | 'unsupported'
  | 'unknown';

export interface CameraErrorInfo {
  code: CameraErrorCode;
  message: string;
}

export type GuidanceCode =
  | 'low-light'
  | 'card-not-found'
  | 'card-too-small'
  | 'card-too-large'
  | 'camera-tilted'
  | 'card-occluded'
  | 'card-wrong-shape'
  | 'card-outside-guide'
  | 'homography-failed'
  | 'card-unstable'
  | 'foot-not-found'
  | 'foot-outside-guide'
  | 'hold-still'
  | 'ppm-fallback';

export interface MeasurementReadiness {
  cameraReady: boolean;
  cardDetected: boolean;
  cardStable: boolean;
  footDetected: boolean;
  homographyValid: boolean;
  trackingReady: boolean;
  canCapture: boolean;
  quality: QualityStatus;
  confidence: number;
  guidance: GuidanceCode[];
  processingTimeMs: number;
}

export interface ShoeSizeRecommendation {
  eu: string;
  usMen: string;
  usWomen: string;
  uk: string;
  fit: 'exact' | 'half-size-up' | 'half-size-down';
}

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface FootROI {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FootSegmentationMask {
  data: Float32Array;
  width: number;
  height: number;
}

export interface LetterboxPlacement {
  x: number;
  y: number;
  width: number;
  height: number;
  targetSize: number;
}

export interface AppState {
  step: AppStep;
  isLoading: boolean;
  opencvReady: boolean;
  mediapipeReady: boolean;
  cameraActive: boolean;
  cardDetection: CardDetectionResult | null;
  footContour: FootContourResult | null;
  measurement: MeasurementResult | null;
  error: string | null;
}

export type AppAction =
  | { type: 'SET_STEP'; payload: AppStep }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_OPENCV_READY'; payload: boolean }
  | { type: 'SET_MEDIAPIPE_READY'; payload: boolean }
  | { type: 'SET_CAMERA_ACTIVE'; payload: boolean }
  | { type: 'SET_CARD_DETECTION'; payload: CardDetectionResult | null }
  | { type: 'SET_FOOT_CONTOUR'; payload: FootContourResult | null }
  | { type: 'SET_MEASUREMENT'; payload: MeasurementResult | null }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET' };
