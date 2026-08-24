// ─── Application constants ──────────────────────────────────────────────────

// Standard ISO/IEC 7810 ID-1 card dimensions (credit card, ID card)
export const CARD_WIDTH_MM = 85.60;
export const CARD_HEIGHT_MM = 53.98;
export const CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM; // ~1.586

// Detection thresholds
export const CARD_MIN_AREA_RATIO = 0.005; // card must be at least 0.5% of frame area
export const CARD_MAX_AREA_RATIO = 0.60;  // card can't be more than 60% of frame
export const CARD_MIN_MEASUREMENT_AREA_RATIO = 0.012;
export const CARD_MAX_MEASUREMENT_AREA_RATIO = 0.18;
export const CARD_CONFIDENCE_THRESHOLD = 0.64;
export const CARD_MIN_EDGE_CONSISTENCY = 0.68;
export const CARD_MIN_PERSPECTIVE_SCORE = 0.62;
export const CARD_TRACK_TARGET_SAMPLES = 3;
export const CARD_TRACK_MIN_SAMPLE_INTERVAL_MS = 80;
export const CARD_TRACK_MAX_GAP_MS = 650;
export const CARD_TRACK_MAX_CENTER_SHIFT_RATIO = 0.035;
export const CARD_TRACK_MAX_SCALE_SHIFT = 0.08;
export const CARD_TRACK_MAX_ANGLE_SHIFT_RAD = (8 * Math.PI) / 180;

// MediaPipe Pose landmark indices
export const POSE_LANDMARKS = {
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

// ROI padding around foot landmarks (in pixels, before scaling)
export const FOOT_ROI_PADDING = 0.05; // 5% of image dimension
export const FOOT_MAX_CARD_DISTANCE_MM = 500;

// Custom FootNet segmenter deployment contract.
export const FOOT_SEGMENTER_MODEL_PATH = '/models/foot_segmenter_fp16_metadata.tflite';
export const FOOT_SEGMENTER_WASM_PATH = '/mediapipe/wasm';
export const FOOT_SEGMENTER_IMAGE_SIZE = 384;
export const FOOT_SEGMENTER_THRESHOLD = 0.5;

// Frame processing settings
export const PROCESS_MIN_INTERVAL_MS = 80;
export const PROCESS_MAX_INTERVAL_MS = 240;
export const PROCESS_TIME_SAFETY_FACTOR = 1.35;

// Placement zones are normalized display coordinates and are mapped back
// through the video's object-cover crop before detection/inference.
export const FOOT_GUIDE_RECT = { x: 110 / 640, y: 50 / 480, width: 330 / 640, height: 370 / 480 };
export const CARD_GUIDE_LAYOUT = {
  x: 460 / 640,
  bottom: 30 / 480,
  width: 155 / 640,
} as const;

// Temporal foot tracking: seven consistent observations over roughly 0.5 s.
export const FOOT_TRACK_TARGET_SAMPLES = 7;
export const FOOT_TRACK_MIN_SAMPLE_INTERVAL_MS = 80;
export const FOOT_TRACK_MIN_DURATION_MS = 480;
export const FOOT_TRACK_MAX_GAP_MS = 750;
export const FOOT_TRACK_MAX_MISSES = 3;
export const FOOT_TRACK_RESET_AFTER_OUTLIERS = 2;

// Shoe size lookup table: [foot length in mm, EU, US Men, US Women, UK]
export interface ShoeSizeEntry {
  lengthMm: number;
  eu: string;
  usMen: string;
  usWomen: string;
  uk: string;
}

export const SHOE_SIZE_TABLE: ShoeSizeEntry[] = [
  { lengthMm: 215, eu: '34',   usMen: '3',    usWomen: '4.5', uk: '2' },
  { lengthMm: 220, eu: '35',   usMen: '3.5',  usWomen: '5',   uk: '2.5' },
  { lengthMm: 225, eu: '36',   usMen: '4',    usWomen: '5.5', uk: '3' },
  { lengthMm: 230, eu: '37',   usMen: '4.5',  usWomen: '6',   uk: '3.5' },
  { lengthMm: 235, eu: '37',   usMen: '5',    usWomen: '6.5', uk: '4' },
  { lengthMm: 240, eu: '38',   usMen: '5.5',  usWomen: '7',   uk: '4.5' },
  { lengthMm: 245, eu: '39',   usMen: '6',    usWomen: '7.5', uk: '5' },
  { lengthMm: 250, eu: '39',   usMen: '6.5',  usWomen: '8',   uk: '5.5' },
  { lengthMm: 255, eu: '40',   usMen: '7',    usWomen: '8.5', uk: '6' },
  { lengthMm: 260, eu: '41',   usMen: '7.5',  usWomen: '9',   uk: '6.5' },
  { lengthMm: 265, eu: '42',   usMen: '8',    usWomen: '9.5', uk: '7' },
  { lengthMm: 270, eu: '42',   usMen: '8.5',  usWomen: '10',  uk: '7.5' },
  { lengthMm: 275, eu: '43',   usMen: '9',    usWomen: '10.5',uk: '8' },
  { lengthMm: 280, eu: '44',   usMen: '9.5',  usWomen: '11',  uk: '8.5' },
  { lengthMm: 285, eu: '44',   usMen: '10',   usWomen: '11.5',uk: '9' },
  { lengthMm: 290, eu: '45',   usMen: '10.5', usWomen: '12',  uk: '9.5' },
  { lengthMm: 295, eu: '46',   usMen: '11',   usWomen: '12.5',uk: '10' },
  { lengthMm: 300, eu: '47',   usMen: '11.5', usWomen: '13',  uk: '10.5' },
  { lengthMm: 305, eu: '48',   usMen: '12',   usWomen: '13.5',uk: '11' },
  { lengthMm: 310, eu: '48',   usMen: '12.5', usWomen: '14',  uk: '11.5' },
  { lengthMm: 315, eu: '49',   usMen: '13',   usWomen: '14.5',uk: '12' },
];
