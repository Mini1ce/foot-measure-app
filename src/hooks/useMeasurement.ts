'use client';
/* eslint-disable @typescript-eslint/no-explicit-any -- OpenCV.js exposes an untyped global runtime. */

import { useCallback, useRef } from 'react';
import type { PoseLandmarkerResult } from '@mediapipe/tasks-vision';
import type {
  CardDetectionResult,
  CardTrackingStatus,
  FootContourResult,
  FootSegmentationMask,
  FootTrackingStatus,
  GuidanceCode,
  MeasurementReadiness,
  MeasurementResult,
} from '@/types';
import { detectCard, drawCardOverlay } from '@/lib/opencv/cardDetector';
import {
  detectFootContour,
  detectFootContourFromModelMask,
  drawFootOverlay,
} from '@/lib/opencv/footContour';
import { prepareFootSegmenterInput } from '@/lib/mediapipe/footSegmenter';
import { calculateFootROI, determineBetterFoot } from '@/lib/mediapipe/poseProcessor';
import {
  buildMeasurementResultFromMillimeters,
  isValidPPM,
} from '@/lib/calculations/dimensions';
import { canCaptureMeasurement } from '@/lib/calculations/measurementPolicy';
import {
  cardTrackingReference,
  FootFrameCalibration,
  FootTracker,
} from '@/lib/tracking/footTracker';
import { CardTracker } from '@/lib/tracking/cardTracker';
import {
  calculateObjectCoverGeometry,
  calculateDisplayGuideRects,
  displayRectToSource,
} from '@/lib/camera/objectCover';
import {
  CARD_CONFIDENCE_THRESHOLD,
  FOOT_SEGMENTER_THRESHOLD,
} from '@/constants';
import { FrameProcessingGate } from '@/lib/performance/frameProcessingGate';

declare const cv: any;

interface OpenCvFrame {
  data: Uint8Array;
  delete(): void;
}

interface ProcessFrameResult {
  processed: boolean;
  cardDetection: CardDetectionResult | null;
  cardTracking: CardTrackingStatus | null;
  footContour: FootContourResult | null;
  trackingStatus: FootTrackingStatus | null;
  measurement: MeasurementResult | null;
  readiness: MeasurementReadiness | null;
}

function unprocessedFrame(): ProcessFrameResult {
  return {
    processed: false,
    cardDetection: null,
    cardTracking: null,
    footContour: null,
    trackingStatus: null,
    measurement: null,
    readiness: null,
  };
}

function averageLuminance(src: OpenCvFrame): number {
  const data = src.data;
  const pixelCount = Math.max(1, data.length / 4);
  const stepPixels = Math.max(1, Math.floor(pixelCount / 1200));
  let total = 0;
  let samples = 0;
  for (let pixel = 0; pixel < pixelCount; pixel += stepPixels) {
    const index = pixel * 4;
    total += data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722;
    samples++;
  }
  return samples > 0 ? total / samples : 0;
}

function cardGuidance(result: CardDetectionResult | null): GuidanceCode | null {
  switch (result?.issue) {
    case 'too-small': return 'card-too-small';
    case 'too-large': return 'card-too-large';
    case 'too-tilted': return 'camera-tilted';
    case 'partially-occluded': return 'card-occluded';
    case 'wrong-shape': return 'card-wrong-shape';
    case 'outside-guide': return 'card-outside-guide';
    case 'homography-failed': return 'homography-failed';
    case 'not-found':
    default: return result?.detected ? null : 'card-not-found';
  }
}

function createReadiness(
  card: CardDetectionResult | null,
  cardTracking: CardTrackingStatus,
  foot: FootContourResult | null,
  tracking: FootTrackingStatus,
  measurement: MeasurementResult | null,
  lowLight: boolean,
  footOutsideGuide: boolean
): MeasurementReadiness {
  const guidance: GuidanceCode[] = [];
  if (lowLight) guidance.push('low-light');
  const cardAdvice = cardGuidance(card);
  if (cardAdvice) guidance.push(cardAdvice);
  if (card?.detected && !cardTracking.isStable) guidance.push('card-unstable');
  if (!foot?.detected) guidance.push('foot-not-found');
  else if (footOutsideGuide) guidance.push('foot-outside-guide');
  if (foot?.detected && !tracking.isStable) guidance.push('hold-still');
  if (tracking.calibrationMode === 'ppm-fallback') guidance.push('ppm-fallback');

  const homographyValid =
    !!card?.measurementReady &&
    !!card.homography &&
    cardTracking.isStable;
  const canCapture = canCaptureMeasurement({
    cameraReady: true,
    cardStable: cardTracking.isStable,
    footDetected: !!foot?.detected,
    homographyValid,
    trackingReady: tracking.isStable,
    measurement,
  });
  return {
    cameraReady: true,
    cardDetected: !!card?.detected,
    cardStable: cardTracking.isStable,
    footDetected: !!foot?.detected,
    homographyValid,
    trackingReady: tracking.isStable,
    canCapture,
    quality: measurement?.quality ?? tracking.quality,
    confidence: measurement?.confidence ?? tracking.confidence,
    guidance: [...new Set(guidance)],
    processingTimeMs: 0,
  };
}

/** Orchestrate card calibration, foot segmentation, and robust 7-frame tracking. */
export function useMeasurement() {
  const footTrackerRef = useRef<FootTracker | null>(null);
  const cardTrackerRef = useRef<CardTracker | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const segmenterCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastFrameSizeRef = useRef('');
  const processingGateRef = useRef<FrameProcessingGate | null>(null);
  if (footTrackerRef.current === null) footTrackerRef.current = new FootTracker();
  if (cardTrackerRef.current === null) cardTrackerRef.current = new CardTracker();
  if (processingGateRef.current === null) processingGateRef.current = new FrameProcessingGate();

  const resetTracking = useCallback(() => {
    footTrackerRef.current?.reset();
    cardTrackerRef.current?.reset();
    lastFrameSizeRef.current = '';
    processingGateRef.current?.reset();
  }, []);

  const processFrame = useCallback(
    (
      video: HTMLVideoElement,
      overlayCanvas: HTMLCanvasElement,
      detectPoseFn: ((video: HTMLVideoElement, timestamp: number) => PoseLandmarkerResult | null) | null,
      segmentFootFn: ((image: HTMLCanvasElement, timestamp: number) => FootSegmentationMask | null) | null,
      opencvReady: boolean,
      poseReady: boolean,
      footSegmenterReady: boolean
    ): ProcessFrameResult => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const schedulingTime = performance.now();
      const processingGate = processingGateRef.current!;
      if (vw <= 0 || vh <= 0 || !processingGate.tryStart(schedulingTime)) return unprocessedFrame();
      const startedAt = performance.now();
      const footTracker = footTrackerRef.current!;
      const cardTracker = cardTrackerRef.current!;
      const output = unprocessedFrame();
      output.processed = true;
      let src: OpenCvFrame | null = null;

      try {
        const trackerTimestamp = video.currentTime > 0 ? video.currentTime * 1000 : schedulingTime;
        if (overlayCanvas.width !== vw) overlayCanvas.width = vw;
        if (overlayCanvas.height !== vh) overlayCanvas.height = vh;
        const overlayContext = overlayCanvas.getContext('2d');
        if (!overlayContext || !opencvReady || typeof cv === 'undefined' || !cv.Mat) {
          footTracker.reset();
          cardTracker.reset();
          output.trackingStatus = footTracker.getStatus();
          output.cardTracking = cardTracker.getStatus();
          return output;
        }
        overlayContext.clearRect(0, 0, vw, vh);

        const frameSize = `${vw}x${vh}`;
        if (lastFrameSizeRef.current && lastFrameSizeRef.current !== frameSize) {
          footTracker.reset();
          cardTracker.reset();
        }
        lastFrameSizeRef.current = frameSize;

        if (!frameCanvasRef.current) frameCanvasRef.current = document.createElement('canvas');
        const frameCanvas = frameCanvasRef.current;
        if (frameCanvas.width !== vw) frameCanvas.width = vw;
        if (frameCanvas.height !== vh) frameCanvas.height = vh;
        const frameContext = frameCanvas.getContext('2d', { willReadFrequently: true });
        if (!frameContext) return output;
        frameContext.drawImage(video, 0, 0, vw, vh);
        const currentFrame = cv.imread(frameCanvas) as OpenCvFrame;
        src = currentFrame;
        const lowLight = averageLuminance(currentFrame) < 55;

        const displayWidth = Math.max(1, overlayCanvas.clientWidth || vw);
        const displayHeight = Math.max(1, overlayCanvas.clientHeight || vh);
        const cover = calculateObjectCoverGeometry(vw, vh, displayWidth, displayHeight);
        const guides = calculateDisplayGuideRects(displayWidth, displayHeight);
        const footGuideRoi = cover && guides
          ? displayRectToSource(cover, guides.foot)
          : { x: vw * 0.08, y: vh * 0.04, width: vw * 0.68, height: vh * 0.91 };
        const cardGuideRoi = cover && guides
          ? displayRectToSource(cover, guides.card)
          : null;

        const cardDetection = detectCard(src, vw, vh, { guideRect: cardGuideRoi });
        output.cardDetection = cardDetection;
        drawCardOverlay(overlayContext, cardDetection, 1, 1);
        const cardTracking = cardTracker.update(cardDetection, vw, vh, trackerTimestamp);
        output.cardTracking = cardTracking;
        if (cardTracking.resetRequired) footTracker.reset();

        let footRoi = footGuideRoi;
        const inferenceTimestamp = performance.now();
        if (poseReady && detectPoseFn) {
          const poseResult = detectPoseFn(video, inferenceTimestamp);
          const landmarks = poseResult?.landmarks?.[0];
          if (landmarks) {
            const side = determineBetterFoot(landmarks);
            footRoi = calculateFootROI(landmarks, vw, vh, side) ?? footRoi;
          }
        }

        let footContour: FootContourResult | null = null;
        const geometryHomography = cardDetection.measurementReady
          ? cardDetection.homography
          : null;
        if (footSegmenterReady && segmentFootFn) {
          if (!segmenterCanvasRef.current) segmenterCanvasRef.current = document.createElement('canvas');
          const prepared = prepareFootSegmenterInput(
            frameCanvas,
            segmenterCanvasRef.current,
            footRoi,
            vw,
            vh
          );
          if (prepared) {
            const segmentation = segmentFootFn(prepared.canvas, inferenceTimestamp);
            if (segmentation) {
              footContour = detectFootContourFromModelMask(
                segmentation,
                prepared.roi,
                prepared.placement,
                vw,
                vh,
                FOOT_SEGMENTER_THRESHOLD,
                cardDetection.corners,
                geometryHomography
              );
            }
          }
        }
        if (!footContour?.detected) {
          footContour = detectFootContour(
            src,
            footRoi,
            vw,
            vh,
            cardDetection.corners,
            geometryHomography
          );
        }
        output.footContour = footContour;
        drawFootOverlay(overlayContext, footContour, 1, 1);
        const footMargin = Math.max(
          4,
          Math.min(footGuideRoi.width, footGuideRoi.height) * 0.015
        );
        const footOutsideGuide = !!footContour?.detected && footContour.contourPoints.some(
          (point) =>
            point.x <= footGuideRoi.x + footMargin ||
            point.y <= footGuideRoi.y + footMargin ||
            point.x >= footGuideRoi.x + footGuideRoi.width - footMargin ||
            point.y >= footGuideRoi.y + footGuideRoi.height - footMargin
        );

        const ppmUsable =
          cardDetection.detected &&
          cardDetection.confidence >= CARD_CONFIDENCE_THRESHOLD &&
          isValidPPM(cardDetection.ppm) &&
          cardTracking.isStable;
        let calibration: FootFrameCalibration | null = null;
        if (ppmUsable && !lowLight && !footOutsideGuide) {
          const fallbackReference = cardTrackingReference(cardDetection.corners);
          const homographyReady = cardDetection.measurementReady && !!cardDetection.homography;
          const confidence = Math.min(
            cardDetection.confidence,
            cardDetection.edgeConsistency,
            cardDetection.perspectiveScore,
            cardTracking.consistency
          );
          calibration = homographyReady && cardDetection.homography
            ? {
                mode: 'homography',
                ppm: cardDetection.ppm,
                confidence,
                homography: cardDetection.homography,
                fallbackReference: null,
              }
            : fallbackReference
              ? {
                  mode: 'ppm-fallback',
                  ppm: cardDetection.ppm,
                  confidence: Math.min(0.45, confidence),
                  homography: null,
                  fallbackReference,
                }
              : null;
        }

        const trackingUpdate = footTracker.update(footContour, calibration, trackerTimestamp);
        output.trackingStatus = trackingUpdate.status;
        let measurement: MeasurementResult | null = null;
        if (trackingUpdate.measurement && trackingUpdate.stabilizedFoot) {
          const tracked = trackingUpdate.measurement;
          measurement = buildMeasurementResultFromMillimeters(
            tracked.lengthMm,
            tracked.widthMm,
            tracked.ppm,
            {
              confidence: tracked.confidence,
              quality: tracked.quality,
              calibrationMode: tracked.calibrationMode,
              homographyValid: tracked.homographyValid,
              warnings: tracked.homographyValid
                ? []
                : ['กำลังใช้ PPM สำรอง ต้องจัดกล้องและบัตรใหม่ก่อนบันทึกผล'],
            }
          );
        }
        output.measurement = measurement;
        output.readiness = createReadiness(
          cardDetection,
          cardTracking,
          footContour,
          trackingUpdate.status,
          measurement,
          lowLight,
          footOutsideGuide
        );
      } catch (error) {
        console.warn('[Measurement] Skipped frame after processing error:', error);
        const timestamp = video.currentTime > 0 ? video.currentTime * 1000 : schedulingTime;
        const cardStatus = cardTracker.update(null, vw, vh, timestamp);
        if (cardStatus.resetRequired) footTracker.reset();
        output.cardTracking = cardStatus;
        output.trackingStatus = footTracker.update(null, null, timestamp).status;
      } finally {
        src?.delete();
        const processingTime = performance.now() - startedAt;
        processingGate.finish(processingTime);
        if (output.readiness) output.readiness.processingTimeMs = processingTime;
      }
      return output;
    },
    []
  );

  return { processFrame, resetTracking };
}
