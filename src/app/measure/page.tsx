'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AppStep,
  CardDetectionResult,
  CardTrackingStatus,
  FootContourResult,
  FootTrackingStatus,
  MeasurementResult,
  MeasurementReadiness,
} from '@/types';
import { useCamera } from '@/hooks/useCamera';
import { useOpenCV } from '@/hooks/useOpenCV';
import { useMediaPipe } from '@/hooks/useMediaPipe';
import { useMeasurement } from '@/hooks/useMeasurement';
import { StepIndicator } from '@/components/ui/StepIndicator';
import { Step1Instructions } from '@/components/steps/Step1Instructions';
import { Step2LiveView } from '@/components/steps/Step2LiveView';
import { Step3Results } from '@/components/steps/Step3Results';
import { initialFootTrackingStatus } from '@/lib/tracking/footTracker';
import { CARD_TRACK_TARGET_SAMPLES } from '@/constants';

function initialCardTrackingStatus(): CardTrackingStatus {
  return {
    sampleCount: 0,
    targetSamples: CARD_TRACK_TARGET_SAMPLES,
    isStable: false,
    consistency: 0,
    issue: 'not-found',
    resetRequired: false,
  };
}

function initialReadiness(): MeasurementReadiness {
  return {
    cameraReady: false,
    cardDetected: false,
    cardStable: false,
    footDetected: false,
    homographyValid: false,
    trackingReady: false,
    canCapture: false,
    quality: 'low',
    confidence: 0,
    guidance: [],
    processingTimeMs: 0,
  };
}

export default function MeasurePage() {
  const [step, setStep] = useState<AppStep>(1);
  const shouldStartCameraRef = useRef(false);
  const wasCameraActiveRef = useRef(false);
  const [cardDetection, setCardDetection] = useState<CardDetectionResult | null>(null);
  const [footContour, setFootContour] = useState<FootContourResult | null>(null);
  const [footTracking, setFootTracking] = useState<FootTrackingStatus>(
    initialFootTrackingStatus
  );
  const [referenceTracking, setReferenceTracking] = useState<CardTrackingStatus>(
    initialCardTrackingStatus
  );
  const [readiness, setReadiness] = useState<MeasurementReadiness>(initialReadiness);
  const [latestMeasurement, setLatestMeasurement] = useState<MeasurementResult | null>(null);
  const [finalMeasurement, setFinalMeasurement] = useState<MeasurementResult | null>(null);

  const {
    videoRef,
    isActive,
    isStarting: cameraStarting,
    error: cameraError,
    startCamera,
    stopCamera,
    flipCamera,
  } = useCamera();
  const { isReady: opencvReady, error: opencvError } = useOpenCV();
  const {
    isReady: mediapipeReady,
    poseReady,
    footSegmenterReady,
    error: mediapipeError,
    detectPose,
    segmentFoot,
  } = useMediaPipe();
  const { processFrame, resetTracking } = useMeasurement();

  // Handle frame processing callback
  const handleProcessFrame = useCallback(
    (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      const result = processFrame(
        video,
        canvas,
        poseReady ? detectPose : null,
        footSegmenterReady ? segmentFoot : null,
        opencvReady,
        poseReady,
        footSegmenterReady
      );

      if (!result.processed) return;
      setCardDetection(result.cardDetection);
      setFootContour(result.footContour);
      setFootTracking(result.trackingStatus ?? initialFootTrackingStatus());
      setReferenceTracking(result.cardTracking ?? initialCardTrackingStatus());
      setReadiness(result.readiness ?? initialReadiness());
      setLatestMeasurement(result.measurement);
    },
    [
      processFrame,
      detectPose,
      segmentFoot,
      opencvReady,
      poseReady,
      footSegmenterReady,
    ]
  );

  const handleStart = useCallback(() => {
    // Switch to step 2 first so the <video> element mounts,
    // then useEffect will call startCamera once videoRef is ready.
    resetTracking();
    setFootTracking(initialFootTrackingStatus());
    setLatestMeasurement(null);
    setReferenceTracking(initialCardTrackingStatus());
    setReadiness(initialReadiness());
    shouldStartCameraRef.current = true;
    setStep(2);
  }, [resetTracking]);

  // Start camera only after Step 2 has mounted (videoRef is ready)
  useEffect(() => {
    if (shouldStartCameraRef.current && step === 2) {
      shouldStartCameraRef.current = false;
      void startCamera('environment');
    }
  }, [step, startCamera]);

  const handleCapture = useCallback((measurement: MeasurementResult) => {
    setFinalMeasurement(measurement);
    setStep(3);
  }, []);

  const handleRemeasure = useCallback(() => {
    resetTracking();
    setFinalMeasurement(null);
    setLatestMeasurement(null);
    setCardDetection(null);
    setFootContour(null);
    setFootTracking(initialFootTrackingStatus());
    setReferenceTracking(initialCardTrackingStatus());
    setReadiness(initialReadiness());
    shouldStartCameraRef.current = true;
    setStep(2);
  }, [resetTracking]);

  const handleFlipCamera = useCallback(() => {
    resetTracking();
    setLatestMeasurement(null);
    setCardDetection(null);
    setFootContour(null);
    setFootTracking(initialFootTrackingStatus());
    setReferenceTracking(initialCardTrackingStatus());
    setReadiness(initialReadiness());
    flipCamera();
  }, [flipCamera, resetTracking]);

  // Stop camera when not in step 2
  useEffect(() => {
    if (step !== 2 && isActive) {
      stopCamera();
    }
  }, [step, isActive, stopCamera]);

  // A stopped/ended stream must invalidate every previously capturable frame.
  useEffect(() => {
    if (wasCameraActiveRef.current && !isActive) {
      resetTracking();
      setCardDetection(null);
      setFootContour(null);
      setFootTracking(initialFootTrackingStatus());
      setReferenceTracking(initialCardTrackingStatus());
      setReadiness(initialReadiness());
      setLatestMeasurement(null);
    }
    wasCameraActiveRef.current = isActive;
  }, [isActive, resetTracking]);

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* Ambient gradient background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute -top-32 -left-32 w-80 h-80 bg-indigo-600/15 rounded-full blur-3xl" />
        <div className="absolute -bottom-32 -right-32 w-80 h-80 bg-violet-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-emerald-600/5 rounded-full blur-3xl" />
      </div>

      {/* Content */}
      <div className="relative flex flex-col flex-1 w-full max-w-md mx-auto px-4 py-6 gap-6">
        {/* Header */}
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
            AI-Powered Foot Measurement
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">FootMeasure</h1>
        </div>

        {/* Step indicator */}
        <StepIndicator
          currentStep={step}
          opencvReady={opencvReady}
          mediapipeReady={mediapipeReady}
        />

        {/* Camera error banner */}
        {cameraError && (
          <div className="rounded-xl bg-red-500/10 border border-red-500/20 p-3 text-red-400 text-xs text-center">
            📷 {cameraError.message}
          </div>
        )}
        {opencvError && (
          <div className="rounded-xl bg-red-500/10 border border-red-500/20 p-3 text-red-400 text-xs text-center">
            Detection engine error: {opencvError}
          </div>
        )}
        {mediapipeError && (
          <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 p-3 text-amber-300 text-xs text-center">
            {mediapipeError}
          </div>
        )}

        {/* Step content */}
        <div className="flex-1">
          {step === 1 && (
            <Step1Instructions
              onStart={handleStart}
              opencvReady={opencvReady}
              footSegmenterReady={footSegmenterReady}
              poseReady={poseReady}
            />
          )}

          {step === 2 && (
            <Step2LiveView
              videoRef={videoRef}
              cameraActive={isActive}
              cameraStarting={cameraStarting}
              cardDetection={cardDetection}
              cardTracking={referenceTracking}
              footContour={footContour}
              trackingStatus={footTracking}
              onProcessFrame={handleProcessFrame}
              onCapture={handleCapture}
              onFlipCamera={handleFlipCamera}
              latestMeasurement={latestMeasurement}
              opencvReady={opencvReady}
              footSegmenterReady={footSegmenterReady}
              poseReady={poseReady}
              readiness={readiness}
            />
          )}

          {step === 3 && finalMeasurement && (
            <Step3Results
              measurement={finalMeasurement}
              onRemeasure={handleRemeasure}
            />
          )}
        </div>

        {/* Footer */}
        <div className="text-center text-[10px] text-white/20">
          All processing is done on-device · No data leaves your browser
        </div>
      </div>
    </div>
  );
}
