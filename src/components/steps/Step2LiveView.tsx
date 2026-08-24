'use client';

import { useCallback } from 'react';
import {
  CardDetectionResult,
  CardTrackingStatus,
  FootContourResult,
  FootTrackingStatus,
  MeasurementResult,
  MeasurementReadiness,
} from '@/types';
import { CameraView } from '@/components/camera/CameraView';
import { ConfidenceBar } from '@/components/ui/ConfidenceBar';
import { canCaptureMeasurement } from '@/lib/calculations/measurementPolicy';

interface Step2LiveViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraActive: boolean;
  cameraStarting: boolean;
  cardDetection: CardDetectionResult | null;
  cardTracking: CardTrackingStatus;
  footContour: FootContourResult | null;
  trackingStatus: FootTrackingStatus;
  onProcessFrame: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
  onCapture: (measurement: MeasurementResult) => void;
  onFlipCamera?: () => void;
  latestMeasurement: MeasurementResult | null;
  opencvReady: boolean;
  footSegmenterReady: boolean;
  poseReady: boolean;
  readiness: MeasurementReadiness;
}

export function Step2LiveView({
  videoRef,
  cameraActive,
  cameraStarting,
  cardDetection,
  cardTracking,
  footContour,
  trackingStatus,
  onProcessFrame,
  onCapture,
  onFlipCamera,
  latestMeasurement,
  opencvReady,
  footSegmenterReady,
  poseReady,
  readiness,
}: Step2LiveViewProps) {
  const canCapture = canCaptureMeasurement({
    cameraReady: cameraActive && readiness.cameraReady,
    cardStable: readiness.cardStable,
    footDetected: readiness.footDetected,
    homographyValid: readiness.homographyValid,
    trackingReady: readiness.trackingReady,
    measurement: latestMeasurement,
  });

  const handleCapture = useCallback(() => {
    if (canCapture && latestMeasurement) {
      onCapture(latestMeasurement);
    }
  }, [canCapture, latestMeasurement, onCapture]);

  const trackingLabel =
    trackingStatus.phase === 'stable'
      ? 'นิ่งแล้ว — พร้อม'
      : trackingStatus.phase === 'stabilizing'
        ? 'อยู่นิ่ง — กำลังตรวจความสม่ำเสมอ'
        : trackingStatus.sampleCount > 0
          ? `เก็บตัวอย่าง ${trackingStatus.sampleCount}/${trackingStatus.targetSamples}`
          : footContour?.detected && !readiness.cardStable
            ? 'รอยืนยันบัตรอ้างอิง'
            : 'กำลังค้นหา…';

  const guidanceMessages: Record<string, string> = {
    'low-light': 'แสงน้อยเกินไป เพิ่มแสงให้สม่ำเสมอและหลีกเลี่ยงเงาบนเท้า',
    'card-not-found': 'วางบัตรอ้างอิงขนาด ID-1 ให้เห็นครบทั้ง 4 มุม',
    'card-too-small': 'บัตรอยู่ไกลเกินไป ลดระยะกล้องลงโดยให้เห็นเท้าครบ',
    'card-too-large': 'บัตรอยู่ใกล้เกินไป ยกกล้องขึ้นจนเห็นเท้าและบัตรครบ',
    'camera-tilted': 'กล้องเอียงมากไป ถือกล้องให้ขนานกับพื้นและมองลงตรงๆ',
    'card-occluded': 'ขอบบัตรไม่ครบ กรุณาเอานิ้วหรือเท้าออกจากบัตร',
    'card-wrong-shape': 'วัตถุที่พบมีสัดส่วนไม่ตรงบัตร ID-1 กรุณาใช้บัตรขนาด 85.60 × 53.98 มม.',
    'card-outside-guide': 'ย้ายบัตรเข้าไปในกรอบสีเขียวข้างเท้า',
    'homography-failed': 'คำนวณระนาบบัตรไม่ได้ วางบัตรให้ราบ เห็นครบ 4 มุม แล้วถือกล้องตรงขึ้น',
    'card-unstable': 'ถือกล้องและบัตรให้นิ่งเพื่อยืนยันบัตรหลายเฟรม',
    'foot-not-found': 'วางเท้าเปล่าให้ครบทั้งส้นและปลายนิ้วภายในกรอบ',
    'foot-outside-guide': 'ขยับเท้าให้ไม่หลุดกรอบ',
    'hold-still': 'อยู่นิ่งๆ ระหว่างเก็บตัวอย่างที่ผ่านเกณฑ์ 7 เฟรม',
    'ppm-fallback': 'กำลังใช้สเกลสำรอง จึงยังบันทึกผลไม่ได้ กรุณาปรับมุมกล้อง',
  };
  const primaryGuidance = readiness.guidance[0];

  return (
    <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500">

      {process.env.NODE_ENV === 'development' && (
        <div className="rounded-lg bg-black/60 border border-white/10 px-3 py-2 font-mono text-[10px] text-white/60 flex flex-wrap gap-x-4 gap-y-1">
          <span>OpenCV: {opencvReady ? 'ready' : 'loading'}</span>
          <span>Foot AI: {footSegmenterReady ? 'ready' : 'fallback'}</span>
          <span>Pose: {poseReady ? 'ready' : 'optional'}</span>
          <span>Card: {cardDetection ? `${(cardDetection.confidence * 100).toFixed(0)}%` : 'none'}</span>
          <span>H: {readiness.homographyValid ? 'valid' : 'invalid'}</span>
          <span>Frame: {readiness.processingTimeMs.toFixed(0)}ms</span>
        </div>
      )}
      {/* Camera feed */}
      <CameraView
        videoRef={videoRef}
        isActive={cameraActive}
        isStarting={cameraStarting}
        showGuide={true}
        cardDetection={cardDetection}
        cardTracking={cardTracking}
        footContour={footContour}
        trackingStatus={trackingStatus}
        readiness={readiness}
        onProcessFrame={onProcessFrame}
        onFlipCamera={onFlipCamera}
      />

      <div className="grid grid-cols-5 gap-1.5" aria-label="สถานะการวัด">
        {[
          ['กล้อง', cameraActive],
          ['บัตร', readiness.cardStable],
          ['เท้า', readiness.footDetected],
          [`${trackingStatus.sampleCount}/${trackingStatus.targetSamples}`, readiness.trackingReady],
          ['บันทึก', canCapture],
        ].map(([label, ready]) => (
          <div key={String(label)} className="rounded-lg bg-white/5 border border-white/10 px-1 py-2 text-center">
            <div className={`mx-auto mb-1 h-2 w-2 rounded-full ${ready ? 'bg-emerald-400' : 'bg-white/15'}`} />
            <div className="text-[9px] text-white/50">{label}</div>
          </div>
        ))}
      </div>

      {primaryGuidance && (
        <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 px-3 py-2.5 text-xs text-amber-200 leading-relaxed">
          {guidanceMessages[primaryGuidance]}
        </div>
      )}

      {/* Detection panel */}
      <div className="grid grid-cols-2 gap-3">
        {/* Card confidence */}
        <div className="rounded-xl bg-white/5 border border-white/10 p-3">
          <div className="text-[10px] font-semibold text-white/40 mb-2 uppercase tracking-wider">
            Card Reference
          </div>
          <ConfidenceBar
            confidence={cardDetection?.confidence ?? 0}
            label="ID/Credit Card"
          />
          {cardDetection?.detected && (
            <div className="mt-2 text-[10px] text-white/30 font-mono">
              {cardDetection.measurementReady ? 'Homography พร้อม' : 'ยังใช้วัดไม่ได้'}
            </div>
          )}
        </div>

        {/* Foot detection */}
        <div className="rounded-xl bg-white/5 border border-white/10 p-3">
          <div className="text-[10px] font-semibold text-white/40 mb-2 uppercase tracking-wider">
            Foot Detection
          </div>
          <div className={`flex items-center gap-2 py-1 ${trackingStatus.isStable ? 'text-emerald-400' : footContour?.detected ? 'text-indigo-300' : 'text-white/25'}`}>
            <div className={`w-2 h-2 rounded-full ${trackingStatus.isStable ? 'bg-emerald-400 animate-pulse' : footContour?.detected ? 'bg-indigo-400 animate-pulse' : 'bg-white/15'}`} />
            <span className="text-xs font-medium">
              {trackingLabel}
            </span>
          </div>
          <div className="mt-2 flex gap-1" aria-label={`Foot tracking ${trackingStatus.sampleCount} of ${trackingStatus.targetSamples}`}>
            {Array.from({ length: trackingStatus.targetSamples }, (_, index) => (
              <span
                key={index}
                className={`h-1.5 flex-1 rounded-full transition-colors ${
                  index < trackingStatus.sampleCount
                    ? trackingStatus.isStable
                      ? 'bg-emerald-400'
                      : 'bg-indigo-400'
                    : 'bg-white/10'
                }`}
              />
            ))}
          </div>
          {process.env.NODE_ENV === 'development' && footContour?.detected && (
            <div className="mt-1 text-[10px] text-white/30 font-mono">
              {footContour.lengthPx.toFixed(0)}px × {footContour.widthPx.toFixed(0)}px
            </div>
          )}
        </div>
      </div>

      {/* Live measurement preview */}
      {latestMeasurement && (
        <div className="rounded-xl bg-gradient-to-r from-indigo-600/20 to-violet-600/20 border border-indigo-500/20 p-3 flex items-center justify-between">
          <div className="space-y-0.5">
            <div className="text-xs text-indigo-300 font-medium">
              ค่าประมาณ • คุณภาพ {latestMeasurement.quality}
            </div>
            <div className="text-white font-mono text-sm">
              {latestMeasurement.footLengthCm} cm × {latestMeasurement.footWidthCm} cm
            </div>
          </div>
          <button
            onClick={handleCapture}
            disabled={!canCapture}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              canCapture
                ? 'bg-indigo-600 hover:bg-indigo-500 text-white active:scale-95 shadow-lg shadow-indigo-500/20'
                : 'bg-white/10 text-white/30 cursor-not-allowed'
            }`}
          >
            บันทึกผล
          </button>
        </div>
      )}

      {/* Tips */}
      <div className="text-center text-xs text-white/30 leading-relaxed">
        บัตรอ้างอิง ID-1 และเท้าต้องวางราบบนพื้นเดียวกัน ระบบไม่อ่านข้อมูลบนบัตรและไม่ใช้ยืนยันตัวตน
      </div>
    </div>
  );
}
