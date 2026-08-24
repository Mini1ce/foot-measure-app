'use client';

import { useEffect, useRef } from 'react';
import {
  CardDetectionResult,
  CardTrackingStatus,
  FootContourResult,
  FootTrackingStatus,
  MeasurementReadiness,
} from '@/types';
import { GuideOverlay } from './GuideOverlay';
import { StatusBadge } from './StatusBadge';
import { FlipHorizontal, ZapOff } from 'lucide-react';

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isActive: boolean;
  isStarting: boolean;
  showGuide: boolean;
  cardDetection: CardDetectionResult | null;
  cardTracking: CardTrackingStatus;
  footContour: FootContourResult | null;
  trackingStatus: FootTrackingStatus;
  readiness: MeasurementReadiness;
  onProcessFrame: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
  onFlipCamera?: () => void;
}

export function CameraView({
  videoRef,
  isActive,
  isStarting,
  showGuide,
  cardDetection,
  cardTracking,
  footContour,
  trackingStatus,
  readiness,
  onProcessFrame,
  onFlipCamera,
}: CameraViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const videoFrameRef = useRef<number>(0);
  const lastMediaTimeRef = useRef(-1);

  useEffect(() => {
    if (!isActive) return;

    let cancelled = false;
    lastMediaTimeRef.current = -1;
    const videoWithFrames = videoRef.current as (HTMLVideoElement & {
      requestVideoFrameCallback?: (
        callback: (now: DOMHighResTimeStamp, metadata: { mediaTime: number }) => void
      ) => number;
      cancelVideoFrameCallback?: (handle: number) => void;
    }) | null;

    function processCurrentFrame(mediaTime?: number) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        const { videoWidth: vw, videoHeight: vh } = video;
        const isNewFrame = mediaTime === undefined || mediaTime !== lastMediaTimeRef.current;
        if (vw > 0 && vh > 0 && isNewFrame) {
          if (mediaTime !== undefined) lastMediaTimeRef.current = mediaTime;
          onProcessFrame(video, canvas);
        }
      }
    }

    function scheduleVideoFrame() {
      if (cancelled || !videoWithFrames?.requestVideoFrameCallback) return;
      videoFrameRef.current = videoWithFrames.requestVideoFrameCallback((_now, metadata) => {
        processCurrentFrame(metadata.mediaTime);
        scheduleVideoFrame();
      });
    }

    function runAnimationFrame() {
      if (cancelled) return;
      processCurrentFrame();
      rafRef.current = requestAnimationFrame(runAnimationFrame);
    }

    if (videoWithFrames?.requestVideoFrameCallback) scheduleVideoFrame();
    else rafRef.current = requestAnimationFrame(runAnimationFrame);
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (videoFrameRef.current) videoWithFrames?.cancelVideoFrameCallback?.(videoFrameRef.current);
    };
  }, [isActive, videoRef, onProcessFrame]);

  return (
    <div className="relative w-full rounded-2xl overflow-hidden bg-black border border-white/10 shadow-2xl shadow-black/50 aspect-[3/4] sm:aspect-[4/3]">
      {/*
        The <video> is now VISIBLE as the background — this ensures we always
        see the camera feed even before OpenCV processes any frame.
        The canvas sits on top with a transparent background so only the
        CV overlays (card outline, foot contour lines) are visible.
      */}
      <video
        ref={videoRef}
        className="absolute inset-0 w-full h-full object-cover"
        playsInline
        muted
        autoPlay
      />

      {/* Transparent canvas for CV overlays only */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        style={{ opacity: isActive ? 1 : 0 }}
      />

      {/* Alignment guide overlay */}
      {showGuide && !readiness.canCapture && <GuideOverlay />}

      {/* Top controls bar */}
      <div className="absolute top-3 left-3 right-3 flex items-start justify-between pointer-events-none">
        <StatusBadge
          cardDetection={cardDetection}
          cardTracking={cardTracking}
          footContour={footContour}
          trackingStatus={trackingStatus}
          readiness={readiness}
        />
        {onFlipCamera && (
          <button
            onClick={onFlipCamera}
            className="pointer-events-auto w-8 h-8 rounded-full bg-black/40 backdrop-blur-sm border border-white/10 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/10 transition-all"
            aria-label="Flip camera"
          >
            <FlipHorizontal className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Camera inactive overlay */}
      {!isActive && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950/95 gap-3">
          <ZapOff className="w-10 h-10 text-white/20" />
          <p className="text-white/50 text-sm font-medium">
            {isStarting ? 'กำลังเปิดกล้อง…' : 'กล้องยังไม่พร้อม'}
          </p>
          <p className="text-white/25 text-xs text-center px-6">
            กรุณาอนุญาตสิทธิ์กล้อง และใช้ HTTPS เมื่อเปิดบนมือถือจริง
          </p>
        </div>
      )}
    </div>
  );
}
