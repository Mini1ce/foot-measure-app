'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { CameraErrorInfo } from '@/types';
import {
  CameraSetupError,
  classifyCameraError,
  stopMediaStream,
  waitForLoadedMetadata,
} from '@/lib/camera/mediaStream';

type FacingMode = 'environment' | 'user';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const requestTokenRef = useRef(0);
  const mountedRef = useRef(true);
  const [isActive, setIsActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<CameraErrorInfo | null>(null);
  const [facingMode, setFacingMode] = useState<FacingMode>('environment');

  const releaseCurrentStream = useCallback((updateState: boolean) => {
    const stream = streamRef.current;
    streamRef.current = null;
    stopMediaStream(stream);
    if (videoRef.current) videoRef.current.srcObject = null;
    if (updateState && mountedRef.current) {
      setIsActive(false);
      setIsStarting(false);
    }
  }, []);

  const invalidatePendingRequest = useCallback(() => {
    requestTokenRef.current++;
  }, []);

  const startCamera = useCallback(async (mode: FacingMode = facingMode) => {
    const requestToken = ++requestTokenRef.current;
    releaseCurrentStream(false);
    setIsActive(false);
    setIsStarting(true);
    setError(null);

    let acquiredStream: MediaStream | null = null;
    try {
      if (!window.isSecureContext) {
        throw new CameraSetupError(
          'insecure-context',
          'มือถือจริงต้องเปิดหน้านี้ผ่าน HTTPS จึงจะใช้กล้องได้'
        );
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new CameraSetupError(
          'unsupported',
          'เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง กรุณาใช้ Safari หรือ Chrome รุ่นปัจจุบัน'
        );
      }

      const attempts: MediaStreamConstraints[] = [
        {
          video: {
            facingMode: { exact: mode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        },
        {
          video: {
            facingMode: { ideal: mode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        },
        { video: true, audio: false },
      ];

      let lastError: unknown = null;
      for (const constraints of attempts) {
        try {
          acquiredStream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (attemptError) {
          lastError = attemptError;
          const errorName = attemptError && typeof attemptError === 'object' && 'name' in attemptError
            ? String(attemptError.name)
            : '';
          if (errorName === 'NotAllowedError' || errorName === 'SecurityError') break;
        }
      }
      if (!acquiredStream) throw lastError ?? new Error('Camera unavailable');

      if (!mountedRef.current || requestToken !== requestTokenRef.current) {
        stopMediaStream(acquiredStream);
        return;
      }
      const video = videoRef.current;
      if (!video) {
        throw new CameraSetupError('playback-failed', 'ไม่พบพื้นที่แสดงภาพจากกล้อง');
      }

      streamRef.current = acquiredStream;
      video.srcObject = acquiredStream;
      video.setAttribute('playsinline', 'true');
      await waitForLoadedMetadata(video);
      if (!mountedRef.current || requestToken !== requestTokenRef.current) {
        stopMediaStream(acquiredStream);
        if (streamRef.current === acquiredStream) streamRef.current = null;
        if (video.srcObject === acquiredStream) video.srcObject = null;
        return;
      }
      await video.play();
      if (video.videoWidth <= 0 || video.videoHeight <= 0) {
        throw new CameraSetupError('metadata-timeout', 'กล้องยังไม่ส่งขนาดภาพที่ใช้งานได้');
      }

      const videoTrack = acquiredStream.getVideoTracks()[0];
      const actualFacing = videoTrack?.getSettings().facingMode;
      const resolvedFacing: FacingMode =
        actualFacing === 'user' || actualFacing === 'environment'
          ? actualFacing
          : mode;
      videoTrack?.addEventListener('ended', () => {
        if (streamRef.current !== acquiredStream || !mountedRef.current) return;
        releaseCurrentStream(true);
        setError({ code: 'in-use', message: 'กล้องหยุดทำงาน กรุณาเปิดกล้องใหม่อีกครั้ง' });
      }, { once: true });

      if (mountedRef.current && requestToken === requestTokenRef.current) {
        setFacingMode(resolvedFacing);
        setIsActive(true);
      }
    } catch (cameraError) {
      if (acquiredStream) stopMediaStream(acquiredStream);
      if (streamRef.current === acquiredStream) streamRef.current = null;
      if (videoRef.current?.srcObject === acquiredStream) videoRef.current.srcObject = null;
      if (mountedRef.current && requestToken === requestTokenRef.current) {
        setError(classifyCameraError(cameraError, window.isSecureContext));
        setIsActive(false);
      }
    } finally {
      if (mountedRef.current && requestToken === requestTokenRef.current) setIsStarting(false);
    }
  }, [facingMode, releaseCurrentStream]);

  const stopCamera = useCallback(() => {
    invalidatePendingRequest();
    releaseCurrentStream(true);
  }, [invalidatePendingRequest, releaseCurrentStream]);

  const flipCamera = useCallback(() => {
    const newMode: FacingMode = facingMode === 'environment' ? 'user' : 'environment';
    void startCamera(newMode);
  }, [facingMode, startCamera]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      invalidatePendingRequest();
      releaseCurrentStream(false);
    };
  }, [invalidatePendingRequest, releaseCurrentStream]);

  return {
    videoRef,
    isActive,
    isStarting,
    error,
    facingMode,
    startCamera,
    stopCamera,
    flipCamera,
  };
}
