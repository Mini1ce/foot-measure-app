'use client';
/* eslint-disable @typescript-eslint/no-explicit-any -- OpenCV.js exposes an untyped global runtime. */

import { useEffect, useState } from 'react';

declare global {
  interface Window {
    cv: any;
  }
}

const OPENCV_SCRIPT_ID = 'opencv-js-runtime';
const OPENCV_URL = '/opencv.js';

/** Load the self-hosted OpenCV runtime without depending on a third-party CDN. */
export function useOpenCV() {
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const markReady = (): boolean => {
      if (!window.cv?.Mat) return false;
      if (!cancelled) {
        setError(null);
        setIsReady(true);
      }
      return true;
    };

    if (markReady()) return;

    let script = document.getElementById(OPENCV_SCRIPT_ID) as HTMLScriptElement | null;
    if (!script) {
      script = document.createElement('script');
      script.id = OPENCV_SCRIPT_ID;
      script.src = OPENCV_URL;
      script.async = true;
      document.head.appendChild(script);
    }

    const handleLoad = () => {
      markReady();
    };
    const handleError = () => {
      if (!cancelled) setError('Failed to load the local OpenCV.js runtime.');
    };
    script.addEventListener('load', handleLoad);
    script.addEventListener('error', handleError);

    let attempts = 0;
    pollTimer = setInterval(() => {
      attempts++;
      if (markReady() && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      } else if (attempts >= 150 && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
        if (!cancelled) setError('OpenCV.js initialization timed out.');
      }
    }, 100);

    return () => {
      cancelled = true;
      script?.removeEventListener('load', handleLoad);
      script?.removeEventListener('error', handleError);
      if (pollTimer) clearInterval(pollTimer);
    };
  }, []);

  return { isReady, error };
}
