'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  ImageSegmenter,
  PoseLandmarker,
  PoseLandmarkerResult,
} from '@mediapipe/tasks-vision';
import type { FootSegmentationMask } from '@/types';
import {
  FOOT_SEGMENTER_MODEL_PATH,
  FOOT_SEGMENTER_WASM_PATH,
} from '@/constants';
import { copyFootMaskAndClose } from '@/lib/mediapipe/segmenterResult';

const POSE_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export function useMediaPipe() {
  const [poseReady, setPoseReady] = useState(false);
  const [footSegmenterReady, setFootSegmenterReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const footSegmenterRef = useRef<ImageSegmenter | null>(null);
  const segmentFailureCountRef = useRef(0);

  useEffect(() => {
    let mounted = true;

    async function initMediaPipe() {
      try {
        const {
          FilesetResolver,
          ImageSegmenter,
          PoseLandmarker,
        } = await import('@mediapipe/tasks-vision');
        const vision = await FilesetResolver.forVisionTasks(FOOT_SEGMENTER_WASM_PATH);

        const createFootSegmenter = async (delegate: 'GPU' | 'CPU') =>
          ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: FOOT_SEGMENTER_MODEL_PATH,
              delegate,
            },
            runningMode: 'VIDEO',
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });

        try {
          const segmenter = await createFootSegmenter('GPU');
          if (mounted) {
            footSegmenterRef.current = segmenter;
            setFootSegmenterReady(true);
          } else {
            segmenter.close();
          }
        } catch {
          try {
            const segmenter = await createFootSegmenter('CPU');
            if (mounted) {
              footSegmenterRef.current = segmenter;
              setFootSegmenterReady(true);
            } else {
              segmenter.close();
            }
          } catch (segmenterError: unknown) {
            if (mounted) {
              const message = segmenterError instanceof Error
                ? segmenterError.message
                : 'Unknown foot model error';
              setError(`Foot model unavailable; using visual fallback: ${message}`);
            }
          }
        }

        const createPose = async (delegate: 'GPU' | 'CPU') =>
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL_PATH, delegate },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });

        try {
          const pose = await createPose('GPU');
          if (mounted) {
            poseRef.current = pose;
            setPoseReady(true);
          } else {
            pose.close();
          }
        } catch {
          try {
            const pose = await createPose('CPU');
            if (mounted) {
              poseRef.current = pose;
              setPoseReady(true);
            } else {
              pose.close();
            }
          } catch {
            // Pose is optional because the guide-aligned ROI works without it.
          }
        }
      } catch (initializationError: unknown) {
        if (mounted) {
          const message = initializationError instanceof Error
            ? initializationError.message
            : 'Unknown MediaPipe error';
          setError(`MediaPipe unavailable; using visual fallback: ${message}`);
        }
      }
    }

    void initMediaPipe();

    return () => {
      mounted = false;
      poseRef.current?.close();
      footSegmenterRef.current?.close();
      poseRef.current = null;
      footSegmenterRef.current = null;
    };
  }, []);

  const detectPose = useCallback(
    (video: HTMLVideoElement, timestamp: number): PoseLandmarkerResult | null => {
      if (!poseRef.current || !poseReady) return null;
      try {
        return poseRef.current.detectForVideo(video, timestamp);
      } catch {
        return null;
      }
    },
    [poseReady]
  );

  const segmentFoot = useCallback(
    (image: HTMLCanvasElement, timestamp: number): FootSegmentationMask | null => {
      if (!footSegmenterRef.current || !footSegmenterReady) return null;
      try {
        const result = footSegmenterRef.current.segmentForVideo(image, timestamp);
        const mask = copyFootMaskAndClose(result);
        segmentFailureCountRef.current = 0;
        return mask;
      } catch {
        segmentFailureCountRef.current++;
        if (segmentFailureCountRef.current >= 3) {
          footSegmenterRef.current?.close();
          footSegmenterRef.current = null;
          setFootSegmenterReady(false);
          setError('โมเดลตรวจเท้าหยุดตอบสนอง จึงเปลี่ยนไปใช้ OpenCV สำรอง');
        }
        return null;
      }
    },
    [footSegmenterReady]
  );

  return {
    isReady: footSegmenterReady || poseReady,
    poseReady,
    footSegmenterReady,
    error,
    detectPose,
    segmentFoot,
  };
}
