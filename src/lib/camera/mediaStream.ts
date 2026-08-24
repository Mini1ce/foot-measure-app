import type { CameraErrorCode, CameraErrorInfo } from '@/types';

export interface StoppableMediaStream {
  getTracks(): Array<{ stop(): void }>;
}

export function stopMediaStream(stream: StoppableMediaStream | null): void {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      // Continue so one faulty browser track cannot prevent the others closing.
    }
  }
}

export class CameraSetupError extends Error {
  constructor(public readonly code: CameraErrorCode, message: string) {
    super(message);
    this.name = 'CameraSetupError';
  }
}

export function classifyCameraError(error: unknown, secureContext = true): CameraErrorInfo {
  if (!secureContext) {
    return {
      code: 'insecure-context',
      message: 'มือถือจริงต้องเปิดหน้านี้ผ่าน HTTPS จึงจะใช้กล้องได้',
    };
  }
  if (error instanceof CameraSetupError) {
    return { code: error.code, message: error.message };
  }

  const name = error && typeof error === 'object' && 'name' in error
    ? String(error.name)
    : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return {
        code: 'permission-denied',
        message: 'ไม่ได้รับสิทธิ์ใช้กล้อง กรุณาอนุญาตกล้องในการตั้งค่าเบราว์เซอร์แล้วลองใหม่',
      };
    case 'NotFoundError':
    case 'OverconstrainedError':
      return {
        code: 'not-found',
        message: 'ไม่พบกล้องที่ใช้งานได้ กรุณาตรวจว่ามือถือหรืออุปกรณ์มีกล้อง',
      };
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return {
        code: 'in-use',
        message: 'เปิดกล้องไม่ได้ กล้องอาจถูกแอปอื่นใช้งานอยู่ กรุณาปิดแอปนั้นแล้วลองใหม่',
      };
    case 'TypeError':
      return {
        code: 'unsupported',
        message: 'เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง หรือหน้านี้ไม่ได้เปิดผ่าน HTTPS',
      };
    default:
      return {
        code: 'unknown',
        message: 'เปิดกล้องไม่สำเร็จ กรุณารีโหลดหน้าแล้วลองอีกครั้ง',
      };
  }
}

export function waitForLoadedMetadata(
  video: HTMLVideoElement,
  timeoutMs = 8000
): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0 && video.videoHeight > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener('loadedmetadata', handleMetadata);
      video.removeEventListener('error', handleError);
    };
    const handleMetadata = () => {
      if (video.videoWidth <= 0 || video.videoHeight <= 0) return;
      cleanup();
      resolve();
    };
    const handleError = () => {
      cleanup();
      reject(new CameraSetupError('playback-failed', 'เบราว์เซอร์อ่านภาพจากกล้องไม่ได้'));
    };
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new CameraSetupError('metadata-timeout', 'กล้องตอบสนองช้าเกินไป กรุณาลองเปิดกล้องใหม่'));
    }, timeoutMs);
    video.addEventListener('loadedmetadata', handleMetadata);
    video.addEventListener('error', handleError);
  });
}
