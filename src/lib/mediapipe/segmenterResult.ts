import type { FootSegmentationMask } from '@/types';

export interface ConfidenceMaskLike {
  width: number;
  height: number;
  getAsFloat32Array(): Float32Array | ArrayLike<number>;
}

export interface SegmenterResultLike {
  confidenceMasks?: ConfidenceMaskLike[];
  close(): void;
}

/** Copy browser-owned WASM memory before always releasing the task result. */
export function copyFootMaskAndClose(result: SegmenterResultLike): FootSegmentationMask | null {
  try {
    const footMask = result.confidenceMasks?.[1];
    if (!footMask) return null;
    return {
      data: new Float32Array(footMask.getAsFloat32Array()),
      width: footMask.width,
      height: footMask.height,
    };
  } finally {
    result.close();
  }
}
