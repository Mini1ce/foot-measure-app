import type { MeasurementResult } from '@/types';

export interface CaptureGateInput {
  cameraReady: boolean;
  cardStable: boolean;
  footDetected: boolean;
  homographyValid: boolean;
  trackingReady: boolean;
  measurement: MeasurementResult | null;
}

/** Single save policy shared by the processing pipeline and both UI buttons. */
export function canCaptureMeasurement(input: CaptureGateInput): boolean {
  return (
    input.cameraReady &&
    input.cardStable &&
    input.footDetected &&
    input.homographyValid &&
    input.trackingReady &&
    input.measurement !== null &&
    input.measurement.calibrationMode === 'homography' &&
    input.measurement.homographyValid &&
    input.measurement.quality !== 'low' &&
    input.measurement.confidence >= 0.62
  );
}
