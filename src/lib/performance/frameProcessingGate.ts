import {
  PROCESS_MAX_INTERVAL_MS,
  PROCESS_MIN_INTERVAL_MS,
  PROCESS_TIME_SAFETY_FACTOR,
} from '@/constants';

/** Synchronous lock plus adaptive cadence for main-thread vision work. */
export class FrameProcessingGate {
  private busy = false;
  private lastStartedAt = Number.NEGATIVE_INFINITY;
  private durationEwma = 0;
  private intervalMs = PROCESS_MIN_INTERVAL_MS;

  reset(): void {
    this.busy = false;
    this.lastStartedAt = Number.NEGATIVE_INFINITY;
    this.durationEwma = 0;
    this.intervalMs = PROCESS_MIN_INTERVAL_MS;
  }

  tryStart(now: number): boolean {
    if (!Number.isFinite(now) || this.busy || now - this.lastStartedAt < this.intervalMs) {
      return false;
    }
    this.busy = true;
    this.lastStartedAt = now;
    return true;
  }

  finish(durationMs: number): void {
    if (!this.busy) return;
    const duration = Number.isFinite(durationMs) ? Math.max(0, durationMs) : 0;
    this.durationEwma = this.durationEwma === 0
      ? duration
      : this.durationEwma * 0.75 + duration * 0.25;
    this.intervalMs = Math.max(
      PROCESS_MIN_INTERVAL_MS,
      Math.min(PROCESS_MAX_INTERVAL_MS, this.durationEwma * PROCESS_TIME_SAFETY_FACTOR)
    );
    this.busy = false;
  }

  getRecommendedIntervalMs(): number {
    return this.intervalMs;
  }

  isBusy(): boolean {
    return this.busy;
  }
}
