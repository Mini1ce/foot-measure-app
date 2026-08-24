'use client';

import type {
  CardDetectionResult,
  CardTrackingStatus,
  FootContourResult,
  FootTrackingStatus,
  MeasurementReadiness,
} from '@/types';
import { Scan, Eye, AlertCircle, CheckCircle2 } from 'lucide-react';

interface StatusBadgeProps {
  cardDetection: CardDetectionResult | null;
  cardTracking: CardTrackingStatus;
  footContour: FootContourResult | null;
  trackingStatus: FootTrackingStatus;
  readiness: MeasurementReadiness;
}

export function StatusBadge({
  cardDetection,
  cardTracking,
  footContour,
  trackingStatus,
  readiness,
}: StatusBadgeProps) {
  const getStatus = () => {
    if (!readiness.cameraReady) {
      return { text: 'กำลังเตรียมกล้อง', icon: Scan, color: 'text-white/60 bg-white/5 border-white/10' };
    }
    if (!cardDetection?.detected) {
      return { text: 'วางบัตรอ้างอิง ID-1 ในกรอบ', icon: Scan, color: 'text-white/60 bg-white/5 border-white/10' };
    }
    if (!cardDetection.measurementReady) {
      return { text: 'ปรับบัตรหรือมุมกล้อง', icon: AlertCircle, color: 'text-amber-300 bg-amber-500/10 border-amber-500/20' };
    }
    if (!cardTracking.isStable) {
      return {
        text: `กำลังยืนยันบัตร ${cardTracking.sampleCount}/${cardTracking.targetSamples}`,
        icon: Eye,
        color: 'text-yellow-300 bg-yellow-500/10 border-yellow-500/20',
      };
    }
    if (!footContour?.detected) {
      return { text: 'พบบัตรแล้ว • จัดเท้าในกรอบ', icon: Eye, color: 'text-indigo-300 bg-indigo-500/10 border-indigo-500/20' };
    }
    if (!trackingStatus.isStable) {
      return {
        text: `เก็บตัวอย่าง ${trackingStatus.sampleCount}/${trackingStatus.targetSamples}`,
        icon: Eye,
        color: 'text-indigo-300 bg-indigo-500/10 border-indigo-500/20',
      };
    }
    if (!readiness.homographyValid) {
      return { text: 'ต้องปรับมุมกล้องก่อนบันทึก', icon: AlertCircle, color: 'text-amber-300 bg-amber-500/10 border-amber-500/20' };
    }
    return { text: 'พร้อมบันทึกผล', icon: CheckCircle2, color: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20' };
  };

  const { text, icon: Icon, color } = getStatus();
  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold backdrop-blur-sm ${color} transition-all duration-300`}>
      <Icon className="w-3.5 h-3.5" />
      {text}
      {readiness.canCapture && <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />}
    </div>
  );
}
