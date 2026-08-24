'use client';

import { MeasurementResult } from '@/types';
import { MeasurementCard } from '@/components/ui/MeasurementCard';
import { Share2, Download } from 'lucide-react';

interface Step3ResultsProps {
  measurement: MeasurementResult;
  onRemeasure: () => void;
}

export function Step3Results({ measurement, onRemeasure }: Step3ResultsProps) {
  const handleShare = async () => {
    const shoeText = measurement.shoeSize
      ? `\nแนวทางไซซ์รองเท้า: EU ${measurement.shoeSize.eu} | US Men ${measurement.shoeSize.usMen} | US Women ${measurement.shoeSize.usWomen} | UK ${measurement.shoeSize.uk}`
      : '';
    const text = `ขนาดเท้า: ${measurement.footLengthCm} ซม. × ${measurement.footWidthCm} ซม.${shoeText}\nวัดด้วย FootMeasure 📏`;

    if (navigator.share) {
      try {
        await navigator.share({ title: 'My Foot Measurement', text });
      } catch {}
    } else {
      await navigator.clipboard.writeText(text);
      alert('Copied to clipboard!');
    }
  };

  return (
    <div className="flex flex-col gap-5 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header */}
      <div className="text-center">
        <div className="text-4xl mb-2">✅</div>
        <h2 className="text-xl font-bold text-white">บันทึกผลการวัดแล้ว</h2>
        <p className="text-white/40 text-sm mt-1">ผลนี้ผ่าน homography และตัวอย่างที่นิ่ง 7 เฟรม</p>
      </div>

      {/* Main result card */}
      <MeasurementCard result={measurement} onRemeasure={onRemeasure} />

      {/* Action row */}
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={handleShare}
          className="flex items-center justify-center gap-2 py-3 rounded-2xl bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-all text-sm font-medium"
        >
          <Share2 className="w-4 h-4" />
          แชร์
        </button>
        <button
          onClick={() => {
            const data = JSON.stringify(measurement, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `foot-measurement-${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="flex items-center justify-center gap-2 py-3 rounded-2xl bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-all text-sm font-medium"
        >
          <Download className="w-4 h-4" />
          ส่งออก JSON
        </button>
      </div>

      {/* Accuracy note */}
      <div className="rounded-xl bg-amber-500/8 border border-amber-500/15 p-3 text-center">
        <p className="text-amber-300/70 text-[11px] leading-relaxed">
          ⚠️ ผลนี้ยังไม่ใช่การวัดทางการแพทย์ และยังต้องตรวจสอบความคลาดเคลื่อนกับโทรศัพท์ บัตร และเท้าจริงหลายขนาดก่อนใช้งานเชิงพาณิชย์
        </p>
      </div>
    </div>
  );
}
