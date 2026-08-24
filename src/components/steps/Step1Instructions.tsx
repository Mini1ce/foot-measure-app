'use client';

import { Camera, CreditCard, Sun, ArrowDownFromLine } from 'lucide-react';

interface Step1InstructionsProps {
  onStart: () => void;
  opencvReady: boolean;
  footSegmenterReady: boolean;
  poseReady: boolean;
}

const TIPS = [
  {
    icon: Sun,
    title: 'แสงสม่ำเสมอ',
    desc: 'เพิ่มแสงให้พอดี และหลีกเลี่ยงเงาทาบบนเท้าหรือบัตร',
    color: 'text-amber-400 bg-amber-500/10',
  },
  {
    icon: ArrowDownFromLine,
    title: 'กล้องมองจากด้านบน',
    desc: 'ถือกล้องขนานกับพื้น มองลงตรงๆ และให้เห็นเท้าครบ',
    color: 'text-indigo-400 bg-indigo-500/10',
  },
  {
    icon: CreditCard,
    title: 'วางบัตร ID-1',
    desc: 'วางบัตรอ้างอิง 85.60 × 53.98 มม. ราบบนพื้นเดียวกับเท้า',
    color: 'text-emerald-400 bg-emerald-500/10',
  },
  {
    icon: Camera,
    title: 'อยู่นิ่ง 7 เฟรม',
    desc: 'ถอดถุงเท้า วางเท้าราบ และถือกล้องนิ่งระหว่างเก็บตัวอย่าง',
    color: 'text-violet-400 bg-violet-500/10',
  },
];

export function Step1Instructions({
  onStart,
  opencvReady,
  footSegmenterReady,
  poseReady,
}: Step1InstructionsProps) {
  const isLoading = !opencvReady;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Hero */}
      <div className="text-center space-y-2">
        <div className="text-6xl mb-3">👟</div>
        <h2 className="text-2xl font-bold text-white">วัดขนาดเท้า</h2>
        <p className="text-white/50 text-sm leading-relaxed max-w-xs mx-auto">
          วัดความยาวและความกว้างด้วยกล้องและบัตรอ้างอิงขนาด ID-1 โดยประมวลผลบนเครื่องเท่านั้น
        </p>
      </div>

      {/* Tips grid */}
      <div className="grid grid-cols-2 gap-3">
        {TIPS.map(({ icon: Icon, title, desc, color }) => (
          <div
            key={title}
            className="rounded-2xl p-3.5 bg-white/5 border border-white/8 hover:border-white/15 transition-all duration-200"
          >
            <div className={`w-8 h-8 rounded-xl flex items-center justify-center mb-2.5 ${color}`}>
              <Icon className="w-4 h-4" />
            </div>
            <div className="text-xs font-semibold text-white mb-1">{title}</div>
            <div className="text-[11px] text-white/40 leading-relaxed">{desc}</div>
          </div>
        ))}
      </div>

      {/* Model loading status */}
      <div className="rounded-xl bg-white/5 border border-white/10 p-3 space-y-2">
        <div className="text-xs font-medium text-white/50 mb-2">สถานะระบบตรวจจับ</div>
        {[
          { label: 'OpenCV.js (Image Processing)', ready: opencvReady },
          { label: 'Foot AI (Custom FootNet)', ready: footSegmenterReady, optional: true },
          { label: 'MediaPipe Pose (Optional ROI)', ready: poseReady, optional: true },
        ].map(({ label, ready, optional }) => (
          <div key={label} className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full flex-shrink-0 transition-colors duration-500 ${ready ? 'bg-emerald-400' : optional ? 'bg-amber-400/70' : 'bg-white/20 animate-pulse'}`} />
            <span className="text-xs text-white/40 flex-1">{label}</span>
            <span className={`text-[10px] font-semibold ${ready ? 'text-emerald-400' : optional ? 'text-amber-400/70' : 'text-white/25'}`}>
              {ready ? 'Ready' : optional ? 'Optional' : 'Loading…'}
            </span>
          </div>
        ))}
      </div>

      {/* CTA */}
      <button
        onClick={onStart}
        disabled={isLoading}
        className={`
          w-full py-4 rounded-2xl text-sm font-bold transition-all duration-300 relative overflow-hidden
          ${isLoading
            ? 'bg-white/5 text-white/25 cursor-not-allowed'
            : 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-xl shadow-indigo-500/20 active:scale-[0.98]'
          }
        `}
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/20 border-t-white/60 rounded-full animate-spin" />
            Loading Models…
          </span>
        ) : (
          '📷 เริ่มวัดเท้า'
        )}
      </button>

      <div className="rounded-xl bg-amber-500/8 border border-amber-500/15 p-3 text-[11px] text-amber-200/70 leading-relaxed text-center">
        การเปิดกล้องบนมือถือจริงต้องใช้ HTTPS ระบบตรวจเฉพาะรูปทรงบัตรอ้างอิง ID-1 ไม่อ่านข้อความและไม่ตรวจสอบตัวตน
      </div>
    </div>
  );
}
