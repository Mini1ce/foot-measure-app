'use client';

import {
  CARD_ASPECT_RATIO,
  CARD_GUIDE_LAYOUT,
  FOOT_GUIDE_RECT,
} from '@/constants';

function percentage(value: number): string {
  return `${value * 100}%`;
}

/** Display-space placement guides aligned with the object-cover ROI mapping. */
export function GuideOverlay() {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <div
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      <div
        className="absolute rounded-3xl border-2 border-dashed border-indigo-400/60 bg-indigo-500/5 shadow-[0_0_24px_rgba(99,102,241,0.22)]"
        style={{
          left: percentage(FOOT_GUIDE_RECT.x),
          top: percentage(FOOT_GUIDE_RECT.y),
          width: percentage(FOOT_GUIDE_RECT.width),
          height: percentage(FOOT_GUIDE_RECT.height),
        }}
      >
        <div className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-px whitespace-nowrap rounded-b-xl bg-indigo-500/90 px-4 py-1 text-[10px] font-bold tracking-wide text-white">
          วางเท้าในกรอบ
        </div>
        <div className="absolute inset-0 flex items-center justify-center text-6xl text-white/10">
          👣
        </div>
      </div>

      <div
        className="absolute"
        style={{
          left: percentage(CARD_GUIDE_LAYOUT.x),
          bottom: percentage(CARD_GUIDE_LAYOUT.bottom),
          width: percentage(CARD_GUIDE_LAYOUT.width),
          aspectRatio: '1',
        }}
      >
        <div
          className="absolute bottom-0 left-0 w-full rounded-xl border-2 border-dashed border-emerald-400/70 bg-emerald-500/8 shadow-[0_0_18px_rgba(16,185,129,0.2)]"
          style={{ aspectRatio: String(CARD_ASPECT_RATIO) }}
        />
        <div
          className="absolute bottom-0 right-0 h-full rounded-xl border-2 border-dashed border-emerald-300/45"
          style={{ aspectRatio: String(1 / CARD_ASPECT_RATIO) }}
        />
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-lg bg-emerald-500/90 px-2 py-0.5 text-[9px] font-bold text-white">
          บัตร ID-1 ↻
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-1 text-center text-[10px] text-white/50">
        ↑ ถือกล้องขนานกับพื้นและมองลงตรงๆ ↑
      </div>
    </div>
  );
}
