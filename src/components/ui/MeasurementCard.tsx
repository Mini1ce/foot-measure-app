'use client';

import { MeasurementResult } from '@/types';
import { Ruler, ArrowLeftRight, Footprints, RefreshCcw } from 'lucide-react';

interface MeasurementCardProps {
  result: MeasurementResult;
  onRemeasure: () => void;
}

const FIT_LABELS = {
  exact: { text: 'Perfect Fit', color: 'text-emerald-400' },
  'half-size-up': { text: 'Go Half Size Up', color: 'text-yellow-400' },
  'half-size-down': { text: 'Go Half Size Down', color: 'text-blue-400' },
};

export function MeasurementCard({ result, onRemeasure }: MeasurementCardProps) {
  const fit = result.shoeSize ? FIT_LABELS[result.shoeSize.fit] : null;

  return (
    <div className="w-full space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Dimension pills */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gradient-to-br from-indigo-600/30 to-purple-600/20 rounded-2xl p-4 border border-indigo-500/20">
          <div className="flex items-center gap-2 mb-2">
            <Ruler className="w-4 h-4 text-indigo-400" />
            <span className="text-xs font-medium text-indigo-300 uppercase tracking-wider">ความยาว</span>
          </div>
          <div className="text-3xl font-bold text-white">
            {result.footLengthCm}
            <span className="text-lg font-normal text-white/50 ml-1">cm</span>
          </div>
          <div className="text-xs text-white/40 mt-1">{(result.footLengthCm * 10).toFixed(0)} mm</div>
        </div>

        <div className="bg-gradient-to-br from-emerald-600/30 to-teal-600/20 rounded-2xl p-4 border border-emerald-500/20">
          <div className="flex items-center gap-2 mb-2">
            <ArrowLeftRight className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-medium text-emerald-300 uppercase tracking-wider">ความกว้าง</span>
          </div>
          <div className="text-3xl font-bold text-white">
            {result.footWidthCm}
            <span className="text-lg font-normal text-white/50 ml-1">cm</span>
          </div>
          <div className="text-xs text-white/40 mt-1">{(result.footWidthCm * 10).toFixed(0)} mm</div>
        </div>
      </div>

      {/* Shoe size recommendation */}
      {result.shoeSize && fit && <div className="bg-white/5 rounded-2xl p-4 border border-white/10">
        <div className="flex items-center gap-2 mb-3">
          <Footprints className="w-4 h-4 text-violet-400" />
          <span className="text-sm font-semibold text-white">แนวทางเลือกไซซ์รองเท้า</span>
          <span className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full bg-white/10 ${fit.color}`}>
            {fit.text}
          </span>
        </div>

        <div className="grid grid-cols-4 gap-2">
          {[
            { label: 'EU', value: result.shoeSize.eu, color: 'bg-violet-600/30 text-violet-300' },
            { label: 'US♂', value: result.shoeSize.usMen, color: 'bg-blue-600/30 text-blue-300' },
            { label: 'US♀', value: result.shoeSize.usWomen, color: 'bg-pink-600/30 text-pink-300' },
            { label: 'UK', value: result.shoeSize.uk, color: 'bg-amber-600/30 text-amber-300' },
          ].map(({ label, value, color }) => (
            <div key={label} className={`rounded-xl p-2 text-center ${color}`}>
              <div className="text-[10px] font-medium opacity-70 mb-0.5">{label}</div>
              <div className="text-lg font-bold">{value}</div>
            </div>
          ))}
        </div>
      </div>}

      {result.warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-200">
          {result.warnings.join(' • ')}
        </div>
      )}

      {/* Calibration info */}
      <div className="text-xs text-white/30 text-center">
        Homography • คุณภาพ {result.quality} • ความมั่นใจ {Math.round(result.confidence * 100)}% · {new Date(result.timestamp).toLocaleTimeString()}
      </div>

      {/* Remeasure button */}
      <button
        onClick={onRemeasure}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-2xl border border-white/10 text-white/60 hover:text-white hover:border-white/30 hover:bg-white/5 transition-all duration-200 text-sm font-medium"
      >
        <RefreshCcw className="w-4 h-4" />
        วัดใหม่
      </button>
    </div>
  );
}
