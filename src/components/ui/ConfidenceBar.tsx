'use client';

interface ConfidenceBarProps {
  confidence: number; // 0–1
  label?: string;
}

export function ConfidenceBar({ confidence, label = 'Card Detection' }: ConfidenceBarProps) {
  const pct = Math.round(confidence * 100);

  const color =
    pct >= 80
      ? 'from-emerald-500 to-green-400'
      : pct >= 55
      ? 'from-yellow-500 to-amber-400'
      : 'from-red-500 to-rose-400';

  const statusText =
    pct >= 70 ? '✓ Locked' : pct >= 40 ? '⟳ Detecting…' : '✗ Not found';

  const statusColor =
    pct >= 70 ? 'text-emerald-400' : pct >= 40 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-white/60">{label}</span>
        <span className={`text-xs font-semibold ${statusColor}`}>{statusText}</span>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex items-center justify-between mt-1">
        <span className="text-[10px] text-white/30">0%</span>
        <span className="text-[10px] font-mono text-white/50">{pct}%</span>
        <span className="text-[10px] text-white/30">100%</span>
      </div>
    </div>
  );
}
