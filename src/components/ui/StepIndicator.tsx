'use client';

import { AppStep } from '@/types';
import { CheckCircle, Loader2 } from 'lucide-react';

interface StepIndicatorProps {
  currentStep: AppStep;
  opencvReady: boolean;
  mediapipeReady: boolean;
}

const STEPS = [
  { id: 1, label: 'Instructions' },
  { id: 2, label: 'Measure' },
  { id: 3, label: 'Results' },
];

export function StepIndicator({ currentStep, opencvReady }: StepIndicatorProps) {
  const isLoading = !opencvReady;

  return (
    <div className="flex items-center justify-center gap-0 w-full max-w-sm mx-auto">
      {STEPS.map((step, idx) => {
        const isCompleted = step.id < currentStep;
        const isActive = step.id === currentStep;
        const isUpcoming = step.id > currentStep;

        return (
          <div key={step.id} className="flex items-center">
            {/* Step circle */}
            <div className="flex flex-col items-center">
              <div
                className={`
                  w-9 h-9 rounded-full flex items-center justify-center transition-all duration-300
                  ${isCompleted ? 'bg-emerald-500 shadow-lg shadow-emerald-500/30' : ''}
                  ${isActive ? 'bg-indigo-600 shadow-lg shadow-indigo-500/40 ring-4 ring-indigo-500/20' : ''}
                  ${isUpcoming ? 'bg-white/10 border border-white/20' : ''}
                `}
              >
                {isCompleted && <CheckCircle className="w-5 h-5 text-white" />}
                {isActive && isLoading && step.id === 2 ? (
                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                ) : isActive ? (
                  <span className="text-white text-sm font-bold">{step.id}</span>
                ) : null}
                {isUpcoming && (
                  <span className="text-white/40 text-sm font-medium">{step.id}</span>
                )}
              </div>
              <span
                className={`mt-1.5 text-xs font-medium whitespace-nowrap transition-colors duration-200 ${
                  isActive
                    ? 'text-indigo-300'
                    : isCompleted
                    ? 'text-emerald-400'
                    : 'text-white/30'
                }`}
              >
                {step.label}
              </span>
            </div>

            {/* Connector line */}
            {idx < STEPS.length - 1 && (
              <div
                className={`
                  h-0.5 w-16 mb-5 mx-1 transition-all duration-500
                  ${isCompleted ? 'bg-emerald-500' : 'bg-white/10'}
                `}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
