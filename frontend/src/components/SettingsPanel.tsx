'use client';

import { FiSliders, FiInfo } from 'react-icons/fi';
import { QualityPreset } from '@/services/api';

interface SettingsPanelProps {
  quality: QualityPreset;
  onQualityChange: (quality: QualityPreset) => void;
  fps: number;
  onFpsChange: (fps: number) => void;
  strength: number;
  onStrengthChange: (strength: number) => void;
  duration: number | null;
  onDurationChange: (duration: number | null) => void;
  disabled?: boolean;
}

const QUALITY_OPTIONS: { value: QualityPreset; label: string; description: string }[] = [
  { value: 'draft', label: 'Draft', description: 'Fast preview, lower quality' },
  { value: 'standard', label: 'Standard', description: 'Balanced quality and speed' },
  { value: 'high', label: 'High', description: 'High quality, slower' },
  { value: 'ultra', label: 'Ultra', description: 'Maximum quality' },
];

const FPS_OPTIONS = [12, 15, 24, 30, 60];

export function SettingsPanel({
  quality,
  onQualityChange,
  fps,
  onFpsChange,
  strength,
  onStrengthChange,
  duration,
  onDurationChange,
  disabled = false,
}: SettingsPanelProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-dark-200">
        <FiSliders className="w-4 h-4" />
        <span className="font-medium">Settings</span>
      </div>

      {/* Quality */}
      <div className="space-y-2">
        <label className="text-sm text-dark-300">Quality</label>
        <div className="grid grid-cols-2 gap-2">
          {QUALITY_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() => onQualityChange(option.value)}
              disabled={disabled}
              title={option.description}
              className={`
                px-3 py-2 rounded-lg text-sm transition-all text-left
                ${quality === option.value
                  ? 'bg-primary-600 text-white'
                  : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* FPS */}
      <div className="space-y-2">
        <label className="text-sm text-dark-300">Frame Rate</label>
        <div className="flex gap-2">
          {FPS_OPTIONS.map((fpsOption) => (
            <button
              key={fpsOption}
              onClick={() => onFpsChange(fpsOption)}
              disabled={disabled}
              className={`
                flex-1 px-2 py-1.5 rounded-lg text-sm transition-all
                ${fps === fpsOption
                  ? 'bg-primary-600 text-white'
                  : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              {fpsOption}
            </button>
          ))}
        </div>
      </div>

      {/* Strength */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm text-dark-300">Transformation Strength</label>
          <span className="text-sm text-primary-400">{Math.round(strength * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={strength * 100}
          onChange={(e) => onStrengthChange(parseInt(e.target.value) / 100)}
          disabled={disabled}
          className="w-full accent-primary-500"
        />
        <div className="flex justify-between text-xs text-dark-500">
          <span>Subtle</span>
          <span>Full replacement</span>
        </div>
      </div>

      {/* Duration */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm text-dark-300">Duration (seconds)</label>
          <button
            onClick={() => onDurationChange(null)}
            disabled={disabled}
            className="text-xs text-primary-400 hover:text-primary-300"
          >
            Auto
          </button>
        </div>
        <input
          type="number"
          min="1"
          max="60"
          value={duration || ''}
          onChange={(e) => onDurationChange(e.target.value ? parseInt(e.target.value) : null)}
          disabled={disabled}
          placeholder="Auto (match input)"
          className="input"
        />
      </div>

      {/* Info */}
      <div className="p-3 bg-dark-800 rounded-lg border border-dark-700">
        <div className="flex items-start gap-2 text-xs text-dark-400">
          <FiInfo className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <p>
            Higher quality and longer duration will take more time to generate.
            Draft mode is recommended for testing prompts.
          </p>
        </div>
      </div>
    </div>
  );
}
