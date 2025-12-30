'use client';

import { useState, useEffect } from 'react';
import { FiZap, FiInfo } from 'react-icons/fi';
import { GenerationMode } from '@/services/api';
import api from '@/services/api';

interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  mode: GenerationMode;
  onModeChange: (mode: GenerationMode) => void;
  disabled?: boolean;
}

const SUGGESTED_PROMPTS = [
  'Replace the person in the video with the person from the reference image',
  'Make my character dance like in the video',
  'Transfer the motion to my character',
  'Animate this portrait with the expressions from the video',
  'Swap my face with the character in the video',
];

export function PromptInput({
  value,
  onChange,
  mode,
  onModeChange,
  disabled = false,
}: PromptInputProps) {
  const [parsedMode, setParsedMode] = useState<GenerationMode | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Debounced prompt parsing
  useEffect(() => {
    if (!value || mode !== 'auto') {
      setParsedMode(null);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const result = await api.parsePrompt(value);
        setParsedMode(result.mode as GenerationMode);
        setConfidence(result.confidence);
      } catch (e) {
        console.error('Failed to parse prompt:', e);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [value, mode]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-dark-200">
          Prompt
        </label>
        {mode === 'auto' && parsedMode && (
          <div className="flex items-center gap-1.5 text-xs text-primary-400">
            <FiZap className="w-3 h-3" />
            <span>
              Will use: {parsedMode.replace(/_/g, ' ')}
              {confidence > 0.7 && ' (high confidence)'}
            </span>
          </div>
        )}
      </div>

      <div className="relative">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          disabled={disabled}
          placeholder="Describe what you want to do..."
          rows={3}
          className="input resize-none"
        />

        {showSuggestions && !value && (
          <div className="absolute top-full left-0 right-0 mt-1 z-10 bg-dark-800 border border-dark-600 rounded-lg shadow-xl overflow-hidden">
            <div className="p-2 text-xs text-dark-400 border-b border-dark-700">
              Suggestions
            </div>
            {SUGGESTED_PROMPTS.map((prompt, index) => (
              <button
                key={index}
                onClick={() => onChange(prompt)}
                className="w-full px-3 py-2 text-left text-sm text-dark-200 hover:bg-dark-700 transition-colors"
              >
                {prompt}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Mode selector */}
      <div className="flex flex-wrap gap-2">
        <ModeButton
          mode="auto"
          currentMode={mode}
          onClick={onModeChange}
          label="Auto"
          description="Automatically detect from prompt"
          disabled={disabled}
        />
        <ModeButton
          mode="vace_pose_transfer"
          currentMode={mode}
          onClick={onModeChange}
          label="Pose Transfer"
          description="Transfer poses to character"
          disabled={disabled}
        />
        <ModeButton
          mode="wan_r2v"
          currentMode={mode}
          onClick={onModeChange}
          label="R2V"
          description="Generate new video"
          disabled={disabled}
        />
        <ModeButton
          mode="liveportrait"
          currentMode={mode}
          onClick={onModeChange}
          label="LivePortrait"
          description="Animate portrait"
          disabled={disabled}
        />
        <ModeButton
          mode="deep_live_cam"
          currentMode={mode}
          onClick={onModeChange}
          label="Face Swap"
          description="Swap faces"
          disabled={disabled}
        />
      </div>
    </div>
  );
}

interface ModeButtonProps {
  mode: GenerationMode;
  currentMode: GenerationMode;
  onClick: (mode: GenerationMode) => void;
  label: string;
  description: string;
  disabled?: boolean;
}

function ModeButton({
  mode,
  currentMode,
  onClick,
  label,
  description,
  disabled,
}: ModeButtonProps) {
  const isActive = mode === currentMode;

  return (
    <button
      onClick={() => onClick(mode)}
      disabled={disabled}
      title={description}
      className={`
        px-3 py-1.5 rounded-lg text-sm font-medium transition-all
        ${isActive
          ? 'bg-primary-600 text-white'
          : 'bg-dark-700 text-dark-300 hover:bg-dark-600 hover:text-white'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      {label}
    </button>
  );
}
