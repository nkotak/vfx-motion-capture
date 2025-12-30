'use client';

import { FiPlay, FiLoader, FiX } from 'react-icons/fi';
import clsx from 'clsx';

interface GenerateButtonProps {
  onClick: () => void;
  onCancel?: () => void;
  disabled?: boolean;
  isGenerating?: boolean;
  progress?: number;
  currentStep?: string;
}

export function GenerateButton({
  onClick,
  onCancel,
  disabled = false,
  isGenerating = false,
  progress = 0,
  currentStep = '',
}: GenerateButtonProps) {
  if (isGenerating) {
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <button
            onClick={onCancel}
            className="btn btn-danger flex items-center gap-2"
          >
            <FiX className="w-4 h-4" />
            Cancel
          </button>
          <div className="flex-1">
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-dark-300">{currentStep || 'Processing...'}</span>
              <span className="text-primary-400">{Math.round(progress)}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-bar-fill"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'btn btn-primary w-full flex items-center justify-center gap-2 py-3 text-lg',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      <FiPlay className="w-5 h-5" />
      Generate
    </button>
  );
}
