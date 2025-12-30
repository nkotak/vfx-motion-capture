'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiX, FiImage } from 'react-icons/fi';
import clsx from 'clsx';
import { UploadResponse } from '@/services/api';

interface ImageUploaderProps {
  value: UploadResponse | null;
  onChange: (file: File) => Promise<void>;
  onClear: () => void;
  label?: string;
  description?: string;
  accept?: string[];
  disabled?: boolean;
}

export function ImageUploader({
  value,
  onChange,
  onClear,
  label = 'Reference Image',
  description = 'Upload the character/person you want to use',
  accept = ['image/jpeg', 'image/png', 'image/webp'],
  disabled = false,
}: ImageUploaderProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      setIsUploading(true);
      setError(null);

      try {
        await onChange(acceptedFiles[0]);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Upload failed');
      } finally {
        setIsUploading(false);
      }
    },
    [onChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
    maxFiles: 1,
    disabled: disabled || isUploading,
  });

  const thumbnailUrl = value?.thumbnail_url || (value?.id ? `/api/files/${value.id}/thumbnail` : null);

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-dark-200">{label}</label>

      {!value ? (
        <div
          {...getRootProps()}
          className={clsx(
            'upload-zone h-48 p-4',
            isDragActive && 'active',
            (disabled || isUploading) && 'opacity-50 cursor-not-allowed'
          )}
        >
          <input {...getInputProps()} />
          {isUploading ? (
            <div className="text-center">
              <div className="animate-spin rounded-full h-10 w-10 border-2 border-primary-500 border-t-transparent mx-auto mb-3" />
              <p className="text-dark-300">Uploading...</p>
            </div>
          ) : (
            <>
              <FiUpload className="w-10 h-10 text-dark-400 mb-3" />
              <p className="text-dark-300 text-center">
                {isDragActive ? 'Drop the image here' : 'Drag & drop or click to upload'}
              </p>
              <p className="text-dark-500 text-sm mt-1">{description}</p>
            </>
          )}
        </div>
      ) : (
        <div className="relative group">
          <div className="aspect-square rounded-xl overflow-hidden bg-dark-800 border border-dark-600">
            {thumbnailUrl ? (
              <img
                src={thumbnailUrl}
                alt="Reference"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <FiImage className="w-16 h-16 text-dark-500" />
              </div>
            )}
          </div>

          <button
            onClick={onClear}
            disabled={disabled}
            className="absolute top-2 right-2 p-1.5 bg-dark-900/80 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
          >
            <FiX className="w-4 h-4" />
          </button>

          <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-dark-900/90 to-transparent">
            <p className="text-sm text-white truncate">{value.filename}</p>
            {value.resolution && (
              <p className="text-xs text-dark-400">
                {value.resolution[0]} x {value.resolution[1]}
              </p>
            )}
          </div>
        </div>
      )}

      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}
