'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import {
  FiCamera,
  FiCameraOff,
  FiRefreshCw,
  FiSettings,
  FiCircle,
} from 'react-icons/fi';
import clsx from 'clsx';

interface CameraFeedProps {
  onFrame?: (frameData: string) => void;
  processedFrame?: string;
  isActive?: boolean;
  onToggle?: () => void;
  fps?: number;
  latency?: number;
  className?: string;
}

export function CameraFeed({
  onFrame,
  processedFrame,
  isActive = false,
  onToggle,
  fps = 0,
  latency = 0,
  className,
}: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const processedRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);

  const [hasCamera, setHasCamera] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');

  // Get available cameras
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      const cameras = devices.filter((d) => d.kind === 'videoinput');
      setDevices(cameras);
      if (cameras.length > 0 && !selectedDevice) {
        setSelectedDevice(cameras[0].deviceId);
      }
    });
  }, [selectedDevice]);

  // Start camera stream
  const startCamera = useCallback(async () => {
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setHasCamera(true);
    } catch (error) {
      console.error('Failed to start camera:', error);
      setHasCamera(false);
    }
  }, [selectedDevice]);

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, []);

  // Frame capture loop
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    ctx.drawImage(video, 0, 0);

    const frameData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    onFrame?.(frameData);

    animationRef.current = requestAnimationFrame(captureFrame);
  }, [isActive, onFrame]);

  // Handle activation
  useEffect(() => {
    if (isActive) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isActive, startCamera, stopCamera]);

  // Start frame capture when active
  useEffect(() => {
    if (isActive && streamRef.current) {
      animationRef.current = requestAnimationFrame(captureFrame);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, captureFrame]);

  // Draw processed frame
  useEffect(() => {
    if (!processedFrame || !processedRef.current) return;

    const canvas = processedRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${processedFrame}`;
  }, [processedFrame]);

  return (
    <div className={clsx('relative', className)}>
      {/* Camera view */}
      <div className="relative aspect-video bg-dark-900 rounded-xl overflow-hidden">
        {hasCamera ? (
          <>
            {/* Raw camera feed (hidden when processing) */}
            <video
              ref={videoRef}
              className={clsx(
                'w-full h-full object-cover',
                processedFrame ? 'hidden' : 'block'
              )}
              muted
              playsInline
            />

            {/* Processed output */}
            <canvas
              ref={processedRef}
              className={clsx(
                'w-full h-full object-cover',
                processedFrame ? 'block' : 'hidden'
              )}
            />

            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Status overlay */}
            <div className="absolute top-3 left-3 flex items-center gap-2">
              {isActive && (
                <div className="flex items-center gap-1.5 px-2 py-1 bg-dark-900/80 rounded-lg text-xs">
                  <FiCircle className={clsx(
                    'w-2 h-2',
                    isRecording ? 'text-red-500 animate-pulse' : 'text-green-500'
                  )} />
                  <span>LIVE</span>
                </div>
              )}
            </div>

            {/* Stats overlay */}
            {isActive && (
              <div className="absolute top-3 right-3 flex items-center gap-3 text-xs">
                <div className="px-2 py-1 bg-dark-900/80 rounded-lg">
                  {fps} FPS
                </div>
                <div className="px-2 py-1 bg-dark-900/80 rounded-lg">
                  {latency.toFixed(0)}ms
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-dark-400">
            <FiCameraOff className="w-16 h-16 mb-4" />
            <p>Camera not available</p>
            <button
              onClick={startCamera}
              className="mt-4 btn btn-secondary flex items-center gap-2"
            >
              <FiRefreshCw className="w-4 h-4" />
              Retry
            </button>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={onToggle}
          className={clsx(
            'btn flex items-center gap-2',
            isActive ? 'btn-danger' : 'btn-primary'
          )}
        >
          <FiCamera className="w-4 h-4" />
          {isActive ? 'Stop' : 'Start'}
        </button>

        {devices.length > 1 && (
          <select
            value={selectedDevice}
            onChange={(e) => setSelectedDevice(e.target.value)}
            disabled={isActive}
            className="select flex-1"
          >
            {devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Camera ${devices.indexOf(device) + 1}`}
              </option>
            ))}
          </select>
        )}

        <button
          onClick={() => setIsRecording(!isRecording)}
          disabled={!isActive}
          className={clsx(
            'btn',
            isRecording ? 'btn-danger' : 'btn-secondary'
          )}
        >
          <FiCircle className={clsx('w-4 h-4', isRecording && 'fill-current')} />
        </button>
      </div>
    </div>
  );
}
