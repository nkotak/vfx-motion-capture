'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import {
  FiCamera,
  FiCameraOff,
  FiRefreshCw,
  FiCircle,
} from 'react-icons/fi';
import clsx from 'clsx';

interface CameraFeedProps {
  onFrame?: (frameData: Blob) => boolean;
  canSendFrame?: () => boolean;
  processedFrame?: string;
  isActive?: boolean;
  onToggle?: () => void;
  captureResolution?: [number, number];
  jpegQuality?: number;
  targetFps?: number;
  fps?: number;
  latency?: number;
  className?: string;
}

export function CameraFeed({
  onFrame,
  canSendFrame,
  processedFrame,
  isActive = false,
  onToggle,
  captureResolution = [1920, 1080],
  jpegQuality = 90,
  targetFps = 30,
  fps = 0,
  latency = 0,
  className,
}: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastCaptureTimeRef = useRef(0);
  const captureInProgressRef = useRef(false);

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

    captureInProgressRef.current = false;
  }, []);

  // Frame capture loop
  const captureFrame = useCallback((now: number) => {
    if (!videoRef.current || !canvasRef.current || !isActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const frameInterval = 1000 / Math.max(targetFps, 1);

    if (!ctx) return;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      animationRef.current = requestAnimationFrame(captureFrame);
      return;
    }

    if (captureInProgressRef.current || now - lastCaptureTimeRef.current < frameInterval) {
      animationRef.current = requestAnimationFrame(captureFrame);
      return;
    }

    if (canSendFrame && !canSendFrame()) {
      animationRef.current = requestAnimationFrame(captureFrame);
      return;
    }

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.drawImage(video, 0, 0, width, height);

    captureInProgressRef.current = true;
    canvas.toBlob((blob) => {
      if (blob && (!canSendFrame || canSendFrame())) {
        const accepted = onFrame?.(blob) ?? false;
        if (accepted) {
          lastCaptureTimeRef.current = performance.now();
        }
      }

      captureInProgressRef.current = false;
      if (isActive) {
        animationRef.current = requestAnimationFrame(captureFrame);
      }
    }, 'image/jpeg', Math.max(0.5, Math.min(jpegQuality, 100)) / 100);
  }, [canSendFrame, isActive, jpegQuality, onFrame, targetFps]);

  // Start camera stream
  const startCamera = useCallback(async () => {
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          width: { ideal: captureResolution[0] },
          height: { ideal: captureResolution[1] },
          frameRate: { ideal: targetFps, max: Math.max(targetFps, 30) },
          facingMode: 'user',
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        if (isActive) {
          lastCaptureTimeRef.current = 0;
          if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
          }
          animationRef.current = requestAnimationFrame(captureFrame);
        }
      }

      setHasCamera(true);
    } catch (error) {
      console.error('Failed to start camera:', error);
      setHasCamera(false);
    }
  }, [captureFrame, captureResolution, isActive, selectedDevice, targetFps]);

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
            <img
              src={processedFrame}
              alt="Processed camera output"
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
