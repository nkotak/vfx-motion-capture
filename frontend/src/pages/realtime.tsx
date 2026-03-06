'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { toast } from 'react-hot-toast';
import { FiArrowLeft, FiCamera, FiSettings, FiPlay, FiSquare } from 'react-icons/fi';

import { ImageUploader } from '@/components/ImageUploader';
import { CameraFeed } from '@/components/CameraFeed';
import { useRealtimeWebSocket } from '@/hooks/useWebSocket';
import api, {
  UploadResponse,
  GenerationMode,
  RealtimeCompatibility,
  RealtimeSession,
  RealtimeSessionMetrics,
  RealtimeWorkerTelemetry,
} from '@/services/api';

const RESOLUTION_PRESETS: Array<{ label: string; value: [number, number] }> = [
  { label: '720p (1280x720)', value: [1280, 720] },
  { label: '1080p (1920x1080)', value: [1920, 1080] },
  { label: '4K UHD (3840x2160)', value: [3840, 2160] },
];

const FPS_PRESETS = [24, 30, 60];

export default function RealtimePage() {
  const [referenceImage, setReferenceImage] = useState<UploadResponse | null>(null);
  const [session, setSession] = useState<RealtimeSession | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [mode, setMode] = useState<'liveportrait' | 'deep_live_cam'>('liveportrait');
  const [processedFrame, setProcessedFrame] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [captureResolution, setCaptureResolution] = useState<[number, number]>([1920, 1080]);
  const [targetFps, setTargetFps] = useState(24);
  const [jpegQuality, setJpegQuality] = useState(92);
  const [compatibility, setCompatibility] = useState<RealtimeCompatibility | null>(null);
  const [sessionMetrics, setSessionMetrics] = useState<RealtimeSessionMetrics | null>(null);
  const [workerTelemetry, setWorkerTelemetry] = useState<RealtimeWorkerTelemetry[]>([]);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  const processedFrameRef = useRef<string | null>(null);

  const updateProcessedFrame = useCallback((nextFrame: string | null) => {
    if (processedFrameRef.current?.startsWith('blob:')) {
      URL.revokeObjectURL(processedFrameRef.current);
    }

    processedFrameRef.current = nextFrame;
    setProcessedFrame(nextFrame);
  }, []);

  useEffect(() => () => {
    if (processedFrameRef.current?.startsWith('blob:')) {
      URL.revokeObjectURL(processedFrameRef.current);
    }
  }, []);

  useEffect(() => {
    let isCancelled = false;
    api.checkRealtimeCompatibility()
      .then((response) => {
        if (!isCancelled) {
          setCompatibility(response);
          if (response.recommended_session) {
            setCaptureResolution(response.recommended_session.input_resolution);
            setTargetFps(response.recommended_session.target_fps);
            setJpegQuality(response.recommended_session.jpeg_quality);
          }
        }
      })
      .catch((error) => {
        if (!isCancelled) {
          console.error('Failed to load realtime compatibility:', error);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!session?.session_id || !isActive) {
      setSessionMetrics(null);
      return;
    }

    let isCancelled = false;
    const loadMetrics = async () => {
      try {
        const response = await api.getRealtimeSessionMetrics(session.session_id);
        if (!isCancelled) {
          setSessionMetrics(response.metrics);
          setSession((previous) => (
            previous && previous.session_id === response.session_id
              ? {
                  ...previous,
                  status: response.status,
                  worker_id: response.worker_id ?? previous.worker_id,
                  config: response.config,
                  metrics: response.metrics,
                }
              : previous
          ));
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load realtime metrics:', error);
        }
      }
    };

    loadMetrics();
    const intervalId = window.setInterval(loadMetrics, 1000);

    return () => {
      isCancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isActive, session?.session_id]);

  useEffect(() => {
    if (!showDebugPanel || !isActive) {
      setWorkerTelemetry([]);
      return;
    }

    let isCancelled = false;
    const loadWorkers = async () => {
      try {
        const response = await api.getRealtimeWorkers();
        if (!isCancelled) {
          setWorkerTelemetry(response.workers);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load realtime worker telemetry:', error);
        }
      }
    };

    loadWorkers();
    const intervalId = window.setInterval(loadWorkers, 1000);

    return () => {
      isCancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isActive, showDebugPanel]);

  const { isConnected, fps, latency, sendFrame, canSendFrame } = useRealtimeWebSocket(
    session?.session_id || null,
    {
      onFrame: (frameUrl) => {
        updateProcessedFrame(frameUrl);
      },
      onError: (error) => {
        toast.error(error);
      },
    }
  );

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    try {
      const response = await api.uploadImage(file);
      setReferenceImage(response);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleStart = async () => {
    if (!referenceImage) {
      toast.error('Please upload a reference image first');
      return;
    }

    try {
      const newSession = await api.createRealtimeSession({
        reference_image_id: referenceImage.id,
        mode,
        target_fps: targetFps,
        face_only: mode === 'liveportrait',
        smoothing: 0.5,
        enhance_face: true,
        input_resolution: captureResolution,
        output_resolution: captureResolution,
        jpeg_quality: jpegQuality,
        jpeg_subsampling: '420',
        binary_transport: true,
        full_frame_inference: true,
        tile_size: null,
        tile_overlap: 64,
        max_inflight_frames: 1,
        allow_frame_drop: true,
        adaptive_quality: true,
        adaptive_latency_budget_ms: Math.round(1000 / targetFps),
        adaptive_jpeg_step: 5,
        adaptive_min_jpeg_quality: 75,
        adaptive_cooldown_frames: 24,
        adaptive_tile_size: 1024,
        adaptive_min_tile_size: 512,
      });

      setSession(newSession);
      setIsActive(true);
      toast.success('Real-time session started');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to start session');
    }
  };

  const handleStop = async () => {
    setIsActive(false);

    if (session) {
      try {
        await api.deleteRealtimeSession(session.session_id);
      } catch (e) {
        console.error('Failed to delete session:', e);
      }
    }

    setSession(null);
    updateProcessedFrame(null);
  };

  const handleFrame = useCallback(
    (frameData: Blob) => {
      if (isConnected) {
        return sendFrame(frameData);
      }
      return false;
    },
    [isConnected, sendFrame]
  );

  const selectedResolution = `${captureResolution[0]}x${captureResolution[1]}`;
  const applyRecommendedPreset = useCallback(() => {
    if (!compatibility?.recommended_session) {
      return;
    }

    setCaptureResolution(compatibility.recommended_session.input_resolution);
    setTargetFps(compatibility.recommended_session.target_fps);
    setJpegQuality(compatibility.recommended_session.jpeg_quality);
    if (compatibility.recommended_mode && compatibility.recommended_mode !== 'auto') {
      setMode(compatibility.recommended_mode as 'liveportrait' | 'deep_live_cam');
    }
  }, [compatibility]);

  return (
    <>
      <Head>
        <title>Real-time Mode | VFX Motion Capture</title>
      </Head>

      <div className="min-h-screen">
        {/* Header */}
        <header className="border-b border-dark-700 bg-dark-900/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Link
                  href="/"
                  className="p-2 rounded-lg hover:bg-dark-700 transition-colors"
                >
                  <FiArrowLeft className="w-5 h-5" />
                </Link>
                <div className="flex items-center gap-3">
                  <FiCamera className="w-6 h-6 text-primary-500" />
                  <h1 className="text-xl font-bold">Real-time Mode</h1>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {isActive && (
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-dark-400">
                      {isConnected ? (
                        <span className="text-green-400">Connected</span>
                      ) : (
                        <span className="text-yellow-400">Connecting...</span>
                      )}
                    </span>
                    <span className="text-dark-400">{fps} FPS</span>
                    <span className="text-dark-400">{latency.toFixed(0)}ms</span>
                    {sessionMetrics && (
                      <>
                        <span className="text-dark-400">Dropped {sessionMetrics.dropped_frames}</span>
                        {typeof sessionMetrics.worker_id === 'number' && (
                          <span className="text-dark-400">Worker {sessionMetrics.worker_id}</span>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {/* Left sidebar - Settings */}
            <div className="space-y-6">
              {/* Reference image */}
              <div className="card p-6">
                <ImageUploader
                  value={referenceImage}
                  onChange={handleUpload}
                  onClear={() => setReferenceImage(null)}
                  label="Reference Character"
                  description="Character to transform into"
                  disabled={isActive}
                />
              </div>

              {/* Mode selection */}
              <div className="card p-6 space-y-4">
                <h3 className="font-medium flex items-center gap-2">
                  <FiSettings className="w-4 h-4" />
                  Mode
                </h3>

                <div className="space-y-2">
                  <button
                    onClick={() => setMode('liveportrait')}
                    disabled={isActive}
                    className={`
                      w-full p-3 rounded-lg text-left transition-all
                      ${mode === 'liveportrait'
                        ? 'bg-primary-600 text-white'
                        : 'bg-dark-700 hover:bg-dark-600'
                      }
                      ${isActive ? 'opacity-50 cursor-not-allowed' : ''}
                    `}
                  >
                    <div className="font-medium">LivePortrait</div>
                    <div className="text-sm opacity-70">Face animation only (fastest)</div>
                  </button>

                  <button
                    onClick={() => setMode('deep_live_cam')}
                    disabled={isActive}
                    className={`
                      w-full p-3 rounded-lg text-left transition-all
                      ${mode === 'deep_live_cam'
                        ? 'bg-primary-600 text-white'
                        : 'bg-dark-700 hover:bg-dark-600'
                      }
                      ${isActive ? 'opacity-50 cursor-not-allowed' : ''}
                    `}
                  >
                    <div className="font-medium">Face Swap</div>
                    <div className="text-sm opacity-70">Full face replacement</div>
                  </button>
                </div>
              </div>

              <div className="card p-6 space-y-4">
                <h3 className="font-medium flex items-center gap-2">
                  <FiSettings className="w-4 h-4" />
                  Realtime pipeline
                </h3>

                <div className="space-y-3">
                  {compatibility && (
                    <div className="rounded-lg border border-dark-700 bg-dark-800 p-3 text-xs text-dark-300">
                      <div className="font-medium text-white">
                        {compatibility.gpu_name || 'Local runtime'} · {compatibility.runtime || 'cpu'}
                      </div>
                      <div className="mt-1">
                        Capability: {compatibility.capability} · Estimated realtime FPS: {compatibility.estimated_fps}
                      </div>
                      {compatibility.recommended_session && (
                        <button
                          onClick={applyRecommendedPreset}
                          disabled={isActive}
                          className="mt-3 btn btn-secondary w-full"
                        >
                          Apply recommended preset
                        </button>
                      )}
                    </div>
                  )}

                  <label className="block text-sm">
                    <span className="mb-1 block text-dark-300">Capture + output resolution</span>
                    <select
                      value={selectedResolution}
                      onChange={(e) => {
                        const [width, height] = e.target.value.split('x').map(Number);
                        setCaptureResolution([width, height]);
                      }}
                      disabled={isActive}
                      className="select w-full"
                    >
                      {RESOLUTION_PRESETS.map((preset) => (
                        <option
                          key={`${preset.value[0]}x${preset.value[1]}`}
                          value={`${preset.value[0]}x${preset.value[1]}`}
                        >
                          {preset.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block text-sm">
                    <span className="mb-1 block text-dark-300">Target FPS</span>
                    <select
                      value={targetFps}
                      onChange={(e) => setTargetFps(Number(e.target.value))}
                      disabled={isActive}
                      className="select w-full"
                    >
                      {FPS_PRESETS.map((fpsOption) => (
                        <option key={fpsOption} value={fpsOption}>
                          {fpsOption} FPS
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block text-sm">
                    <span className="mb-1 block text-dark-300">JPEG quality ({jpegQuality})</span>
                    <input
                      type="range"
                      min={80}
                      max={100}
                      step={1}
                      value={jpegQuality}
                      onChange={(e) => setJpegQuality(Number(e.target.value))}
                      disabled={isActive}
                      className="w-full"
                    />
                  </label>

                  <p className="text-xs text-dark-400">
                    Sessions use binary JPEG transport, full-frame inference, and keep the output
                    resolution matched to the requested camera resolution.
                  </p>
                </div>
              </div>

              {/* Controls */}
              <div className="card p-6">
                {!isActive ? (
                  <button
                    onClick={handleStart}
                    disabled={!referenceImage}
                    className="btn btn-primary w-full flex items-center justify-center gap-2 py-3"
                  >
                    <FiPlay className="w-5 h-5" />
                    Start
                  </button>
                ) : (
                  <button
                    onClick={handleStop}
                    className="btn btn-danger w-full flex items-center justify-center gap-2 py-3"
                  >
                    <FiSquare className="w-5 h-5" />
                    Stop
                  </button>
                )}

                <button
                  onClick={() => setShowDebugPanel((current) => !current)}
                  className="btn btn-secondary w-full mt-3"
                >
                  {showDebugPanel ? 'Hide debug panel' : 'Show debug panel'}
                </button>
              </div>
            </div>

            {/* Main area - Camera feed */}
            <div className="lg:col-span-3">
              <div className="card p-6">
                <h2 className="text-lg font-semibold mb-4">Live Output</h2>

                <CameraFeed
                  onFrame={handleFrame}
                  canSendFrame={canSendFrame}
                  processedFrame={processedFrame || undefined}
                  isActive={isActive}
                  onToggle={isActive ? handleStop : handleStart}
                  captureResolution={session?.config.input_resolution ?? captureResolution}
                  jpegQuality={session?.config.jpeg_quality ?? jpegQuality}
                  targetFps={session?.config.target_fps ?? targetFps}
                  fps={fps}
                  latency={latency}
                />

                {showDebugPanel && (
                  <div className="mt-6 grid grid-cols-1 xl:grid-cols-2 gap-4 text-sm">
                    <div className="rounded-lg border border-dark-700 bg-dark-800 p-4 space-y-3">
                      <h3 className="font-medium">Session debug</h3>
                      <div className="grid grid-cols-2 gap-3 text-dark-300">
                        <div>Worker latency avg: {sessionMetrics?.avg_worker_latency_ms?.toFixed(1) ?? '0.0'}ms</div>
                        <div>Total latency avg: {sessionMetrics?.avg_total_latency_ms?.toFixed(1) ?? '0.0'}ms</div>
                        <div>Decode avg: {sessionMetrics?.avg_decode_ms?.toFixed(1) ?? '0.0'}ms</div>
                        <div>Inference avg: {sessionMetrics?.avg_inference_ms?.toFixed(1) ?? '0.0'}ms</div>
                        <div>Encode avg: {sessionMetrics?.avg_encode_ms?.toFixed(1) ?? '0.0'}ms</div>
                        <div>Resize avg: {sessionMetrics?.avg_resize_ms?.toFixed(1) ?? '0.0'}ms</div>
                        <div>Processing mode: {sessionMetrics?.current_processing_mode || 'n/a'}</div>
                        <div>Effective quality: {sessionMetrics?.current_jpeg_quality ?? 'n/a'}</div>
                        <div>Tile size: {sessionMetrics?.current_tile_size ?? 'off'}</div>
                        <div>Full-frame: {String(sessionMetrics?.current_full_frame_inference ?? true)}</div>
                        <div>Shared mem in: {sessionMetrics?.shared_memory_in_count ?? 0}</div>
                        <div>Shared mem out: {sessionMetrics?.shared_memory_out_count ?? 0}</div>
                        <div>Inline in: {sessionMetrics?.inline_transport_in_count ?? 0}</div>
                        <div>Inline out: {sessionMetrics?.inline_transport_out_count ?? 0}</div>
                      </div>

                      {sessionMetrics?.adaptive_events && sessionMetrics.adaptive_events.length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Adaptive events</h4>
                          <div className="space-y-2 max-h-48 overflow-auto">
                            {sessionMetrics.adaptive_events.slice().reverse().map((event) => (
                              <div key={`${event.timestamp}-${event.message}`} className="rounded bg-dark-900 px-3 py-2 text-xs text-dark-300">
                                <div className="text-white">{event.message}</div>
                                <div className="mt-1 opacity-70">
                                  {new Date(event.timestamp).toLocaleTimeString()} · quality {event.jpeg_quality ?? 'n/a'} · tile {event.tile_size ?? 'off'}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="rounded-lg border border-dark-700 bg-dark-800 p-4 space-y-3">
                      <h3 className="font-medium">Worker telemetry</h3>
                      <div className="space-y-3">
                        {workerTelemetry.length === 0 ? (
                          <div className="text-dark-400">No worker telemetry available</div>
                        ) : workerTelemetry.map((worker) => (
                          <div key={worker.worker_id} className="rounded bg-dark-900 px-3 py-3 text-xs text-dark-300 space-y-2">
                            <div className="flex items-center justify-between text-white">
                              <span>Worker {worker.worker_id}</span>
                              <span>{worker.process_alive ? 'alive' : 'stopped'}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div>Active sessions: {worker.active_sessions}</div>
                              <div>Saturation: {(worker.saturation * 100).toFixed(0)}%</div>
                              <div>Pending: {worker.pending_requests}</div>
                              <div>Processed: {worker.processed_requests}</div>
                              <div>Input queue: {worker.input_queue_size}</div>
                              <div>Output queue: {worker.output_queue_size}</div>
                              <div>Shared mem in: {worker.shared_memory_in_count}</div>
                              <div>Shared mem out: {worker.shared_memory_out_count}</div>
                              <div>Latency avg: {worker.avg_latency_ms.toFixed(1)}ms</div>
                              <div>Errors: {worker.error_count}</div>
                            </div>
                            {worker.session_ids.length > 0 && (
                              <div className="opacity-70">Sessions: {worker.session_ids.join(', ')}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {!isActive && !referenceImage && (
                  <div className="mt-6 p-4 bg-dark-800 rounded-lg border border-dark-700">
                    <h3 className="font-medium mb-2">Getting Started</h3>
                    <ol className="list-decimal list-inside space-y-1 text-sm text-dark-400">
                      <li>Upload a reference image of the character you want to become</li>
                      <li>Select a processing mode (LivePortrait for face animation)</li>
                      <li>Click Start to begin real-time transformation</li>
                      <li>Your camera feed will be transformed in real-time</li>
                    </ol>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
