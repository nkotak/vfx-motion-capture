/**
 * API client for VFX Motion Capture backend
 */

const API_BASE = '/api';

export interface UploadResponse {
  id: string;
  filename: string;
  file_type: 'image' | 'video';
  size_bytes: number;
  duration?: number;
  resolution?: [number, number];
  fps?: number;
  thumbnail_url?: string;
}

export interface GenerateRequest {
  reference_image_id: string;
  input_video_id?: string;
  prompt: string;
  mode: GenerationMode;
  quality: QualityPreset;
  duration?: number;
  fps?: number;
  resolution?: [number, number];
  output_format?: 'mp4' | 'webm' | 'gif';
  strength?: number;
  preserve_background?: boolean;
  seed?: number;
  extra_params?: Record<string, unknown>;
}

export type GenerationMode =
  | 'vace_pose_transfer'
  | 'vace_motion_transfer'
  | 'wan_r2v'
  | 'liveportrait'
  | 'deep_live_cam'
  | 'auto';

export type QualityPreset = 'draft' | 'standard' | 'high' | 'ultra';

export interface GenerationModeInfo {
  value: GenerationMode;
  name: string;
  label: string;
  description: string;
  suggested_prompt: string;
  requires_input_video: boolean;
  supports_input_video: boolean;
  supports_prompt: boolean;
  experimental: boolean;
}

export const FALLBACK_GENERATION_MODES: GenerationModeInfo[] = [
  {
    value: 'auto',
    name: 'AUTO',
    label: 'Auto',
    description: 'Automatically selects the best generation method based on your prompt and inputs.',
    suggested_prompt: 'Describe what you want to do with the reference image and video',
    requires_input_video: false,
    supports_input_video: true,
    supports_prompt: true,
    experimental: false,
  },
  {
    value: 'vace_pose_transfer',
    name: 'VACE_POSE_TRANSFER',
    label: 'Pose Transfer',
    description: 'Transfers the motion and poses from the input video to the reference character.',
    suggested_prompt: 'Replace the person in the video with the person from the reference image',
    requires_input_video: true,
    supports_input_video: true,
    supports_prompt: true,
    experimental: false,
  },
  {
    value: 'vace_motion_transfer',
    name: 'VACE_MOTION_TRANSFER',
    label: 'Motion Transfer',
    description: 'Applies the movement sequence from a source video to animate your character.',
    suggested_prompt: 'Make my character dance like in the reference video',
    requires_input_video: true,
    supports_input_video: true,
    supports_prompt: true,
    experimental: false,
  },
  {
    value: 'wan_r2v',
    name: 'WAN_R2V',
    label: 'Reference to Video',
    description: 'Generates a new video featuring your reference character from a text prompt.',
    suggested_prompt: 'Generate a video of my character walking through a forest',
    requires_input_video: false,
    supports_input_video: false,
    supports_prompt: true,
    experimental: false,
  },
  {
    value: 'liveportrait',
    name: 'LIVEPORTRAIT',
    label: 'LivePortrait',
    description: 'Animates a portrait image using facial motion from a driving video.',
    suggested_prompt: 'Animate this portrait with the expressions from the video',
    requires_input_video: true,
    supports_input_video: true,
    supports_prompt: true,
    experimental: false,
  },
  {
    value: 'deep_live_cam',
    name: 'DEEP_LIVE_CAM',
    label: 'Face Swap',
    description: 'Replaces faces in the target video with the face from your reference image.',
    suggested_prompt: 'Swap my face with the character in the video',
    requires_input_video: true,
    supports_input_video: true,
    supports_prompt: true,
    experimental: false,
  },
];

export function getFallbackModeInfo(mode: GenerationMode): GenerationModeInfo {
  return FALLBACK_GENERATION_MODES.find((entry) => entry.value === mode) ?? FALLBACK_GENERATION_MODES[0];
}

export type JobStatus =
  | 'pending'
  | 'queued'
  | 'processing'
  | 'extracting_pose'
  | 'generating'
  | 'encoding'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface JobResponse {
  id: string;
  status: JobStatus;
  progress: number;
  current_step: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  request: GenerateRequest;
  result_url?: string;
  thumbnail_url?: string;
  error?: string;
  metadata: Record<string, unknown>;
}

export interface JobProgress {
  job_id: string;
  status: JobStatus;
  progress: number;
  step: string;
  eta_seconds?: number;
  preview_url?: string;
  error?: string;
}

export interface RealtimeConfig {
  reference_image_id: string;
  mode: 'liveportrait' | 'deep_live_cam';
  target_fps: number;
  face_only: boolean;
  smoothing: number;
  enhance_face: boolean;
  input_resolution: [number, number];
  output_resolution: [number, number];
  jpeg_quality: number;
  jpeg_subsampling: '444' | '422' | '420' | 'gray';
  binary_transport: boolean;
  full_frame_inference: boolean;
  tile_size?: number | null;
  tile_overlap: number;
  max_inflight_frames: number;
  allow_frame_drop: boolean;
  adaptive_quality?: boolean;
  adaptive_latency_budget_ms?: number | null;
  adaptive_jpeg_step?: number;
  adaptive_min_jpeg_quality?: number;
  adaptive_cooldown_frames?: number;
  adaptive_tile_size?: number | null;
  adaptive_min_tile_size?: number;
  adaptive_fps_step?: number;
  adaptive_min_target_fps?: number;
}

export interface RealtimeSession {
  session_id: string;
  websocket_url: string;
  config: RealtimeConfig;
  status: string;
  worker_id?: number;
  metrics?: RealtimeSessionMetrics;
}

export interface RealtimeSessionMetrics {
  received_frames: number;
  processed_frames: number;
  dropped_frames: number;
  bytes_in: number;
  bytes_out: number;
  avg_worker_latency_ms: number;
  last_worker_latency_ms: number;
  avg_total_latency_ms: number;
  last_total_latency_ms: number;
  avg_decode_ms: number;
  last_decode_ms: number;
  avg_inference_ms: number;
  last_inference_ms: number;
  avg_encode_ms: number;
  last_encode_ms: number;
  avg_resize_ms: number;
  last_resize_ms: number;
  avg_tile_count: number;
  last_tile_count: number;
  shared_memory_in_count: number;
  shared_memory_in_bytes: number;
  shared_memory_out_count: number;
  shared_memory_out_bytes: number;
  inline_transport_in_count: number;
  inline_transport_in_bytes: number;
  inline_transport_out_count: number;
  inline_transport_out_bytes: number;
  adaptive_adjustment_count: number;
  adaptive_events?: Array<{
    timestamp: string;
    message: string;
    jpeg_quality?: number | null;
    tile_size?: number | null;
    full_frame_inference?: boolean | null;
    target_fps?: number | null;
  }>;
  current_jpeg_quality?: number | null;
  current_tile_size?: number | null;
  current_full_frame_inference?: boolean | null;
  current_target_fps?: number | null;
  current_processing_mode?: string | null;
  worker_id?: number | null;
  last_updated_at?: string | null;
}

export interface RealtimeSessionMetricsResponse {
  session_id: string;
  status: string;
  worker_id?: number;
  config: RealtimeConfig;
  metrics: RealtimeSessionMetrics;
}

export interface RealtimeWorkerTelemetry {
  worker_id: number;
  pending_requests: number;
  inflight_queue_depth: number;
  processed_requests: number;
  error_count: number;
  avg_latency_ms: number;
  last_latency_ms: number;
  shared_memory_in_count: number;
  shared_memory_in_bytes: number;
  shared_memory_out_count: number;
  shared_memory_out_bytes: number;
  inline_transport_in_count: number;
  inline_transport_in_bytes: number;
  inline_transport_out_count: number;
  inline_transport_out_bytes: number;
  input_queue_size: number;
  output_queue_size: number;
  active_sessions: number;
  session_ids: string[];
  process_alive: boolean;
  saturation: number;
}

export interface RealtimeCompatibility {
  gpu_available: boolean;
  gpu_name?: string;
  gpu_memory_gb?: number;
  capability: string;
  estimated_fps: number;
  runtime?: 'cpu' | 'cuda' | 'mps';
  recommended_session?: {
    input_resolution: [number, number];
    output_resolution: [number, number];
    target_fps: number;
    jpeg_quality: number;
    worker_processes: number;
    full_frame_inference: boolean;
  };
  recommended_mode?: GenerationMode;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || error.detail || `Request failed: ${response.status}`);
    }

    return response.json();
  }

  // File uploads
  async uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload/image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || error.detail || 'Upload failed');
    }

    return response.json();
  }

  async uploadVideo(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload/video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || error.detail || 'Upload failed');
    }

    return response.json();
  }

  // Generation
  async generate(request: GenerateRequest): Promise<JobResponse> {
    return this.request<JobResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async generatePreview(request: GenerateRequest): Promise<JobResponse> {
    return this.request<JobResponse>('/generate/preview', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async parsePrompt(prompt: string): Promise<{
    mode: GenerationMode;
    action: string;
    subject: string;
    parameters: Record<string, unknown>;
    confidence: number;
    cleaned_prompt: string;
    mode_description: string;
  }> {
    return this.request('/generate/parse-prompt', {
      method: 'POST',
      body: JSON.stringify({ prompt }),
    });
  }

  async getModes(): Promise<{ modes: GenerationModeInfo[] }> {
    return this.request('/generate/modes');
  }

  async getQualityPresets(): Promise<{ presets: Array<{
    value: QualityPreset;
    name: string;
    description: string;
    estimated_time_factor: number;
  }> }> {
    return this.request('/generate/quality-presets');
  }

  // Jobs
  async listJobs(status?: JobStatus, limit?: number): Promise<JobResponse[]> {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (limit) params.set('limit', limit.toString());

    return this.request(`/jobs?${params}`);
  }

  async getJob(jobId: string): Promise<JobResponse> {
    return this.request(`/jobs/${jobId}`);
  }

  async getJobProgress(jobId: string): Promise<JobProgress> {
    return this.request(`/jobs/${jobId}/progress`);
  }

  async cancelJob(jobId: string): Promise<JobResponse> {
    return this.request(`/jobs/${jobId}/cancel`, {
      method: 'POST',
    });
  }

  async deleteJob(jobId: string): Promise<void> {
    return this.request(`/jobs/${jobId}`, {
      method: 'DELETE',
    });
  }

  async retryJob(jobId: string): Promise<JobResponse> {
    return this.request(`/jobs/${jobId}/retry`, {
      method: 'POST',
    });
  }

  async getJobResult(jobId: string): Promise<{
    job_id: string;
    result_url: string;
    thumbnail_url?: string;
    metadata: Record<string, unknown>;
  }> {
    return this.request(`/jobs/${jobId}/result`);
  }

  // Files
  async listFiles(fileType?: string, limit?: number): Promise<UploadResponse[]> {
    const params = new URLSearchParams();
    if (fileType) params.set('file_type', fileType);
    if (limit) params.set('limit', limit.toString());

    return this.request(`/files?${params}`);
  }

  async getFile(fileId: string): Promise<UploadResponse> {
    return this.request(`/files/${fileId}`);
  }

  async deleteFile(fileId: string): Promise<void> {
    return this.request(`/files/${fileId}`, {
      method: 'DELETE',
    });
  }

  getFileThumbnailUrl(fileId: string): string {
    return `${this.baseUrl}/files/${fileId}/thumbnail`;
  }

  getFileDownloadUrl(fileId: string): string {
    return `${this.baseUrl}/files/${fileId}/download`;
  }

  // Real-time
  async createRealtimeSession(config: RealtimeConfig): Promise<RealtimeSession> {
    return this.request('/realtime/session', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getRealtimeSession(sessionId: string): Promise<RealtimeSession> {
    return this.request(`/realtime/session/${sessionId}`);
  }

  async getRealtimeSessionMetrics(sessionId: string): Promise<RealtimeSessionMetricsResponse> {
    return this.request(`/realtime/session/${sessionId}/metrics`);
  }

  async getRealtimeWorkers(): Promise<{
    workers: RealtimeWorkerTelemetry[];
    shared_memory_enabled: boolean;
    shared_memory_threshold_bytes: number;
    worker_processes: number;
  }> {
    return this.request('/realtime/workers');
  }

  async deleteRealtimeSession(sessionId: string): Promise<void> {
    return this.request(`/realtime/session/${sessionId}`, {
      method: 'DELETE',
    });
  }

  async checkRealtimeCompatibility(): Promise<RealtimeCompatibility> {
    return this.request('/realtime/check-compatibility');
  }

  // Health
  async healthCheck(): Promise<{
    status: string;
    version: string;
    comfyui: string;
  }> {
    const response = await fetch('/health');
    return response.json();
  }
}

export const api = new ApiClient();
export default api;
