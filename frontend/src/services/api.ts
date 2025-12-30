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
}

export interface RealtimeSession {
  session_id: string;
  websocket_url: string;
  config: RealtimeConfig;
  status: string;
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
    mode_description: string;
  }> {
    return this.request('/generate/parse-prompt', {
      method: 'POST',
      body: JSON.stringify(prompt),
    });
  }

  async getModes(): Promise<{ modes: Array<{
    value: GenerationMode;
    name: string;
    description: string;
    suggested_prompt: string;
  }> }> {
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

  async deleteRealtimeSession(sessionId: string): Promise<void> {
    return this.request(`/realtime/session/${sessionId}`, {
      method: 'DELETE',
    });
  }

  async checkRealtimeCompatibility(): Promise<{
    gpu_available: boolean;
    gpu_name?: string;
    gpu_memory_gb?: number;
    capability: string;
    estimated_fps: number;
    recommended_mode?: GenerationMode;
  }> {
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
