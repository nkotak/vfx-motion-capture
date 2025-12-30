/**
 * Hook for managing video generation state
 */

import { create } from 'zustand';
import { api, GenerateRequest, JobResponse, JobProgress, UploadResponse, GenerationMode, QualityPreset } from '@/services/api';

interface GenerationState {
  // Uploads
  referenceImage: UploadResponse | null;
  inputVideo: UploadResponse | null;

  // Generation config
  prompt: string;
  mode: GenerationMode;
  quality: QualityPreset;
  duration: number | null;
  fps: number;
  strength: number;

  // Job state
  currentJob: JobResponse | null;
  jobProgress: JobProgress | null;
  isGenerating: boolean;
  error: string | null;

  // Actions
  setReferenceImage: (image: UploadResponse | null) => void;
  setInputVideo: (video: UploadResponse | null) => void;
  setPrompt: (prompt: string) => void;
  setMode: (mode: GenerationMode) => void;
  setQuality: (quality: QualityPreset) => void;
  setDuration: (duration: number | null) => void;
  setFps: (fps: number) => void;
  setStrength: (strength: number) => void;

  uploadReferenceImage: (file: File) => Promise<void>;
  uploadInputVideo: (file: File) => Promise<void>;
  startGeneration: () => Promise<void>;
  cancelGeneration: () => Promise<void>;
  updateProgress: (progress: JobProgress) => void;
  clearJob: () => void;
  reset: () => void;
}

const initialState = {
  referenceImage: null,
  inputVideo: null,
  prompt: 'Replace the person in the video with the person from the reference image',
  mode: 'auto' as GenerationMode,
  quality: 'standard' as QualityPreset,
  duration: null,
  fps: 24,
  strength: 0.85,
  currentJob: null,
  jobProgress: null,
  isGenerating: false,
  error: null,
};

export const useGenerationStore = create<GenerationState>((set, get) => ({
  ...initialState,

  setReferenceImage: (image) => set({ referenceImage: image }),
  setInputVideo: (video) => set({ inputVideo: video }),
  setPrompt: (prompt) => set({ prompt }),
  setMode: (mode) => set({ mode }),
  setQuality: (quality) => set({ quality }),
  setDuration: (duration) => set({ duration }),
  setFps: (fps) => set({ fps }),
  setStrength: (strength) => set({ strength }),

  uploadReferenceImage: async (file) => {
    set({ error: null });
    try {
      const response = await api.uploadImage(file);
      set({ referenceImage: response });
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Upload failed';
      set({ error });
      throw e;
    }
  },

  uploadInputVideo: async (file) => {
    set({ error: null });
    try {
      const response = await api.uploadVideo(file);
      set({ inputVideo: response });
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Upload failed';
      set({ error });
      throw e;
    }
  },

  startGeneration: async () => {
    const state = get();

    if (!state.referenceImage) {
      set({ error: 'Please upload a reference image' });
      return;
    }

    set({ isGenerating: true, error: null });

    try {
      const request: GenerateRequest = {
        reference_image_id: state.referenceImage.id,
        input_video_id: state.inputVideo?.id,
        prompt: state.prompt,
        mode: state.mode,
        quality: state.quality,
        fps: state.fps,
        strength: state.strength,
      };

      if (state.duration) {
        request.duration = state.duration;
      }

      const job = await api.generate(request);
      set({
        currentJob: job,
        jobProgress: {
          job_id: job.id,
          status: job.status,
          progress: job.progress,
          step: job.current_step,
        },
      });
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Generation failed';
      set({ error, isGenerating: false });
    }
  },

  cancelGeneration: async () => {
    const { currentJob } = get();
    if (!currentJob) return;

    try {
      await api.cancelJob(currentJob.id);
      set({
        isGenerating: false,
        currentJob: null,
        jobProgress: null,
      });
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Failed to cancel';
      set({ error });
    }
  },

  updateProgress: (progress) => {
    set({ jobProgress: progress });

    if (progress.status === 'completed' || progress.status === 'failed' || progress.status === 'cancelled') {
      set({ isGenerating: false });

      // Fetch the complete job to get result URL
      if (progress.status === 'completed') {
        api.getJob(progress.job_id).then((job) => {
          set({ currentJob: job });
        });
      }
    }
  },

  clearJob: () => {
    set({
      currentJob: null,
      jobProgress: null,
      isGenerating: false,
      error: null,
    });
  },

  reset: () => {
    set(initialState);
  },
}));

export function useVideoGeneration() {
  return useGenerationStore();
}
