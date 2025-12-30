'use client';

import { useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { toast } from 'react-hot-toast';
import { FiVideo, FiCamera, FiSettings, FiGithub } from 'react-icons/fi';

import { ImageUploader } from '@/components/ImageUploader';
import { VideoUploader } from '@/components/VideoUploader';
import { PromptInput } from '@/components/PromptInput';
import { GenerateButton } from '@/components/GenerateButton';
import { VideoPlayer } from '@/components/VideoPlayer';
import { SettingsPanel } from '@/components/SettingsPanel';

import { useGenerationStore } from '@/hooks/useVideoGeneration';
import { useJobWebSocket } from '@/hooks/useWebSocket';

export default function Home() {
  const {
    referenceImage,
    inputVideo,
    prompt,
    mode,
    quality,
    fps,
    strength,
    duration,
    currentJob,
    jobProgress,
    isGenerating,
    error,
    setPrompt,
    setMode,
    setQuality,
    setFps,
    setStrength,
    setDuration,
    uploadReferenceImage,
    uploadInputVideo,
    setReferenceImage,
    setInputVideo,
    startGeneration,
    cancelGeneration,
    updateProgress,
    clearJob,
  } = useGenerationStore();

  // WebSocket for job progress
  useJobWebSocket(currentJob?.id || null, {
    onProgress: (progress) => {
      updateProgress(progress);
    },
    onComplete: () => {
      toast.success('Video generated successfully!');
    },
    onError: (err) => {
      toast.error(err);
    },
  });

  // Show errors
  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  const handleGenerate = async () => {
    if (!referenceImage) {
      toast.error('Please upload a reference image');
      return;
    }
    await startGeneration();
  };

  const hasResult = currentJob?.status === 'completed' && currentJob?.result_url;

  return (
    <>
      <Head>
        <title>VFX Motion Capture</title>
        <meta name="description" content="Real-time VFX motion capture using AI" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen">
        {/* Header */}
        <header className="border-b border-dark-700 bg-dark-900/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FiVideo className="w-8 h-8 text-primary-500" />
                <h1 className="text-xl font-bold">VFX Motion Capture</h1>
              </div>

              <nav className="flex items-center gap-4">
                <Link
                  href="/realtime"
                  className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-dark-700 transition-colors"
                >
                  <FiCamera className="w-4 h-4" />
                  <span>Real-time</span>
                </Link>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg hover:bg-dark-700 transition-colors"
                >
                  <FiGithub className="w-5 h-5" />
                </a>
              </nav>
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left column - Inputs */}
            <div className="lg:col-span-2 space-y-6">
              {/* Upload section */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold mb-4">Inputs</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <ImageUploader
                    value={referenceImage}
                    onChange={uploadReferenceImage}
                    onClear={() => setReferenceImage(null)}
                    label="Reference Character"
                    description="The person/character to appear in the output"
                    disabled={isGenerating}
                  />

                  <VideoUploader
                    value={inputVideo}
                    onChange={uploadInputVideo}
                    onClear={() => setInputVideo(null)}
                    label="Motion Video (Optional)"
                    description="Video with motion/poses to transfer"
                    disabled={isGenerating}
                  />
                </div>
              </div>

              {/* Prompt section */}
              <div className="card p-6">
                <PromptInput
                  value={prompt}
                  onChange={setPrompt}
                  mode={mode}
                  onModeChange={setMode}
                  disabled={isGenerating}
                />
              </div>

              {/* Generate button */}
              <div className="card p-6">
                <GenerateButton
                  onClick={handleGenerate}
                  onCancel={cancelGeneration}
                  disabled={!referenceImage || isGenerating}
                  isGenerating={isGenerating}
                  progress={jobProgress?.progress || 0}
                  currentStep={jobProgress?.step}
                />
              </div>

              {/* Output section */}
              {hasResult && (
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold">Output</h2>
                    <button
                      onClick={clearJob}
                      className="text-sm text-dark-400 hover:text-white"
                    >
                      Clear
                    </button>
                  </div>

                  <VideoPlayer
                    src={currentJob.result_url!}
                    poster={currentJob.thumbnail_url}
                    onRetry={handleGenerate}
                  />

                  {currentJob.metadata && (
                    <div className="mt-4 flex gap-4 text-sm text-dark-400">
                      {currentJob.metadata.duration && (
                        <span>Duration: {(currentJob.metadata.duration as number).toFixed(1)}s</span>
                      )}
                      {currentJob.metadata.resolution && (
                        <span>
                          Resolution: {(currentJob.metadata.resolution as number[])[0]}x
                          {(currentJob.metadata.resolution as number[])[1]}
                        </span>
                      )}
                      {currentJob.metadata.mode && (
                        <span>Mode: {(currentJob.metadata.mode as string).replace(/_/g, ' ')}</span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right column - Settings */}
            <div className="space-y-6">
              <div className="card p-6">
                <SettingsPanel
                  quality={quality}
                  onQualityChange={setQuality}
                  fps={fps}
                  onFpsChange={setFps}
                  strength={strength}
                  onStrengthChange={setStrength}
                  duration={duration}
                  onDurationChange={setDuration}
                  disabled={isGenerating}
                />
              </div>

              {/* Quick tips */}
              <div className="card p-6">
                <h3 className="font-medium mb-3">Quick Tips</h3>
                <ul className="space-y-2 text-sm text-dark-400">
                  <li>Use a clear, front-facing reference image for best results</li>
                  <li>Input videos with distinct poses work better</li>
                  <li>Start with Draft quality to test your prompts</li>
                  <li>Try different modes if the auto-detected one doesnt work well</li>
                </ul>
              </div>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-dark-700 mt-16">
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-center justify-between text-sm text-dark-400">
              <p>VFX Motion Capture - Powered by Wan 2.6, LivePortrait</p>
              <p>Built with Next.js and FastAPI</p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
}
