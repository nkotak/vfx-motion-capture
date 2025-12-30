'use client';

import { useRef, useState } from 'react';
import {
  FiPlay,
  FiPause,
  FiVolume2,
  FiVolumeX,
  FiMaximize,
  FiDownload,
  FiRefreshCw,
} from 'react-icons/fi';
import clsx from 'clsx';

interface VideoPlayerProps {
  src: string;
  poster?: string;
  onRetry?: () => void;
  showDownload?: boolean;
  className?: string;
}

export function VideoPlayer({
  src,
  poster,
  onRetry,
  showDownload = true,
  className,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const toggleFullscreen = () => {
    if (videoRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        videoRef.current.requestFullscreen();
      }
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const current = videoRef.current.currentTime;
      const total = videoRef.current.duration;
      setProgress((current / total) * 100);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (videoRef.current) {
      const rect = e.currentTarget.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      videoRef.current.currentTime = pos * videoRef.current.duration;
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = 'output.mp4';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className={clsx('relative group', className)}>
      <video
        ref={videoRef}
        src={src}
        poster={poster}
        className="w-full rounded-xl bg-dark-900"
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={() => {
          if (videoRef.current) {
            setDuration(videoRef.current.duration);
          }
        }}
      />

      {/* Controls overlay */}
      <div className="absolute inset-0 flex flex-col justify-end bg-gradient-to-t from-dark-900/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-xl">
        {/* Progress bar */}
        <div
          className="h-1 bg-dark-700 mx-4 mb-2 rounded-full cursor-pointer"
          onClick={handleSeek}
        >
          <div
            className="h-full bg-primary-500 rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Control buttons */}
        <div className="flex items-center gap-3 px-4 pb-4">
          <button
            onClick={togglePlay}
            className="p-2 hover:bg-dark-700/50 rounded-lg transition-colors"
          >
            {isPlaying ? (
              <FiPause className="w-5 h-5" />
            ) : (
              <FiPlay className="w-5 h-5" />
            )}
          </button>

          <button
            onClick={toggleMute}
            className="p-2 hover:bg-dark-700/50 rounded-lg transition-colors"
          >
            {isMuted ? (
              <FiVolumeX className="w-5 h-5" />
            ) : (
              <FiVolume2 className="w-5 h-5" />
            )}
          </button>

          <span className="text-sm text-dark-300">
            {formatTime(videoRef.current?.currentTime || 0)} / {formatTime(duration)}
          </span>

          <div className="flex-1" />

          {onRetry && (
            <button
              onClick={onRetry}
              className="p-2 hover:bg-dark-700/50 rounded-lg transition-colors"
              title="Regenerate"
            >
              <FiRefreshCw className="w-5 h-5" />
            </button>
          )}

          {showDownload && (
            <button
              onClick={handleDownload}
              className="p-2 hover:bg-dark-700/50 rounded-lg transition-colors"
              title="Download"
            >
              <FiDownload className="w-5 h-5" />
            </button>
          )}

          <button
            onClick={toggleFullscreen}
            className="p-2 hover:bg-dark-700/50 rounded-lg transition-colors"
            title="Fullscreen"
          >
            <FiMaximize className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Center play button */}
      {!isPlaying && (
        <button
          onClick={togglePlay}
          className="absolute inset-0 flex items-center justify-center"
        >
          <div className="p-5 bg-dark-900/80 rounded-full hover:bg-primary-600/80 transition-colors">
            <FiPlay className="w-10 h-10 ml-1" />
          </div>
        </button>
      )}
    </div>
  );
}
