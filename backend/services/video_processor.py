"""
Video processing service using FFmpeg.
Handles video encoding, decoding, frame extraction, and format conversion.
"""

import asyncio
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, AsyncIterator
from dataclasses import dataclass
import ffmpeg
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from backend.core.config import settings
from backend.core.models import VideoMetadata
from backend.core.exceptions import VideoProcessingError, InvalidVideoError


@dataclass
class FrameInfo:
    """Information about an extracted frame."""
    index: int
    timestamp: float
    path: Path
    width: int
    height: int


class VideoProcessor:
    """
    Async video processing service.

    Provides methods for:
    - Video metadata extraction
    - Frame extraction and resampling
    - Video encoding with various codecs
    - Audio extraction and merging
    - Format conversion
    - Thumbnail generation
    """

    def __init__(self):
        self.temp_dir = settings.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def get_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extract metadata from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            VideoMetadata object with video information
        """
        if not video_path.exists():
            raise InvalidVideoError(f"Video file not found: {video_path}")

        try:
            probe = await asyncio.to_thread(
                ffmpeg.probe, str(video_path)
            )

            # Find video stream
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"),
                None
            )
            if not video_stream:
                raise InvalidVideoError("No video stream found")

            # Find audio stream
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None
            )

            # Calculate duration
            duration = float(probe["format"].get("duration", 0))
            if duration == 0 and "duration" in video_stream:
                duration = float(video_stream["duration"])

            # Get FPS
            fps_str = video_stream.get("r_frame_rate", "24/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 24.0
            else:
                fps = float(fps_str)

            # Get frame count
            frame_count = int(video_stream.get("nb_frames", 0))
            if frame_count == 0:
                frame_count = int(duration * fps)

            return VideoMetadata(
                path=video_path,
                duration=duration,
                fps=fps,
                frame_count=frame_count,
                width=int(video_stream["width"]),
                height=int(video_stream["height"]),
                codec=video_stream.get("codec_name", "unknown"),
                has_audio=audio_stream is not None,
                audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                file_size_bytes=int(probe["format"].get("size", 0)),
            )

        except ffmpeg.Error as e:
            raise InvalidVideoError(f"Failed to probe video: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            raise VideoProcessingError(f"Failed to get video metadata: {str(e)}")

    async def extract_frames(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
        fps: Optional[float] = None,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> List[FrameInfo]:
        """
        Extract frames from a video.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save frames (uses temp if None)
            fps: Target FPS for extraction (None = use video's FPS)
            start_time: Start time in seconds
            duration: Duration to extract in seconds
            max_frames: Maximum number of frames to extract
            resolution: Target resolution (width, height)

        Returns:
            List of FrameInfo objects for extracted frames
        """
        metadata = await self.get_metadata(video_path)

        if output_dir is None:
            output_dir = self.temp_dir / f"frames_{video_path.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        target_fps = fps or metadata.fps
        if max_frames and duration:
            target_fps = min(target_fps, max_frames / duration)

        # Build FFmpeg command
        stream = ffmpeg.input(str(video_path), ss=start_time)

        if duration:
            stream = stream.filter("trim", duration=duration)

        if fps:
            stream = stream.filter("fps", fps=target_fps)

        if resolution:
            stream = stream.filter("scale", resolution[0], resolution[1])

        output_pattern = str(output_dir / "frame_%06d.png")

        try:
            await asyncio.to_thread(
                lambda: (
                    stream
                    .output(output_pattern, start_number=0)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            )
        except ffmpeg.Error as e:
            raise VideoProcessingError(
                f"Frame extraction failed: {e.stderr.decode() if e.stderr else str(e)}"
            )

        # Collect frame info
        frames = []
        for i, frame_path in enumerate(sorted(output_dir.glob("frame_*.png"))):
            if max_frames and i >= max_frames:
                break

            img = Image.open(frame_path)
            frames.append(FrameInfo(
                index=i,
                timestamp=start_time + (i / target_fps),
                path=frame_path,
                width=img.width,
                height=img.height,
            ))

        logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames

    async def extract_frames_as_arrays(
        self,
        video_path: Path,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> AsyncIterator[Tuple[int, np.ndarray]]:
        """
        Extract frames as numpy arrays (memory efficient streaming).

        Yields:
            Tuples of (frame_index, frame_array)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise InvalidVideoError(f"Could not open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = fps or video_fps
            frame_skip = int(video_fps / target_fps) if target_fps < video_fps else 1

            frame_idx = 0
            output_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    if resolution:
                        frame = cv2.resize(frame, resolution)

                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    yield output_idx, frame
                    output_idx += 1

                    if max_frames and output_idx >= max_frames:
                        break

                frame_idx += 1

                # Yield control periodically
                if frame_idx % 30 == 0:
                    await asyncio.sleep(0)

        finally:
            cap.release()

    async def frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float = 24.0,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        audio_path: Optional[Path] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Path:
        """
        Encode frames into a video file.

        Args:
            frames_dir: Directory containing frame images
            output_path: Output video path
            fps: Output FPS
            codec: Video codec (libx264, libx265, libvpx-vp9)
            crf: Constant Rate Factor (quality, lower = better)
            preset: Encoding preset (ultrafast to veryslow)
            audio_path: Optional audio file to merge
            resolution: Optional output resolution

        Returns:
            Path to the output video
        """
        frame_pattern = str(frames_dir / "frame_%06d.png")

        # Check if frames exist
        if not list(frames_dir.glob("frame_*.png")):
            # Try other patterns
            for pattern in ["*.png", "*.jpg", "*.jpeg"]:
                files = sorted(frames_dir.glob(pattern))
                if files:
                    frame_pattern = str(frames_dir / pattern.replace("*", "%06d"))
                    break
            else:
                raise VideoProcessingError(f"No frames found in {frames_dir}")

        stream = ffmpeg.input(frame_pattern, framerate=fps)

        if resolution:
            stream = stream.filter("scale", resolution[0], resolution[1])

        output_args = {
            "vcodec": codec,
            "crf": crf,
            "preset": preset,
            "pix_fmt": "yuv420p",  # Compatibility
        }

        if audio_path and audio_path.exists():
            audio_stream = ffmpeg.input(str(audio_path))
            stream = ffmpeg.output(
                stream,
                audio_stream,
                str(output_path),
                acodec="aac",
                **output_args
            )
        else:
            stream = ffmpeg.output(stream, str(output_path), **output_args)

        try:
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise VideoProcessingError(
                f"Video encoding failed: {e.stderr.decode() if e.stderr else str(e)}"
            )

        logger.info(f"Encoded video: {output_path}")
        return output_path

    async def arrays_to_video(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: float = 24.0,
        codec: str = "libx264",
        crf: int = 18,
    ) -> Path:
        """
        Encode numpy arrays directly to video (faster, no intermediate files).

        Args:
            frames: List of RGB numpy arrays
            output_path: Output video path
            fps: Output FPS
            codec: Video codec
            crf: Quality factor

        Returns:
            Path to output video
        """
        if not frames:
            raise VideoProcessingError("No frames provided")

        height, width = frames[0].shape[:2]

        # Use FFmpeg pipe
        process = (
            ffmpeg
            .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", r=fps)
            .output(str(output_path), vcodec=codec, crf=crf, pix_fmt="yuv420p")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            await asyncio.to_thread(process.wait)
        except Exception as e:
            process.kill()
            raise VideoProcessingError(f"Failed to encode video: {e}")

        logger.info(f"Encoded {len(frames)} frames to {output_path}")
        return output_path

    async def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        format: str = "wav",
    ) -> Optional[Path]:
        """
        Extract audio from a video file.

        Returns:
            Path to extracted audio, or None if no audio
        """
        metadata = await self.get_metadata(video_path)
        if not metadata.has_audio:
            return None

        if output_path is None:
            output_path = self.temp_dir / f"{video_path.stem}_audio.{format}"

        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(stream, str(output_path), acodec="pcm_s16le" if format == "wav" else "aac")
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None

        return output_path

    async def merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        audio_offset: float = 0.0,
    ) -> Path:
        """
        Merge audio into a video file.

        Args:
            video_path: Input video path
            audio_path: Audio file path
            output_path: Output video path
            audio_offset: Audio offset in seconds

        Returns:
            Path to merged video
        """
        video_stream = ffmpeg.input(str(video_path))
        audio_stream = ffmpeg.input(str(audio_path))

        if audio_offset != 0:
            audio_stream = audio_stream.filter("adelay", f"{int(audio_offset * 1000)}|{int(audio_offset * 1000)}")

        stream = ffmpeg.output(
            video_stream,
            audio_stream,
            str(output_path),
            vcodec="copy",
            acodec="aac",
            shortest=None,
        )

        try:
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise VideoProcessingError(f"Audio merge failed: {e.stderr.decode() if e.stderr else str(e)}")

        return output_path

    async def generate_thumbnail(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        timestamp: float = 1.0,
        size: Tuple[int, int] = (320, 180),
    ) -> Path:
        """
        Generate a thumbnail from a video.

        Args:
            video_path: Input video path
            output_path: Output thumbnail path
            timestamp: Time in seconds to capture
            size: Thumbnail size (width, height)

        Returns:
            Path to generated thumbnail
        """
        if output_path is None:
            output_path = self.temp_dir / f"{video_path.stem}_thumb.jpg"

        try:
            stream = (
                ffmpeg
                .input(str(video_path), ss=timestamp)
                .filter("scale", size[0], size[1])
                .output(str(output_path), vframes=1)
            )
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            # Try at the beginning if timestamp fails
            stream = (
                ffmpeg
                .input(str(video_path))
                .filter("scale", size[0], size[1])
                .output(str(output_path), vframes=1)
            )
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )

        return output_path

    async def convert_format(
        self,
        input_path: Path,
        output_path: Path,
        codec: Optional[str] = None,
        crf: int = 23,
    ) -> Path:
        """
        Convert video to a different format.

        Args:
            input_path: Input video path
            output_path: Output video path
            codec: Video codec (auto-detected from extension if None)
            crf: Quality factor

        Returns:
            Path to converted video
        """
        # Determine codec from extension
        ext = output_path.suffix.lower()
        if codec is None:
            codec_map = {
                ".mp4": "libx264",
                ".webm": "libvpx-vp9",
                ".mov": "libx264",
                ".avi": "mpeg4",
            }
            codec = codec_map.get(ext, "libx264")

        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.output(stream, str(output_path), vcodec=codec, crf=crf, acodec="aac")

        try:
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise VideoProcessingError(f"Format conversion failed: {e.stderr.decode() if e.stderr else str(e)}")

        return output_path

    async def resize_video(
        self,
        input_path: Path,
        output_path: Path,
        width: int,
        height: int,
        maintain_aspect: bool = True,
    ) -> Path:
        """
        Resize a video to specified dimensions.
        """
        if maintain_aspect:
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        else:
            scale_filter = f"scale={width}:{height}"

        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.output(
            stream.filter("scale", width, height),
            str(output_path),
            vcodec="libx264",
            crf=18,
            acodec="copy",
        )

        try:
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise VideoProcessingError(f"Video resize failed: {e.stderr.decode() if e.stderr else str(e)}")

        return output_path

    async def trim_video(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Path:
        """
        Trim a video to a specific segment.
        """
        stream = ffmpeg.input(str(input_path), ss=start_time)

        if duration:
            stream = ffmpeg.output(stream, str(output_path), t=duration, vcodec="copy", acodec="copy")
        elif end_time:
            stream = ffmpeg.output(stream, str(output_path), to=end_time - start_time, vcodec="copy", acodec="copy")
        else:
            stream = ffmpeg.output(stream, str(output_path), vcodec="copy", acodec="copy")

        try:
            await asyncio.to_thread(
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise VideoProcessingError(f"Video trim failed: {e.stderr.decode() if e.stderr else str(e)}")

        return output_path

    def cleanup_temp_files(self, pattern: str = "*") -> int:
        """
        Clean up temporary files matching pattern.

        Returns:
            Number of files deleted
        """
        count = 0
        for file in self.temp_dir.glob(pattern):
            try:
                if file.is_file():
                    file.unlink()
                    count += 1
                elif file.is_dir():
                    import shutil
                    shutil.rmtree(file)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}")
        return count


# Singleton instance
_processor: Optional[VideoProcessor] = None


def get_video_processor() -> VideoProcessor:
    """Get the global video processor instance."""
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor
