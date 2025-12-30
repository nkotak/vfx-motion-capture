"""
File management service for handling uploads and outputs.
"""

import asyncio
import uuid
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import aiofiles
import aiofiles.os
from loguru import logger

from backend.core.config import settings
from backend.core.models import UploadResponse, VideoMetadata
from backend.core.exceptions import InvalidVideoError, InvalidImageError, FileNotFoundError
from backend.services.video_processor import VideoProcessor, get_video_processor


@dataclass
class FileInfo:
    """Information about an uploaded file."""

    id: str
    filename: str
    original_filename: str
    file_type: str  # "image" or "video"
    path: Path
    size_bytes: int
    content_hash: str
    uploaded_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    thumbnail_path: Optional[Path] = None
    expires_at: Optional[datetime] = None


class FileManager:
    """
    Manages file uploads, storage, and cleanup.

    Features:
    - Secure file upload with validation
    - Thumbnail generation
    - Content hashing for deduplication
    - Automatic cleanup of expired files
    - Metadata extraction for videos
    """

    def __init__(
        self,
        upload_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        temp_dir: Optional[Path] = None,
        max_file_age_hours: int = 24,
    ):
        self.upload_dir = upload_dir or settings.upload_dir
        self.output_dir = output_dir or settings.output_dir
        self.temp_dir = temp_dir or settings.temp_dir
        self.max_file_age = timedelta(hours=max_file_age_hours)

        # In-memory file index (in production, use a database)
        self._files: Dict[str, FileInfo] = {}

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories."""
        for dir_path in [
            self.upload_dir,
            self.upload_dir / "images",
            self.upload_dir / "videos",
            self.output_dir,
            self.temp_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _generate_file_id(self) -> str:
        """Generate a unique file ID."""
        return str(uuid.uuid4())

    async def _compute_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_file_type(self, filename: str) -> str:
        """Determine file type from extension."""
        ext = Path(filename).suffix.lower()
        if ext in settings.supported_image_formats:
            return "image"
        elif ext in settings.supported_video_formats:
            return "video"
        else:
            raise InvalidVideoError(f"Unsupported file format: {ext}")

    async def save_upload(
        self,
        content: bytes,
        filename: str,
        generate_thumbnail: bool = True,
    ) -> FileInfo:
        """
        Save an uploaded file.

        Args:
            content: File content as bytes
            filename: Original filename
            generate_thumbnail: Whether to generate a thumbnail

        Returns:
            FileInfo object with file details
        """
        # Validate file type
        file_type = self._get_file_type(filename)

        # Validate size
        size_mb = len(content) / (1024 * 1024)
        max_size = settings.max_video_size_mb if file_type == "video" else settings.max_image_size_mb
        if size_mb > max_size:
            raise InvalidVideoError(f"File too large: {size_mb:.1f}MB (max: {max_size}MB)")

        # Generate file ID and path
        file_id = self._generate_file_id()
        ext = Path(filename).suffix.lower()
        safe_filename = f"{file_id}{ext}"

        # Determine storage directory
        subdir = "videos" if file_type == "video" else "images"
        file_path = self.upload_dir / subdir / safe_filename

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        # Compute hash
        content_hash = await self._compute_hash(file_path)

        # Check for duplicate
        existing = await self._find_by_hash(content_hash)
        if existing:
            # Remove new file and return existing
            await aiofiles.os.remove(file_path)
            logger.debug(f"Duplicate file detected, returning existing: {existing.id}")
            return existing

        # Create file info
        file_info = FileInfo(
            id=file_id,
            filename=safe_filename,
            original_filename=filename,
            file_type=file_type,
            path=file_path,
            size_bytes=len(content),
            content_hash=content_hash,
            uploaded_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.max_file_age,
        )

        # Extract metadata
        if file_type == "video":
            try:
                processor = get_video_processor()
                metadata = await processor.get_metadata(file_path)
                file_info.metadata = {
                    "duration": metadata.duration,
                    "fps": metadata.fps,
                    "width": metadata.width,
                    "height": metadata.height,
                    "codec": metadata.codec,
                    "has_audio": metadata.has_audio,
                }

                # Validate duration
                if metadata.duration > settings.max_video_duration:
                    await aiofiles.os.remove(file_path)
                    raise InvalidVideoError(
                        f"Video too long: {metadata.duration:.1f}s (max: {settings.max_video_duration}s)"
                    )

            except InvalidVideoError:
                raise
            except Exception as e:
                logger.warning(f"Failed to extract video metadata: {e}")

        elif file_type == "image":
            try:
                from PIL import Image
                img = Image.open(file_path)
                file_info.metadata = {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }
            except Exception as e:
                logger.warning(f"Failed to extract image metadata: {e}")

        # Generate thumbnail
        if generate_thumbnail:
            try:
                thumb_path = await self._generate_thumbnail(file_path, file_type)
                file_info.thumbnail_path = thumb_path
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail: {e}")

        # Store in index
        self._files[file_id] = file_info

        logger.info(f"Saved upload: {file_id} ({file_type}, {size_mb:.1f}MB)")
        return file_info

    async def _generate_thumbnail(
        self,
        file_path: Path,
        file_type: str,
        size: tuple = (320, 180),
    ) -> Path:
        """Generate a thumbnail for a file."""
        thumb_filename = f"{file_path.stem}_thumb.jpg"
        thumb_path = self.upload_dir / "thumbnails" / thumb_filename
        thumb_path.parent.mkdir(parents=True, exist_ok=True)

        if file_type == "video":
            processor = get_video_processor()
            await processor.generate_thumbnail(file_path, thumb_path, size=size)
        else:
            # Image thumbnail
            from PIL import Image
            img = Image.open(file_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumb_path, "JPEG", quality=85)

        return thumb_path

    async def _find_by_hash(self, content_hash: str) -> Optional[FileInfo]:
        """Find an existing file by content hash."""
        for file_info in self._files.values():
            if file_info.content_hash == content_hash:
                return file_info
        return None

    def get_file(self, file_id: str) -> FileInfo:
        """
        Get file info by ID.

        Raises FileNotFoundError if not found.
        """
        if file_id not in self._files:
            raise FileNotFoundError(file_id)
        return self._files[file_id]

    def get_file_path(self, file_id: str) -> Path:
        """Get the path to a file by ID."""
        file_info = self.get_file(file_id)
        return file_info.path

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by ID.

        Returns True if deleted, False if not found.
        """
        if file_id not in self._files:
            return False

        file_info = self._files[file_id]

        # Delete main file
        if file_info.path.exists():
            await aiofiles.os.remove(file_info.path)

        # Delete thumbnail
        if file_info.thumbnail_path and file_info.thumbnail_path.exists():
            await aiofiles.os.remove(file_info.thumbnail_path)

        del self._files[file_id]
        logger.debug(f"Deleted file: {file_id}")
        return True

    async def save_output(
        self,
        content: bytes,
        filename: str,
        job_id: str,
    ) -> Path:
        """
        Save a generated output file.

        Args:
            content: File content
            filename: Output filename
            job_id: Associated job ID

        Returns:
            Path to saved file
        """
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        output_path = job_dir / filename

        async with aiofiles.open(output_path, "wb") as f:
            await f.write(content)

        logger.debug(f"Saved output: {output_path}")
        return output_path

    def get_output_path(self, job_id: str, filename: str = None) -> Path:
        """Get path to an output file or directory."""
        job_dir = self.output_dir / job_id
        if filename:
            return job_dir / filename
        return job_dir

    async def cleanup_expired(self) -> int:
        """
        Clean up expired files.

        Returns:
            Number of files deleted
        """
        now = datetime.utcnow()
        expired = [
            file_id
            for file_id, info in self._files.items()
            if info.expires_at and info.expires_at < now
        ]

        for file_id in expired:
            await self.delete_file(file_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired files")

        return len(expired)

    async def cleanup_temp(self) -> int:
        """Clean up temporary files."""
        count = 0
        for item in self.temp_dir.iterdir():
            try:
                if item.is_file():
                    await aiofiles.os.remove(item)
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to delete temp item {item}: {e}")
        return count

    def to_upload_response(self, file_info: FileInfo) -> UploadResponse:
        """Convert FileInfo to UploadResponse for API."""
        return UploadResponse(
            id=file_info.id,
            filename=file_info.original_filename,
            file_type=file_info.file_type,
            size_bytes=file_info.size_bytes,
            duration=file_info.metadata.get("duration"),
            resolution=(
                file_info.metadata.get("width"),
                file_info.metadata.get("height")
            ) if file_info.metadata.get("width") else None,
            fps=file_info.metadata.get("fps"),
            thumbnail_url=f"/api/files/{file_info.id}/thumbnail" if file_info.thumbnail_path else None,
        )

    def list_files(
        self,
        file_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[FileInfo]:
        """List uploaded files."""
        files = list(self._files.values())

        if file_type:
            files = [f for f in files if f.file_type == file_type]

        # Sort by upload time (newest first)
        files.sort(key=lambda f: f.uploaded_at, reverse=True)

        return files[:limit]


# Singleton instance
_file_manager: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """Get the global file manager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager
