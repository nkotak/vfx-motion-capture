"""
ComfyUI API Client for executing workflows.
Handles WebSocket connections, workflow execution, and progress tracking.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable, AsyncIterator
from dataclasses import dataclass
import aiohttp
import aiofiles
from loguru import logger

from backend.core.config import settings
from backend.core.exceptions import (
    ComfyUIConnectionError,
    ComfyUIExecutionError,
)


@dataclass
class WorkflowProgress:
    """Progress information for a workflow execution."""
    prompt_id: str
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    progress: float = 0.0
    max_progress: float = 100.0
    current_step: str = ""
    preview_image: Optional[bytes] = None


@dataclass
class WorkflowResult:
    """Result from a workflow execution."""
    prompt_id: str
    success: bool
    outputs: Dict[str, Any]
    images: list[Path]
    videos: list[Path]
    error: Optional[str] = None


class ComfyUIClient:
    """
    Async client for ComfyUI API.

    Supports:
    - Executing workflows with parameter injection
    - Real-time progress tracking via WebSocket
    - Uploading images/videos to ComfyUI
    - Downloading generated outputs
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        use_ssl: bool = None,
        client_id: str = None,
    ):
        self.host = host or settings.comfyui_host
        self.port = port or settings.comfyui_port
        self.use_ssl = use_ssl if use_ssl is not None else settings.comfyui_use_ssl
        self.client_id = client_id or str(uuid.uuid4())

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}/ws?clientId={self.client_id}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=settings.comfyui_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close all connections."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def health_check(self) -> bool:
        """Check if ComfyUI is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/system_stats") as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"ComfyUI health check failed: {e}")
            return False

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get ComfyUI system statistics."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/system_stats") as resp:
            if resp.status != 200:
                raise ComfyUIConnectionError(f"Failed to get system stats: {resp.status}")
            return await resp.json()

    async def upload_image(
        self,
        image_path: Path,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> Dict[str, str]:
        """
        Upload an image to ComfyUI's input folder.

        Returns:
            Dict with 'name', 'subfolder', 'type' keys
        """
        session = await self._get_session()

        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()

        data = aiohttp.FormData()
        data.add_field(
            "image",
            image_data,
            filename=image_path.name,
            content_type="image/png"
        )
        data.add_field("subfolder", subfolder)
        data.add_field("overwrite", str(overwrite).lower())

        async with session.post(f"{self.base_url}/upload/image", data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ComfyUIConnectionError(f"Failed to upload image: {text}")
            result = await resp.json()
            logger.debug(f"Uploaded image: {result}")
            return result

    async def upload_video(
        self,
        video_path: Path,
        subfolder: str = "videos",
    ) -> Dict[str, str]:
        """
        Upload a video to ComfyUI's input folder.

        Returns:
            Dict with 'name', 'subfolder', 'type' keys
        """
        session = await self._get_session()

        async with aiofiles.open(video_path, "rb") as f:
            video_data = await f.read()

        data = aiohttp.FormData()
        data.add_field(
            "image",  # ComfyUI uses 'image' field for all uploads
            video_data,
            filename=video_path.name,
            content_type="video/mp4"
        )
        data.add_field("subfolder", subfolder)
        data.add_field("overwrite", "true")

        async with session.post(f"{self.base_url}/upload/image", data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ComfyUIConnectionError(f"Failed to upload video: {text}")
            result = await resp.json()
            logger.debug(f"Uploaded video: {result}")
            return result

    async def load_workflow(self, workflow_path: Path) -> Dict[str, Any]:
        """Load a workflow JSON file."""
        async with aiofiles.open(workflow_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    def inject_parameters(
        self,
        workflow: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Inject parameters into a workflow.

        Parameters can target specific nodes using format:
        - "node_id.input_name": value
        - Or use special keys that map to known node patterns
        """
        workflow = json.loads(json.dumps(workflow))  # Deep copy

        for key, value in parameters.items():
            if "." in key:
                # Direct node.input targeting
                node_id, input_name = key.split(".", 1)
                if node_id in workflow:
                    if "inputs" in workflow[node_id]:
                        workflow[node_id]["inputs"][input_name] = value
            else:
                # Search for nodes with matching input names
                for node_id, node in workflow.items():
                    if isinstance(node, dict) and "inputs" in node:
                        if key in node["inputs"]:
                            node["inputs"][key] = value

        return workflow

    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Queue a workflow for execution.

        Returns:
            prompt_id for tracking the execution
        """
        session = await self._get_session()

        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        if extra_data:
            payload["extra_data"] = extra_data

        async with session.post(
            f"{self.base_url}/prompt",
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ComfyUIExecutionError(f"Failed to queue prompt: {text}")
            result = await resp.json()
            prompt_id = result.get("prompt_id")
            if not prompt_id:
                raise ComfyUIExecutionError("No prompt_id in response")
            logger.info(f"Queued prompt: {prompt_id}")
            return prompt_id

    async def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get execution history for a prompt."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
            if resp.status != 200:
                return None
            history = await resp.json()
            return history.get(prompt_id)

    async def cancel_prompt(self, prompt_id: str) -> bool:
        """Cancel a running prompt."""
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/interrupt",
            json={"prompt_id": prompt_id}
        ) as resp:
            return resp.status == 200

    async def _connect_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Establish WebSocket connection."""
        async with self._ws_lock:
            if self._ws is None or self._ws.closed:
                session = await self._get_session()
                self._ws = await session.ws_connect(self.ws_url)
                logger.debug(f"WebSocket connected: {self.ws_url}")
            return self._ws

    async def execute_workflow(
        self,
        workflow: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[WorkflowProgress], None]] = None,
    ) -> WorkflowResult:
        """
        Execute a workflow and wait for completion.

        Args:
            workflow: The workflow to execute
            parameters: Parameters to inject into the workflow
            progress_callback: Optional callback for progress updates

        Returns:
            WorkflowResult with outputs and generated files
        """
        # Inject parameters if provided
        if parameters:
            workflow = self.inject_parameters(workflow, parameters)

        # Queue the prompt
        prompt_id = await self.queue_prompt(workflow)

        # Connect to WebSocket for progress
        ws = await self._connect_websocket()

        progress = WorkflowProgress(prompt_id=prompt_id)
        images: list[Path] = []
        videos: list[Path] = []

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    msg_data = data.get("data", {})

                    if msg_data.get("prompt_id") != prompt_id:
                        continue

                    if msg_type == "progress":
                        progress.progress = msg_data.get("value", 0)
                        progress.max_progress = msg_data.get("max", 100)
                        progress.node_id = msg_data.get("node")
                        if progress_callback:
                            progress_callback(progress)

                    elif msg_type == "executing":
                        node_id = msg_data.get("node")
                        if node_id is None:
                            # Execution complete
                            break
                        progress.node_id = node_id
                        progress.current_step = f"Executing node: {node_id}"
                        if progress_callback:
                            progress_callback(progress)

                    elif msg_type == "executed":
                        # Node finished executing
                        output = msg_data.get("output", {})
                        if "images" in output:
                            for img in output["images"]:
                                img_path = await self._download_output(
                                    img["filename"],
                                    img.get("subfolder", ""),
                                    img.get("type", "output")
                                )
                                images.append(img_path)
                        if "videos" in output or "gifs" in output:
                            for vid in output.get("videos", []) + output.get("gifs", []):
                                vid_path = await self._download_output(
                                    vid["filename"],
                                    vid.get("subfolder", ""),
                                    vid.get("type", "output")
                                )
                                videos.append(vid_path)

                    elif msg_type == "execution_error":
                        error_msg = msg_data.get("exception_message", "Unknown error")
                        raise ComfyUIExecutionError(
                            error_msg,
                            node_id=msg_data.get("node_id")
                        )

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ComfyUIConnectionError(f"WebSocket error: {ws.exception()}")

            # Get final history
            history = await self.get_history(prompt_id)
            outputs = {}
            if history:
                outputs = history.get("outputs", {})

            return WorkflowResult(
                prompt_id=prompt_id,
                success=True,
                outputs=outputs,
                images=images,
                videos=videos,
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowResult(
                prompt_id=prompt_id,
                success=False,
                outputs={},
                images=[],
                videos=[],
                error=str(e),
            )

    async def _download_output(
        self,
        filename: str,
        subfolder: str,
        output_type: str,
    ) -> Path:
        """Download an output file from ComfyUI."""
        session = await self._get_session()

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": output_type,
        }

        async with session.get(f"{self.base_url}/view", params=params) as resp:
            if resp.status != 200:
                raise ComfyUIConnectionError(f"Failed to download {filename}")

            # Save to output directory
            output_path = settings.output_dir / filename
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(await resp.read())

            logger.debug(f"Downloaded output: {output_path}")
            return output_path

    async def stream_progress(
        self,
        prompt_id: str,
    ) -> AsyncIterator[WorkflowProgress]:
        """
        Stream progress updates for a running workflow.

        Yields WorkflowProgress objects until execution completes.
        """
        ws = await self._connect_websocket()

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")
                msg_data = data.get("data", {})

                if msg_data.get("prompt_id") != prompt_id:
                    continue

                progress = WorkflowProgress(prompt_id=prompt_id)

                if msg_type == "progress":
                    progress.progress = msg_data.get("value", 0)
                    progress.max_progress = msg_data.get("max", 100)
                    progress.node_id = msg_data.get("node")
                    yield progress

                elif msg_type == "executing":
                    node_id = msg_data.get("node")
                    if node_id is None:
                        # Execution complete
                        progress.progress = 100
                        progress.current_step = "Complete"
                        yield progress
                        break
                    progress.node_id = node_id
                    progress.current_step = f"Executing: {node_id}"
                    yield progress

                elif msg_type == "execution_error":
                    progress.current_step = f"Error: {msg_data.get('exception_message', 'Unknown')}"
                    yield progress
                    break


# Singleton instance
_client: Optional[ComfyUIClient] = None


def get_comfyui_client() -> ComfyUIClient:
    """Get the global ComfyUI client instance."""
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client
