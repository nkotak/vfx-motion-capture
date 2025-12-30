"""
Prompt parsing service for natural language instructions.
Maps user prompts to appropriate generation modes and parameters.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from backend.core.models import GenerationMode, QualityPreset


@dataclass
class ParsedPrompt:
    """Result of parsing a user prompt."""

    mode: GenerationMode
    action: str  # e.g., "replace", "animate", "transfer", "swap"
    subject: str  # What to transform (e.g., "person", "face", "body")
    parameters: Dict[str, Any]
    original_prompt: str
    cleaned_prompt: str  # Prompt without action keywords
    confidence: float  # How confident the parser is in its interpretation


class PromptIntent(str, Enum):
    """Recognized user intents."""
    REPLACE_PERSON = "replace_person"
    TRANSFER_MOTION = "transfer_motion"
    ANIMATE_PORTRAIT = "animate_portrait"
    SWAP_FACE = "swap_face"
    GENERATE_VIDEO = "generate_video"
    UNKNOWN = "unknown"


class PromptParser:
    """
    Parses natural language prompts to determine generation mode and parameters.

    Supports prompts like:
    - "Replace the person in the video with the person from the reference image"
    - "Make my character dance like in the video"
    - "Animate this portrait with my expressions"
    - "Swap my face with the character"
    - "Transfer the motion to my character"
    """

    # Action keywords mapped to intents
    ACTION_PATTERNS = {
        PromptIntent.REPLACE_PERSON: [
            r"replace\s+(the\s+)?person",
            r"replace\s+(the\s+)?character",
            r"replace\s+(the\s+)?subject",
            r"put\s+(me|myself|the\s+person|the\s+character)\s+in",
            r"insert\s+(me|myself|the\s+person|the\s+character)\s+into",
            r"swap\s+(the\s+)?person",
            r"change\s+(the\s+)?person",
            r"substitute\s+(the\s+)?person",
        ],
        PromptIntent.TRANSFER_MOTION: [
            r"transfer\s+(the\s+)?motion",
            r"transfer\s+(the\s+)?movement",
            r"transfer\s+(the\s+)?pose",
            r"copy\s+(the\s+)?motion",
            r"copy\s+(the\s+)?movement",
            r"apply\s+(the\s+)?motion",
            r"apply\s+(the\s+)?movement",
            r"match\s+(the\s+)?motion",
            r"follow\s+(the\s+)?motion",
            r"mimic\s+(the\s+)?motion",
            r"make\s+.+\s+(dance|move|walk|run|jump)\s+like",
            r"(dance|move|walk|run|jump)\s+like",
        ],
        PromptIntent.ANIMATE_PORTRAIT: [
            r"animate\s+(this\s+)?(portrait|image|photo|picture)",
            r"bring\s+(this\s+)?(portrait|image|photo|picture)\s+to\s+life",
            r"animate\s+(with\s+)?my\s+expressions?",
            r"make\s+(this\s+)?(portrait|image|photo|picture)\s+(talk|speak|move)",
            r"lip\s*sync",
            r"talking\s+head",
        ],
        PromptIntent.SWAP_FACE: [
            r"swap\s+(my\s+)?face",
            r"face\s+swap",
            r"replace\s+(my\s+)?face",
            r"put\s+(my\s+)?face",
            r"change\s+(the\s+)?face",
            r"deepfake",
        ],
        PromptIntent.GENERATE_VIDEO: [
            r"generate\s+(a\s+)?video",
            r"create\s+(a\s+)?video",
            r"make\s+(a\s+)?video",
            r"produce\s+(a\s+)?video",
        ],
    }

    # Mode mapping based on intent
    INTENT_TO_MODE = {
        PromptIntent.REPLACE_PERSON: GenerationMode.VACE_POSE_TRANSFER,
        PromptIntent.TRANSFER_MOTION: GenerationMode.VACE_MOTION_TRANSFER,
        PromptIntent.ANIMATE_PORTRAIT: GenerationMode.LIVEPORTRAIT,
        PromptIntent.SWAP_FACE: GenerationMode.DEEP_LIVE_CAM,
        PromptIntent.GENERATE_VIDEO: GenerationMode.WAN_R2V,
        PromptIntent.UNKNOWN: GenerationMode.AUTO,
    }

    # Quality keywords
    QUALITY_PATTERNS = {
        QualityPreset.DRAFT: [r"quick", r"fast", r"draft", r"preview", r"low\s*quality"],
        QualityPreset.STANDARD: [r"standard", r"normal", r"default", r"balanced"],
        QualityPreset.HIGH: [r"high\s*quality", r"hq", r"detailed", r"good\s*quality"],
        QualityPreset.ULTRA: [r"ultra", r"maximum", r"best", r"highest\s*quality", r"4k", r"cinematic"],
    }

    # Subject patterns
    SUBJECT_PATTERNS = {
        "person": [r"person", r"people", r"human", r"man", r"woman", r"guy", r"girl"],
        "face": [r"face", r"head", r"portrait"],
        "body": [r"body", r"full\s*body", r"figure"],
        "character": [r"character", r"avatar", r"subject"],
    }

    def __init__(self):
        # Compile regex patterns for efficiency
        self._compiled_actions = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.ACTION_PATTERNS.items()
        }
        self._compiled_quality = {
            quality: [re.compile(p, re.IGNORECASE) for p in patterns]
            for quality, patterns in self.QUALITY_PATTERNS.items()
        }
        self._compiled_subjects = {
            subject: [re.compile(p, re.IGNORECASE) for p in patterns]
            for subject, patterns in self.SUBJECT_PATTERNS.items()
        }

    def parse(self, prompt: str) -> ParsedPrompt:
        """
        Parse a natural language prompt.

        Args:
            prompt: User's natural language instruction

        Returns:
            ParsedPrompt with extracted information
        """
        prompt = prompt.strip()
        if not prompt:
            return ParsedPrompt(
                mode=GenerationMode.AUTO,
                action="unknown",
                subject="person",
                parameters={},
                original_prompt="",
                cleaned_prompt="",
                confidence=0.0,
            )

        # Detect intent
        intent, intent_confidence = self._detect_intent(prompt)

        # Map to generation mode
        mode = self.INTENT_TO_MODE.get(intent, GenerationMode.AUTO)

        # Detect quality preference
        quality = self._detect_quality(prompt)

        # Detect subject
        subject = self._detect_subject(prompt)

        # Extract action word
        action = self._extract_action(intent)

        # Clean prompt (remove action keywords)
        cleaned = self._clean_prompt(prompt, intent)

        # Extract any numeric parameters
        params = self._extract_parameters(prompt)
        params["quality"] = quality

        # Calculate overall confidence
        confidence = intent_confidence

        logger.debug(
            f"Parsed prompt: intent={intent}, mode={mode}, "
            f"subject={subject}, confidence={confidence:.2f}"
        )

        return ParsedPrompt(
            mode=mode,
            action=action,
            subject=subject,
            parameters=params,
            original_prompt=prompt,
            cleaned_prompt=cleaned,
            confidence=confidence,
        )

    def _detect_intent(self, prompt: str) -> Tuple[PromptIntent, float]:
        """Detect the user's intent from the prompt."""
        best_intent = PromptIntent.UNKNOWN
        best_confidence = 0.0
        best_match_length = 0

        for intent, patterns in self._compiled_actions.items():
            for pattern in patterns:
                match = pattern.search(prompt)
                if match:
                    # Confidence based on match length and position
                    match_len = len(match.group())
                    position_factor = 1.0 - (match.start() / len(prompt)) * 0.3

                    confidence = min(1.0, (match_len / len(prompt)) * 3 * position_factor)

                    # Prefer longer matches
                    if match_len > best_match_length or (
                        match_len == best_match_length and confidence > best_confidence
                    ):
                        best_intent = intent
                        best_confidence = confidence
                        best_match_length = match_len

        # If no strong match, try heuristics
        if best_confidence < 0.3:
            lower_prompt = prompt.lower()

            # Check for video-related content
            if any(word in lower_prompt for word in ["video", "clip", "footage"]):
                if any(word in lower_prompt for word in ["reference", "image", "photo"]):
                    best_intent = PromptIntent.REPLACE_PERSON
                    best_confidence = 0.5
                else:
                    best_intent = PromptIntent.GENERATE_VIDEO
                    best_confidence = 0.4

            # Check for face-related content
            elif "face" in lower_prompt:
                best_intent = PromptIntent.SWAP_FACE
                best_confidence = 0.5

            # Check for animation content
            elif any(word in lower_prompt for word in ["animate", "expression", "portrait"]):
                best_intent = PromptIntent.ANIMATE_PORTRAIT
                best_confidence = 0.5

        return best_intent, best_confidence

    def _detect_quality(self, prompt: str) -> QualityPreset:
        """Detect quality preference from prompt."""
        for quality, patterns in self._compiled_quality.items():
            for pattern in patterns:
                if pattern.search(prompt):
                    return quality
        return QualityPreset.STANDARD

    def _detect_subject(self, prompt: str) -> str:
        """Detect the subject being transformed."""
        for subject, patterns in self._compiled_subjects.items():
            for pattern in patterns:
                if pattern.search(prompt):
                    return subject
        return "person"

    def _extract_action(self, intent: PromptIntent) -> str:
        """Get action word for an intent."""
        action_map = {
            PromptIntent.REPLACE_PERSON: "replace",
            PromptIntent.TRANSFER_MOTION: "transfer",
            PromptIntent.ANIMATE_PORTRAIT: "animate",
            PromptIntent.SWAP_FACE: "swap",
            PromptIntent.GENERATE_VIDEO: "generate",
            PromptIntent.UNKNOWN: "process",
        }
        return action_map.get(intent, "process")

    def _clean_prompt(self, prompt: str, intent: PromptIntent) -> str:
        """Remove action keywords from prompt."""
        cleaned = prompt

        # Remove matched patterns
        if intent in self._compiled_actions:
            for pattern in self._compiled_actions[intent]:
                cleaned = pattern.sub("", cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract numeric and other parameters from prompt."""
        params = {}

        # Duration (e.g., "5 seconds", "10s")
        duration_match = re.search(r"(\d+)\s*(seconds?|s\b)", prompt, re.IGNORECASE)
        if duration_match:
            params["duration"] = int(duration_match.group(1))

        # FPS (e.g., "30 fps", "24fps")
        fps_match = re.search(r"(\d+)\s*fps", prompt, re.IGNORECASE)
        if fps_match:
            params["fps"] = int(fps_match.group(1))

        # Resolution (e.g., "1080p", "720p", "4k")
        res_match = re.search(r"(\d+)p\b", prompt, re.IGNORECASE)
        if res_match:
            height = int(res_match.group(1))
            width = int(height * 16 / 9)
            params["resolution"] = (width, height)

        if re.search(r"\b4k\b", prompt, re.IGNORECASE):
            params["resolution"] = (3840, 2160)

        # Strength (e.g., "strength 0.8", "80% strength")
        strength_match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*strength", prompt, re.IGNORECASE)
        if strength_match:
            value = float(strength_match.group(1))
            params["strength"] = value / 100 if value > 1 else value

        # Preserve background
        if re.search(r"preserve\s+background|keep\s+background", prompt, re.IGNORECASE):
            params["preserve_background"] = True
        elif re.search(r"replace\s+background|change\s+background|new\s+background", prompt, re.IGNORECASE):
            params["preserve_background"] = False

        return params

    def suggest_prompt(self, mode: GenerationMode) -> str:
        """Suggest a prompt template for a given mode."""
        suggestions = {
            GenerationMode.VACE_POSE_TRANSFER: (
                "Replace the person in the video with the person from the reference image"
            ),
            GenerationMode.VACE_MOTION_TRANSFER: (
                "Make my character dance like in the reference video"
            ),
            GenerationMode.LIVEPORTRAIT: (
                "Animate this portrait with my expressions from the camera"
            ),
            GenerationMode.DEEP_LIVE_CAM: (
                "Swap my face with the character in the video"
            ),
            GenerationMode.WAN_R2V: (
                "Generate a video of my character walking through a forest"
            ),
            GenerationMode.AUTO: (
                "Describe what you want to do with the reference image and video"
            ),
        }
        return suggestions.get(mode, suggestions[GenerationMode.AUTO])

    def get_mode_description(self, mode: GenerationMode) -> str:
        """Get a human-readable description of a generation mode."""
        descriptions = {
            GenerationMode.VACE_POSE_TRANSFER: (
                "Pose Transfer: Transfers the motion and poses from the input video "
                "to the reference character while preserving their appearance."
            ),
            GenerationMode.VACE_MOTION_TRANSFER: (
                "Motion Transfer: Applies the motion sequence from a reference video "
                "to animate your character with the same movements."
            ),
            GenerationMode.LIVEPORTRAIT: (
                "Portrait Animation: Animates a static portrait image using facial "
                "expressions and head movements from a driving video or camera."
            ),
            GenerationMode.DEEP_LIVE_CAM: (
                "Face Swap: Replaces faces in the target video with the face from "
                "your reference image in real-time."
            ),
            GenerationMode.WAN_R2V: (
                "Reference-to-Video: Generates a new video featuring your reference "
                "character in AI-generated scenes based on your text prompt."
            ),
            GenerationMode.AUTO: (
                "Auto Mode: Automatically selects the best generation method based "
                "on your prompt and inputs."
            ),
        }
        return descriptions.get(mode, "Unknown mode")


# Singleton instance
_parser: Optional[PromptParser] = None


def get_prompt_parser() -> PromptParser:
    """Get the global prompt parser instance."""
    global _parser
    if _parser is None:
        _parser = PromptParser()
    return _parser
