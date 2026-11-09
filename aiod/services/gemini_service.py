"""Google Gemini Multimodal Vision Intelligence Service for AIOD."""

import base64
import mimetypes
from pathlib import Path
from typing import Optional, Union, Dict, Any
import requests

from aiod.core.config import settings
from aiod.utils.exceptions import AIODError


class GeminiVisionService:
    """
    Integrates Google Gemini Vision API for rich scene captioning,
    contextual understanding, and deep vision analytics alongside YOLO detections.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model_name = "gemini-1.5-flash"
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    @property
    def is_configured(self) -> bool:
        """Check whether a Gemini API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def analyze_scene(
        self,
        image_path: Union[str, Path],
        user_prompt: Optional[str] = None,
        detection_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an image using Gemini Multimodal Vision.

        Args:
            image_path: Path to the image file.
            user_prompt: Custom question or prompt for Gemini.
            detection_summary: Optional YOLO detection labels to provide scene context.

        Returns:
            Dictionary with 'success', 'analysis', and 'model'.
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Gemini API key is not configured. Set GEMINI_API_KEY in .env.",
            }

        img_file = Path(image_path).resolve()
        if not img_file.exists():
            return {
                "success": False,
                "error": f"Image file not found: {img_file}",
            }

        mime_type, _ = mimetypes.guess_type(str(img_file))
        if not mime_type:
            mime_type = "image/jpeg"

        try:
            with open(img_file, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read image for Gemini: {e}",
            }

        # Build prompt
        prompt_parts = []
        if detection_summary:
            prompt_parts.append(f"Context from YOLO Object Detector: {detection_summary}\n")

        if user_prompt:
            prompt_parts.append(user_prompt)
        else:
            prompt_parts.append(
                "Analyze this image thoroughly. Describe the scene, explain the key objects, "
                "their interactions, and any notable visual insights or anomalies."
            )

        full_prompt = "\n".join(prompt_parts)

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 800,
            },
        }

        try:
            response = requests.post(
                f"{self.endpoint}?key={self.api_key}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code != 200:
                err_json = response.json() if response.content else {}
                err_msg = err_json.get("error", {}).get("message", f"HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Gemini API returned error: {err_msg}",
                    "status_code": response.status_code,
                }

            result_json = response.json()
            candidates = result_json.get("candidates", [])
            if not candidates:
                return {
                    "success": False,
                    "error": "Gemini returned no response candidates.",
                }

            text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {
                "success": True,
                "analysis": text_content.strip(),
                "model": self.model_name,
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Gemini API request timed out after 30 seconds.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Gemini request failed: {e}",
            }


# Default singleton instance
gemini_service = GeminiVisionService()

