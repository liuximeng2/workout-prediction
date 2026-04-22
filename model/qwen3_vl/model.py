"""Qwen3-VL zero-shot classifier via Ollama HTTP API.

Frames are sent as base64-encoded JPEGs in a single chat message.
The model returns a plain-text class name which is matched against label2id.
"""

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Dict, List

from PIL import Image


class Qwen3VLClassifier:
    """Zero-shot exercise classifier backed by Qwen3-VL running in Ollama."""

    def __init__(
        self,
        ollama_base_url: str,
        ollama_model: str,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        request_timeout: int = 180,
    ) -> None:
        self.url = f"{ollama_base_url.rstrip('/')}/api/chat"
        self.model = ollama_model
        self.label2id = label2id
        self.id2label = id2label
        self.timeout = request_timeout
        self._class_names = sorted(label2id.keys())
        self._prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        class_list = "\n".join(f"- {c}" for c in self._class_names)
        return (
            "/no_think\n"
            "You are an exercise classifier. The images below are frames sampled "
            "in temporal order from a gym workout video.\n\n"
            "Identify the single exercise being performed. Choose exactly one label "
            f"from this list:\n{class_list}\n\n"
            "Reply with ONLY the exact label name — no explanation, no punctuation."
        )

    @staticmethod
    def _encode_frame(img: Image.Image, size: int) -> str:
        # Resize keeping aspect ratio so short side == size
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def predict(self, frames: List[Image.Image], frame_size: int = 336) -> int:
        """Run one inference call and return the predicted class index.

        Args:
            frames:     Ordered list of PIL RGB images (one clip).
            frame_size: Short-side resize before encoding.

        Returns:
            Predicted label index (falls back to 0 on parse failure).
        """
        images_b64 = [self._encode_frame(f, frame_size) for f in frames]

        payload = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": self._prompt,
                "images": images_b64,
            }],
            "stream": False,
            "options": {"temperature": 0},
        }).encode()

        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama request failed — is the server running at {self.url}?\n{e}"
            ) from e

        raw = data["message"]["content"].strip().lower()

        # Exact match
        for name, idx in self.label2id.items():
            if raw == name.lower():
                return idx

        # Substring match (model may add minor punctuation)
        for name, idx in self.label2id.items():
            if name.lower() in raw:
                return idx

        return 0


def build_model(
    ollama_base_url: str,
    ollama_model: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    request_timeout: int = 180,
):
    """Factory function matching the registry interface.

    Returns ``(classifier, None)`` — no processor needed.
    """
    clf = Qwen3VLClassifier(
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        label2id=label2id,
        id2label=id2label,
        request_timeout=request_timeout,
    )
    return clf, None
