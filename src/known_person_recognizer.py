"""
known_person_recognizer.py
==========================
Loads face encodings from known_faces/known/ at startup.
During detection, checks each YOLO-detected person crop against
the known encodings. If a match is found the person is NOT flagged
as a threat.

Folder structure expected:
    known_faces/
        known/
            img_0001.jpg   ← any image name is fine
            img_0002.jpg
            ...
            alice.jpg      ← name used as person label if no number suffix

Dependencies:
    pip install face_recognition   (installs dlib automatically)
    OR  pip install deepface        (heavier but no dlib build needed)

The module tries face_recognition first; if unavailable it falls back
to DeepFace, and if that too is missing it disables recognition
gracefully (all persons treated as unknown / potential threat).
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
KNOWN_FACES_DIR   = os.path.join(os.path.dirname(__file__), "known_faces", "known")
FACE_MATCH_TOLE   = 0.55   # lower = stricter match (face_recognition L2 distance)
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Minimum crop size to bother running recognition (pixels)
MIN_CROP_H = 40
MIN_CROP_W = 30


# ─────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────
def _load_backend():
    """
    Try to import face_recognition; fall back to DeepFace;
    fall back to NullRecognizer.
    Returns (backend_name, module_or_None)
    """
    try:
        import face_recognition as fr
        logger.info("Known-person backend: face_recognition (dlib)")
        return "face_recognition", fr
    except ImportError:
        pass

    try:
        from deepface import DeepFace
        logger.info("Known-person backend: DeepFace")
        return "deepface", DeepFace
    except ImportError:
        pass

    logger.warning(
        "Neither face_recognition nor DeepFace found. "
        "Known-person recognition DISABLED. "
        "Install with: pip install face_recognition"
    )
    return "none", None


_BACKEND_NAME, _BACKEND = _load_backend()


# ─────────────────────────────────────────────────────────────
# KnownPersonRecognizer
# ─────────────────────────────────────────────────────────────
class KnownPersonRecognizer:
    """
    Loads encodings once at init; exposes is_known_person(frame, bbox)
    which returns (True, name) or (False, "").
    """

    def __init__(self, known_dir: str = KNOWN_FACES_DIR):
        self._known_encodings: List[np.ndarray] = []
        self._known_names: List[str]             = []
        self._backend = _BACKEND_NAME
        self._enabled = _BACKEND is not None

        if self._enabled:
            self._load_known_faces(known_dir)
        else:
            logger.warning("KnownPersonRecognizer running in DISABLED mode.")

    # ── Encoding loader ───────────────────────────────────────

    def _load_known_faces(self, folder: str):
        if not os.path.isdir(folder):
            logger.warning(
                "known_faces folder not found: %s — "
                "known-person recognition will not run.", folder
            )
            self._enabled = False
            return

        images = [
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        ]

        if not images:
            logger.warning("No images found in %s.", folder)
            self._enabled = False
            return

        loaded = 0
        for img_path in sorted(images):
            name = self._name_from_path(img_path)
            enc  = self._encode_image(str(img_path))
            if enc is not None:
                self._known_encodings.append(enc)
                self._known_names.append(name)
                loaded += 1

        logger.info(
            "Loaded %d known-person encoding(s) from %s.", loaded, folder
        )
        if loaded == 0:
            self._enabled = False

    @staticmethod
    def _name_from_path(path: Path) -> str:
        """
        Derive a human-readable label from the file name.
        img_0001.jpg  → "Person 1"
        alice.jpg     → "Alice"
        """
        stem = path.stem  # e.g. "img_0001" or "alice"
        if stem.lower().startswith("img_") and stem[4:].isdigit():
            return f"Person {int(stem[4:])}"
        return stem.replace("_", " ").title()

    def _encode_image(self, path: str) -> Optional[np.ndarray]:
        """Return a face encoding for the first face in the image, or None."""
        try:
            if self._backend == "face_recognition":
                import face_recognition as fr
                import cv2
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    return None
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                encs = fr.face_encodings(img_rgb)
                return encs[0] if encs else None
            # DeepFace: store the path; verification done per query
            # (DeepFace doesn't have a simple encode-then-compare like face_recognition)
            return np.array([path], dtype=object)   # sentinel — path wrapper
        except Exception as exc:
            logger.debug("Encoding failed for %s: %s", path, exc)
            return None

    # ── Public API ────────────────────────────────────────────

    def is_known_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, str]:
        """
        Check whether the person inside bbox is a known individual.

        Parameters
        ----------
        frame : full BGR frame
        bbox  : (x1, y1, x2, y2) bounding box from YOLO

        Returns
        -------
        (True,  name)  if matched
        (False, "")    if not matched or recognition disabled
        """
        if not self._enabled or not self._known_encodings:
            return False, ""

        x1, y1, x2, y2 = bbox
        crop = frame[
            max(0, y1): max(0, y2),
            max(0, x1): max(0, x2),
        ]

        if crop.shape[0] < MIN_CROP_H or crop.shape[1] < MIN_CROP_W:
            return False, ""

        try:
            if self._backend == "face_recognition":
                return self._check_face_recognition(crop)
            elif self._backend == "deepface":
                return self._check_deepface(crop)
        except Exception as exc:
            logger.debug("Recognition error: %s", exc)

        return False, ""

    # ── face_recognition backend ──────────────────────────────

    def _check_face_recognition(
        self, crop_bgr: np.ndarray
    ) -> Tuple[bool, str]:
        import face_recognition as fr
        import cv2

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        encs = fr.face_encodings(rgb)
        if not encs:
            return False, ""

        query_enc = encs[0]
        # Compare against all known encodings
        distances = fr.face_distance(self._known_encodings, query_enc)
        best_idx  = int(np.argmin(distances))
        if distances[best_idx] <= FACE_MATCH_TOLE:
            name = self._known_names[best_idx]
            logger.info("Known person matched: %s (dist=%.3f)", name, distances[best_idx])
            return True, name

        return False, ""

    # ── DeepFace backend ──────────────────────────────────────

    def _check_deepface(self, crop_bgr: np.ndarray) -> Tuple[bool, str]:
        from deepface import DeepFace
        import cv2, tempfile, os

        # Save crop to temp file (DeepFace works with paths)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, crop_bgr)

        try:
            for enc_entry, name in zip(self._known_encodings, self._known_names):
                known_path = str(enc_entry[0])
                try:
                    result = DeepFace.verify(
                        tmp_path, known_path,
                        model_name="Facenet",
                        enforce_detection=False,
                        silent=True,
                    )
                    if result.get("verified", False):
                        logger.info("Known person matched (DeepFace): %s", name)
                        return True, name
                except Exception:
                    continue
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return False, ""

    # ── Stats ─────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def known_count(self) -> int:
        return len(self._known_encodings)
