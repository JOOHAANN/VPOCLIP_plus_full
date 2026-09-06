"""Open-set gating for realtime recognition.

The classifier always takes an arg-max over the text bank, so on its own it can
never say "I do not know". Three checks run in front of it, in this order:

  1. person presence - YOLO sees nobody, so there is nothing to recognise
  2. novelty         - entropy or prototype distance rejects unfamiliar clips
  3. motion energy   - a motionless person is either idle or lying on the floor

A check can only push the decision towards unknown, never away from it. The
accepted label is finally passed through a hysteresis filter so the displayed
state does not flicker between neighbouring windows.
"""

import math

import numpy as np
import torch

UNKNOWN_LABEL = -1
UNKNOWN_NAME = "unknown"

# Class 0 of the custom50 YOLO model is "person", and the object RS map keeps
# one channel per class, so presence can be read straight off the map.
PERSON_SLOT = 0


def person_present(object_map, person_slot=PERSON_SLOT, min_response=0.0):
    """True when the person channel of the object RS map fired.

    Accepts [O,H,W], [B,O,H,W] or [B,1,O,H,W]; an all-zero channel means the
    detector found no person.
    """

    channel = torch.as_tensor(object_map)[..., person_slot, :, :]
    return bool(channel.amax().item() > min_response)


def motion_energy(joints):
    """Mean per-joint displacement between consecutive frames of the window.

    ``joints`` is a sequence of [25,3] skeletons, one per buffered frame.
    """

    if joints is None or len(joints) < 2:
        return 0.0
    stacked = np.asarray(joints, dtype=np.float32)
    return float(np.linalg.norm(np.diff(stacked, axis=0), axis=-1).mean())


def normalized_entropy(cosine, temperature):
    """Softmax entropy over the prototype scores, rescaled to [0,1].

    The learned logit scale saturates the softmax, so the cosine scores are
    re-softmaxed at a fixed temperature instead. 1.0 means "all classes look
    equally likely", 0.0 means "one class dominates".
    """

    scores = torch.as_tensor(cosine, dtype=torch.float32).flatten()
    if scores.numel() < 2:
        return 0.0
    probs = torch.softmax(scores / max(temperature, 1e-6), dim=0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(entropy / math.log(probs.numel()))


class UnknownGate:
    """Turns raw prototype scores into an action label or ``unknown``."""

    def __init__(self, candidate_labels, config=None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.candidate_labels = [int(label) for label in candidate_labels]

        self.require_person = bool(config.get("require_person", True))
        self.person_slot = int(config.get("person_slot", PERSON_SLOT))

        novelty = config.get("novelty") or {}
        self.method = str(novelty.get("method", "entropy")).lower()
        if self.method not in {"entropy", "prototype", "zscore"}:
            raise ValueError(
                f"novelty.method must be 'entropy', 'prototype' or 'zscore', got {self.method!r}"
            )
        self.temperature = float(novelty.get("temperature", 0.1))
        self.entropy_threshold = float(novelty.get("entropy_threshold", 0.55))
        self.cosine_threshold = float(novelty.get("cosine_threshold", 0.25))

        # zscore compares the distance to the predicted class's own centroid
        # against how far that class's training samples usually sit, so one tau
        # covers classes with very different spreads.
        self.tau = float(novelty.get("tau", 3.0))
        self.centroids = None
        if self.method == "zscore":
            path = novelty.get("centroids")
            if not path:
                raise ValueError("novelty.method=zscore needs novelty.centroids")
            stats = np.load(path)
            self.centroids = {
                "classes": stats["classes"].tolist(),
                "vectors": torch.as_tensor(stats["centroids"], dtype=torch.float32),
                "mean": torch.as_tensor(stats["mean_distance"], dtype=torch.float32),
                "std": torch.as_tensor(stats["std_distance"], dtype=torch.float32),
            }

        motion = config.get("motion") or {}
        self.motion_enabled = bool(motion.get("enabled", True))
        self.static_threshold = float(motion.get("static_threshold", 0.004))
        self.static_labels = [int(label) for label in motion.get("static_labels", [52])]
        self.static_cosine_threshold = float(
            motion.get("static_cosine_threshold", self.cosine_threshold)
        )

        self.hysteresis = max(1, int(config.get("hysteresis", 3)))
        self._state = UNKNOWN_LABEL
        self._pending = None
        self._pending_count = 0

    def __call__(self, cosine, object_map=None, joints=None, embedding=None):
        """Gate one window. ``cosine`` holds one score per candidate label."""

        cosine = torch.as_tensor(cosine, dtype=torch.float32).flatten().cpu()
        if cosine.numel() != len(self.candidate_labels):
            raise ValueError(
                f"Expected {len(self.candidate_labels)} scores, got {cosine.numel()}"
            )

        # Without a skeleton stream there is nothing to measure movement on.
        has_motion = joints is not None and len(joints) >= 2

        best = int(torch.argmax(cosine))
        label = self.candidate_labels[best]
        info = {
            "raw_label": label,
            "cosine": float(cosine[best]),
            "entropy": normalized_entropy(cosine, self.temperature),
            "motion": motion_energy(joints),
            "person": True,
            "zscore": None,
            "reason": "accepted",
        }
        if self.method == "zscore":
            info["zscore"] = self._zscore(embedding, label)

        if not self.enabled:
            info["label"] = label
            return info

        if self.require_person and object_map is not None:
            info["person"] = person_present(object_map, self.person_slot)

        if not info["person"]:
            label = UNKNOWN_LABEL
            info["reason"] = "no person"
        elif self._is_novel(info):
            label = UNKNOWN_LABEL
            info["reason"] = f"novel ({self.method})"
        elif self.motion_enabled and has_motion and info["motion"] < self.static_threshold:
            label = self._static_label(cosine)
            info["reason"] = "static"

        info["label"] = self._smooth(label)
        return info

    def _is_novel(self, info):
        if self.method == "entropy":
            return info["entropy"] > self.entropy_threshold
        if self.method == "zscore":
            # z is None when the predicted class has no centroid (an unseen
            # class added after calibration); nothing to compare against.
            return info["zscore"] is not None and info["zscore"] > self.tau
        return info["cosine"] < self.cosine_threshold

    def _zscore(self, embedding, label):
        """How many class-internal standard deviations this clip sits from C_k."""

        if self.centroids is None or embedding is None:
            return None
        try:
            index = self.centroids["classes"].index(int(label))
        except ValueError:
            return None
        embedding = torch.as_tensor(embedding, dtype=torch.float32).flatten().cpu()
        embedding = embedding / embedding.norm().clamp_min(1e-12)
        distance = 1.0 - float(embedding @ self.centroids["vectors"][index])
        return (distance - float(self.centroids["mean"][index])) / float(self.centroids["std"][index])

    def _static_label(self, cosine):
        """A motionless person is either unknown or one of the static actions.

        Everything else needs movement to happen, so the candidate set shrinks
        to ``static_labels`` (fallen on the floor by default) plus unknown.
        """

        best_label = UNKNOWN_LABEL
        best_score = -1.0
        for label in self.static_labels:
            if label not in self.candidate_labels:
                continue
            score = float(cosine[self.candidate_labels.index(label)])
            if score > best_score:
                best_label, best_score = label, score

        if best_score < self.static_cosine_threshold:
            return UNKNOWN_LABEL
        return best_label

    def _smooth(self, label):
        """Switch state only after ``hysteresis`` consecutive agreeing windows."""

        if label == self._state:
            self._pending, self._pending_count = None, 0
            return self._state

        if label == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = label, 1

        if self._pending_count >= self.hysteresis:
            self._state, self._pending, self._pending_count = label, None, 0
        return self._state


def label_name(label, names=None):
    if label == UNKNOWN_LABEL:
        return UNKNOWN_NAME
    if names and int(label) in names:
        return names[int(label)]
    return f"class {int(label)}"
