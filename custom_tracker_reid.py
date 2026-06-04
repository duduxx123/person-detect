from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from reid_osnet import PersonReIDEncoder
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.utils import matching


class ReIDDetectionEncoder:
    """Adapter that extracts ReID embeddings for BoT-SORT detections."""

    def __init__(self, weights_path, input_size=(128, 256)):
        self.reid = PersonReIDEncoder(weights_path, input_size=input_size)

    def _crop_from_xywh(self, image_bgr, det):
        cx, cy, width, height = det[:4]
        x1 = int(round(cx - width / 2.0))
        y1 = int(round(cy - height / 2.0))
        x2 = int(round(cx + width / 2.0))
        y2 = int(round(cy + height / 2.0))

        h, w = image_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_bgr[y1:y2, x1:x2]
        return crop if crop.size else None

    def inference(self, image_bgr, dets):
        features = []
        for det in dets:
            crop = self._crop_from_xywh(image_bgr, det)
            features.append(None if crop is None else self.reid.encode(crop))
        return features


def _embedding_distance_with_missing(tracks, detections):
    cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    track_indices = [i for i, track in enumerate(tracks) if getattr(track, "smooth_feat", None) is not None]
    det_indices = [i for i, det in enumerate(detections) if getattr(det, "curr_feat", None) is not None]
    if not track_indices or not det_indices:
        return cost_matrix

    track_features = np.asarray([tracks[i].smooth_feat for i in track_indices], dtype=np.float32)
    det_features = np.asarray([detections[i].curr_feat for i in det_indices], dtype=np.float32)
    distances = np.maximum(0.0, cdist(track_features, det_features, metric="cosine")).astype(np.float32)
    cost_matrix[np.ix_(track_indices, det_indices)] = distances
    return cost_matrix


def install_custom_botsort_reid(weights_path, input_size=(128, 256)):
    """Monkey-patch Ultralytics BOTSORT so it can use the local OSNet encoder."""

    weights_path = str(Path(weights_path))
    input_size = tuple(input_size)

    if getattr(BOTSORT, "_custom_reid_patched", False):
        BOTSORT._custom_reid_weights_path = weights_path
        BOTSORT._custom_reid_input_size = input_size
        return

    original_init = BOTSORT.__init__

    def patched_init(self, args, frame_rate=30):
        original_init(self, args, frame_rate)
        self.encoder = None
        if args.get("with_reid", False):
            reid_weights = args.get("reid_weights_path", weights_path)
            reid_input_width = int(args.get("reid_input_width", input_size[0]))
            reid_input_height = int(args.get("reid_input_height", input_size[1]))
            self.encoder = ReIDDetectionEncoder(
                weights_path=reid_weights,
                input_size=(reid_input_width, reid_input_height),
            )

    def patched_init_track(self, dets, scores, cls, img=None):
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None and img is not None:
            features = self.encoder.inference(img, dets)
            return [BOTrack(xywh, score, cls_id, feat) for xywh, score, cls_id, feat in zip(dets, scores, cls, features)]
        return [BOTrack(xywh, score, cls_id) for xywh, score, cls_id in zip(dets, scores, cls)]

    def patched_get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = _embedding_distance_with_missing(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    BOTSORT.__init__ = patched_init
    BOTSORT.init_track = patched_init_track
    BOTSORT.get_dists = patched_get_dists
    BOTSORT._custom_reid_patched = True
    BOTSORT._custom_reid_weights_path = weights_path
    BOTSORT._custom_reid_input_size = input_size
