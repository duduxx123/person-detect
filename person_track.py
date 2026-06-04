import argparse
import math
import os
import time
import uuid

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath("./.ultralytics"))

from custom_tracker_reid import install_custom_botsort_reid
from reid_osnet import PersonReIDEncoder
from ultralytics import YOLO


MODEL_PATH = "./yolo11n.pt"
TRACKER_PATH = "./custom_person_botsort.yaml"
REID_WEIGHTS_PATH = (
    "./osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)

# 下面这些阈值控制“旧身份能保留多久、什么时候允许认回、认回时更看重什么”。
# 如果你后面要调效果，优先从这组参数入手。
LOST_UUID_TTL_SECONDS = 12.0
IDENTITY_BANK_TTL_SECONDS = 90.0
MIN_REASSIGN_IOU = 0.05
BASE_REASSIGN_CENTER_DISTANCE = 80.0
MAX_REASSIGN_CENTER_DISTANCE = 260.0
REID_SIMILARITY_THRESHOLD = 0.75
RECENT_REENTRY_REID_THRESHOLD = 0.72
REID_STRONG_MATCH_THRESHOLD = 0.84
REID_VERY_STRONG_MATCH_THRESHOLD = 0.90
STRONG_REID_ONLY_MAX_AGE_SECONDS = 3.0
RECENT_REENTRY_MAX_AGE_SECONDS = 1.8
RECENT_REENTRY_MIN_IOU = 0.18
RECENT_REENTRY_MAX_DISTANCE = 140.0
MIN_CANDIDATE_MARGIN = 0.035
IDENTITY_BANK_MATCH_MAX_AGE_SECONDS = 30.0
IDENTITY_BANK_REID_THRESHOLD = 0.88
IDENTITY_BANK_VERY_STRONG_THRESHOLD = 0.93
REID_EMBEDDING_BLEND = 0.85
UUID_GALLERY_SIZE = 6
MIN_REID_BOX_WIDTH = 24
MIN_REID_BOX_HEIGHT = 48
MIN_REID_UPDATE_CONF = 0.35
MIN_REID_BOX_ASPECT_RATIO = 0.22
MAX_REID_BOX_ASPECT_RATIO = 1.05
MIN_REID_SHARPNESS = 18.0
MIN_REID_QUALITY_SCORE = 0.45
IDENTITY_BANK_MIN_QUALITY_SCORE = 0.58
MOTION_SPEED_PIXELS_PER_SECOND = 160.0
DEBUG_TRACK_EVENTS = True


def bbox_iou(box_a, box_b):
    # 计算两个检测框的 IoU，用来衡量“新出现的位置”和“之前消失的位置”是否重合。
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def center_of(box):
    # 返回检测框中心点，后面会用来估计位移和速度。
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(box_a, box_b):
    # 当两个框没有重叠时，用中心点距离补充判断空间接近程度。
    ax, ay = center_of(box_a)
    bx, by = center_of(box_b)
    return math.hypot(ax - bx, ay - by)


def cosine_similarity(embedding_a, embedding_b):
    # ReID 特征已经做过 L2 归一化，直接点积就是余弦相似度。
    if embedding_a is None or embedding_b is None:
        return -1.0
    return float(np.dot(embedding_a, embedding_b))


def blend_embeddings(previous_embedding, current_embedding):
    # 对同一条轨迹的特征做指数平滑，减少某一帧模糊、遮挡带来的抖动。
    if current_embedding is None:
        return previous_embedding
    if previous_embedding is None:
        return current_embedding
    blended = REID_EMBEDDING_BLEND * previous_embedding + (1.0 - REID_EMBEDDING_BLEND) * current_embedding
    norm = np.linalg.norm(blended)
    if norm <= 0:
        return current_embedding
    return blended / norm


def clamp_box(box, width, height):
    # 把检测框限制在图像范围内，避免后续裁剪 person_crop 越界。
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compute_reid_quality(conf, box, crop):
    # 用一个简单质量分数过滤掉小框、扁框、模糊框和疑似遮挡帧，
    # 尽量避免把“坏特征”写进 gallery / identity_bank。
    width = box[2] - box[0]
    height = box[3] - box[1]
    aspect_ratio = width / max(height, 1)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    conf_score = float(np.clip((conf - MIN_REID_UPDATE_CONF) / 0.35, 0.0, 1.0))
    width_score = float(np.clip(width / 80.0, 0.0, 1.0))
    height_score = float(np.clip(height / 160.0, 0.0, 1.0))
    sharpness_score = float(np.clip(sharpness / 80.0, 0.0, 1.0))
    aspect_score = 1.0 if MIN_REID_BOX_ASPECT_RATIO <= aspect_ratio <= MAX_REID_BOX_ASPECT_RATIO else 0.25
    quality_score = (
        conf_score * 0.25
        + width_score * 0.2
        + height_score * 0.2
        + sharpness_score * 0.2
        + aspect_score * 0.15
    )

    return {
        "quality_score": quality_score,
        "sharpness": sharpness,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "usable": (
            conf >= MIN_REID_UPDATE_CONF
            and width >= MIN_REID_BOX_WIDTH
            and height >= MIN_REID_BOX_HEIGHT
            and MIN_REID_BOX_ASPECT_RATIO <= aspect_ratio <= MAX_REID_BOX_ASPECT_RATIO
            and sharpness >= MIN_REID_SHARPNESS
            and quality_score >= MIN_REID_QUALITY_SCORE
        ),
    }


def should_use_embedding_update(quality_info):
    return quality_info["usable"]


def append_to_gallery(gallery, embedding):
    # 每个 UUID 维护一个小型 gallery，而不是只存最后一帧特征。
    # 这样人物姿态、朝向变化后，仍然有机会在历史特征里找到接近样本。
    if embedding is None:
        return list(gallery or [])

    updated = list(gallery or [])
    if not updated:
        return [embedding]

    if max(cosine_similarity(embedding, saved) for saved in updated) < 0.995:
        updated.append(embedding)
    else:
        updated[-1] = embedding

    if len(updated) > UUID_GALLERY_SIZE:
        updated = updated[-UUID_GALLERY_SIZE:]
    return updated


def best_gallery_similarity(embedding, gallery, fallback_embedding=None):
    # 用当前人像特征和历史 gallery 逐个比较，得到最大相似度和平均相似度。
    # 最大值决定“像不像同一个人”，平均值用于让打分更稳一点。
    candidates = list(gallery or [])
    if fallback_embedding is not None:
        candidates.append(fallback_embedding)
    if embedding is None or not candidates:
        return -1.0, -1.0

    similarities = [cosine_similarity(embedding, candidate) for candidate in candidates if candidate is not None]
    if not similarities:
        return -1.0, -1.0
    return max(similarities), float(sum(similarities) / len(similarities))


def clone_track_info(info):
    return {
        "bbox": list(info["bbox"]),
        "last_seen": info["last_seen"],
        "embedding": info["embedding"],
        "gallery": list(info.get("gallery", [])),
        "velocity": info.get("velocity", (0.0, 0.0)),
        "quality_score": info.get("quality_score", 0.0),
    }


def adaptive_center_distance_limit(current_box, previous_info, now_ts):
    # 允许的空间偏移不是固定像素，而是会随着目标尺寸、丢失时长、估计速度动态放宽。
    # 这样短时遮挡时限制更严格，离开久一点时又不会过于死板。
    prev_box = previous_info["bbox"]
    prev_width = prev_box[2] - prev_box[0]
    prev_height = prev_box[3] - prev_box[1]
    curr_width = current_box[2] - current_box[0]
    curr_height = current_box[3] - current_box[1]

    base = max(prev_width, prev_height, curr_width, curr_height, BASE_REASSIGN_CENTER_DISTANCE)
    elapsed = max(now_ts - previous_info["last_seen"], 0.0)
    velocity = previous_info.get("velocity", (0.0, 0.0))
    predicted_motion = math.hypot(*velocity) * elapsed
    relaxed_limit = base + predicted_motion + elapsed * MOTION_SPEED_PIXELS_PER_SECOND
    return min(relaxed_limit, MAX_REASSIGN_CENTER_DISTANCE)


def _match_from_candidates(current_box, current_embedding, candidates, now_ts, active_uuids, source_name):
    best_uuid = None
    best_info = None
    best_score = float("-inf")
    best_candidate_debug = None
    candidate_pool = []

    for person_uuid, info in list(candidates.items()):
        if person_uuid in active_uuids:
            continue

        elapsed = now_ts - info["last_seen"]
        if source_name == "identity_bank" and elapsed > IDENTITY_BANK_MATCH_MAX_AGE_SECONDS:
            continue

        iou = bbox_iou(current_box, info["bbox"])
        distance = center_distance(current_box, info["bbox"])
        max_similarity, mean_similarity = best_gallery_similarity(
            current_embedding,
            info.get("gallery"),
            fallback_embedding=info.get("embedding"),
        )
        center_limit = adaptive_center_distance_limit(current_box, info, now_ts)
        spatial_ok = iou >= MIN_REASSIGN_IOU or distance <= center_limit
        strong_appearance_ok = max_similarity >= REID_STRONG_MATCH_THRESHOLD
        very_strong_appearance_ok = max_similarity >= REID_VERY_STRONG_MATCH_THRESHOLD
        appearance_ok = max_similarity >= REID_SIMILARITY_THRESHOLD
        far_reid_ok = elapsed <= STRONG_REID_ONLY_MAX_AGE_SECONDS and very_strong_appearance_ok
        recent_reentry_ok = (
            elapsed <= RECENT_REENTRY_MAX_AGE_SECONDS
            and max_similarity >= RECENT_REENTRY_REID_THRESHOLD
            and (iou >= RECENT_REENTRY_MIN_IOU or distance <= min(center_limit, RECENT_REENTRY_MAX_DISTANCE))
        )

        candidate_pool.append(
            {
                "uuid": person_uuid,
                "max_similarity": max_similarity,
                "elapsed": elapsed,
                "iou": iou,
                "distance": distance,
                "quality_score": info.get("quality_score", 0.0),
            }
        )

        candidate_debug = {
            "source": source_name,
            "candidate_uuid": person_uuid,
            "elapsed": elapsed,
            "iou": iou,
            "distance": distance,
            "center_limit": center_limit,
            "max_similarity": max_similarity,
            "mean_similarity": mean_similarity,
            "spatial_ok": spatial_ok,
            "appearance_ok": appearance_ok,
            "strong_appearance_ok": strong_appearance_ok,
            "very_strong_appearance_ok": very_strong_appearance_ok,
            "accepted": False,
            "reason": "rejected",
            "score": None,
            "margin": None,
        }

        accepted = False
        reason = "weak_reid_and_spatial"
        if source_name == "lost_tracks":
            if far_reid_ok:
                accepted = True
                reason = "far_reentry_strong_reid"
            elif recent_reentry_ok:
                accepted = True
                reason = "recent_reentry_rescue"
            elif strong_appearance_ok and spatial_ok:
                accepted = True
                reason = "strong_reid+spatial"
            elif appearance_ok and spatial_ok:
                accepted = True
                reason = "reid+spatial"
            elif appearance_ok and not spatial_ok:
                reason = "appearance_only_rejected_by_spatial"
            elif spatial_ok and not appearance_ok:
                reason = "spatial_only_rejected_by_reid"
        else:
            if elapsed <= STRONG_REID_ONLY_MAX_AGE_SECONDS and max_similarity >= IDENTITY_BANK_VERY_STRONG_THRESHOLD:
                accepted = True
                reason = "identity_bank_strong_reid"
            elif max_similarity >= IDENTITY_BANK_REID_THRESHOLD and spatial_ok:
                accepted = True
                reason = "identity_bank_reid+spatial"

        candidate_debug["reason"] = reason
        candidate_debug["_accepted_pre_margin"] = accepted
        if best_candidate_debug is None or candidate_debug["max_similarity"] > best_candidate_debug["max_similarity"]:
            best_candidate_debug = candidate_debug

    if candidate_pool:
        candidate_pool.sort(key=lambda item: item["max_similarity"], reverse=True)
        top_similarity = candidate_pool[0]["max_similarity"]
        second_similarity = candidate_pool[1]["max_similarity"] if len(candidate_pool) > 1 else None
        candidate_margin = top_similarity - second_similarity if second_similarity is not None else None
    else:
        top_similarity = None
        candidate_margin = None

    for person_uuid, info in list(candidates.items()):
        if person_uuid in active_uuids:
            continue

        elapsed = now_ts - info["last_seen"]
        if source_name == "identity_bank" and elapsed > IDENTITY_BANK_MATCH_MAX_AGE_SECONDS:
            continue

        iou = bbox_iou(current_box, info["bbox"])
        distance = center_distance(current_box, info["bbox"])
        max_similarity, mean_similarity = best_gallery_similarity(
            current_embedding,
            info.get("gallery"),
            fallback_embedding=info.get("embedding"),
        )
        center_limit = adaptive_center_distance_limit(current_box, info, now_ts)
        spatial_ok = iou >= MIN_REASSIGN_IOU or distance <= center_limit
        strong_appearance_ok = max_similarity >= REID_STRONG_MATCH_THRESHOLD
        very_strong_appearance_ok = max_similarity >= REID_VERY_STRONG_MATCH_THRESHOLD
        far_reid_ok = elapsed <= STRONG_REID_ONLY_MAX_AGE_SECONDS and very_strong_appearance_ok
        recent_reentry_ok = (
            elapsed <= RECENT_REENTRY_MAX_AGE_SECONDS
            and max_similarity >= RECENT_REENTRY_REID_THRESHOLD
            and (iou >= RECENT_REENTRY_MIN_IOU or distance <= min(center_limit, RECENT_REENTRY_MAX_DISTANCE))
        )

        accepted = False
        reason = "weak_reid_and_spatial"
        if source_name == "lost_tracks":
            if far_reid_ok:
                accepted = True
                reason = "far_reentry_strong_reid"
            elif recent_reentry_ok:
                accepted = True
                reason = "recent_reentry_rescue"
            elif strong_appearance_ok and spatial_ok:
                accepted = True
                reason = "strong_reid+spatial"
            elif max_similarity >= REID_SIMILARITY_THRESHOLD and spatial_ok:
                accepted = True
                reason = "reid+spatial"
            elif max_similarity >= REID_SIMILARITY_THRESHOLD and not spatial_ok:
                reason = "appearance_only_rejected_by_spatial"
            elif spatial_ok and max_similarity < REID_SIMILARITY_THRESHOLD:
                reason = "spatial_only_rejected_by_reid"
        else:
            if elapsed <= STRONG_REID_ONLY_MAX_AGE_SECONDS and max_similarity >= IDENTITY_BANK_VERY_STRONG_THRESHOLD:
                accepted = True
                reason = "identity_bank_strong_reid"
            elif max_similarity >= IDENTITY_BANK_REID_THRESHOLD and spatial_ok:
                accepted = True
                reason = "identity_bank_reid+spatial"

        if candidate_margin is not None and max_similarity == top_similarity:
            if candidate_margin < MIN_CANDIDATE_MARGIN and max_similarity < IDENTITY_BANK_VERY_STRONG_THRESHOLD:
                accepted = False
                reason = "ambiguous_top_candidate"

        if not accepted:
            if best_candidate_debug is None or max_similarity > best_candidate_debug["max_similarity"]:
                best_candidate_debug = {
                    "source": source_name,
                    "candidate_uuid": person_uuid,
                    "elapsed": elapsed,
                    "iou": iou,
                    "distance": distance,
                    "center_limit": center_limit,
                    "max_similarity": max_similarity,
                    "mean_similarity": mean_similarity,
                    "spatial_ok": spatial_ok,
                    "appearance_ok": max_similarity >= REID_SIMILARITY_THRESHOLD,
                    "strong_appearance_ok": strong_appearance_ok,
                    "very_strong_appearance_ok": very_strong_appearance_ok,
                    "accepted": False,
                    "reason": reason,
                    "score": None,
                    "margin": candidate_margin,
                }
            continue

        spatial_score = max(0.0, 1.0 - distance / max(center_limit, 1.0))
        time_penalty = min(elapsed / max(IDENTITY_BANK_TTL_SECONDS, LOST_UUID_TTL_SECONDS), 1.0)
        score = max_similarity * 0.7
        score += mean_similarity * 0.1
        score += iou * 0.08
        score += spatial_score * 0.08
        score += info.get("quality_score", 0.0) * 0.08
        score -= time_penalty * 0.04
        if far_reid_ok or recent_reentry_ok:
            score += 0.05
        score += max(candidate_margin or 0.0, 0.0) * 0.1

        candidate_debug = {
            "source": source_name,
            "candidate_uuid": person_uuid,
            "elapsed": elapsed,
            "iou": iou,
            "distance": distance,
            "center_limit": center_limit,
            "max_similarity": max_similarity,
            "mean_similarity": mean_similarity,
            "spatial_ok": spatial_ok,
            "appearance_ok": max_similarity >= REID_SIMILARITY_THRESHOLD,
            "strong_appearance_ok": strong_appearance_ok,
            "very_strong_appearance_ok": very_strong_appearance_ok,
            "accepted": True,
            "reason": reason,
            "score": score,
            "margin": candidate_margin,
        }

        if score > best_score:
            best_uuid = person_uuid
            best_info = info
            best_score = score
            best_candidate_debug = candidate_debug

    return best_uuid, best_info, best_candidate_debug


def match_lost_uuid(current_box, current_embedding, lost_tracks, identity_bank, now_ts, active_uuids):
    # 当底层 tracker 给了一个“新 track_id”时，这里会尝试从最近丢失的人里找回旧 UUID。
    # 核心依据是：ReID 外观相似度 + 空间接近程度 + 丢失时间惩罚。
    for person_uuid, info in list(lost_tracks.items()):
        if now_ts - info["last_seen"] > LOST_UUID_TTL_SECONDS:
            lost_tracks.pop(person_uuid, None)

    for person_uuid, info in list(identity_bank.items()):
        if now_ts - info["last_seen"] > IDENTITY_BANK_TTL_SECONDS:
            identity_bank.pop(person_uuid, None)

    best_uuid, best_info, best_candidate_debug = _match_from_candidates(
        current_box,
        current_embedding,
        lost_tracks,
        now_ts,
        active_uuids,
        source_name="lost_tracks",
    )
    if best_uuid is not None:
        lost_tracks.pop(best_uuid, None)
        return best_uuid, best_info, best_candidate_debug

    return _match_from_candidates(
        current_box,
        current_embedding,
        identity_bank,
        now_ts,
        active_uuids,
        source_name="identity_bank",
    )


def estimate_velocity(previous_box, current_box, dt):
    # 用前后两次中心点位置估计一个简单速度，供“离开后可能走到哪里”做粗预测。
    if dt <= 0:
        return (0.0, 0.0)
    previous_center = center_of(previous_box)
    current_center = center_of(current_box)
    return ((current_center[0] - previous_center[0]) / dt, (current_center[1] - previous_center[1]) / dt)


def log_track_event(message):
    # 统一控制调试日志输出，后面如果你不想看日志，改这里即可。
    if DEBUG_TRACK_EVENTS:
        print(f"[track-debug] {message}")


def fmt_debug_number(value, precision=3, default="NA"):
    # 调试日志中有些字段可能为空，这里统一做安全格式化，避免日志本身触发异常。
    if value is None:
        return default
    return f"{value:.{precision}f}"


def update_identity_bank(identity_bank, person_uuid, track_info):
    # identity_bank 是一个更“长记性”的身份库，
    # 用来处理从画面一边出去、另一边回来这类远场重入。
    existing = identity_bank.get(person_uuid)
    candidate = clone_track_info(track_info)
    if existing is None:
        identity_bank[person_uuid] = candidate
        return

    merged_gallery = list(existing.get("gallery", []))
    for emb in candidate.get("gallery", []):
        merged_gallery = append_to_gallery(merged_gallery, emb)

    best_embedding = existing["embedding"]
    best_quality = existing.get("quality_score", 0.0)
    if candidate.get("quality_score", 0.0) >= best_quality:
        best_embedding = candidate["embedding"]
        best_quality = candidate.get("quality_score", 0.0)

    identity_bank[person_uuid] = {
        "bbox": candidate["bbox"],
        "last_seen": candidate["last_seen"],
        "embedding": best_embedding,
        "gallery": merged_gallery,
        "velocity": candidate.get("velocity", (0.0, 0.0)),
        "quality_score": best_quality,
    }


def parse_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def build_argparser():
    parser = argparse.ArgumentParser(description="YOLO11 person tracking with ReID")
    parser.add_argument("--source", default="0", help="camera index, video path, or stream url")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO model path")
    parser.add_argument("--tracker", default=TRACKER_PATH, help="tracker yaml path")
    parser.add_argument("--reid-weights", default=REID_WEIGHTS_PATH, help="ReID weights path")
    parser.add_argument("--window-title", default="YOLO11 Person Tracking", help="OpenCV window title")
    return parser


def main():
    args = build_argparser().parse_args()
    source = parse_source(args.source)

    install_custom_botsort_reid(args.reid_weights)
    yolo = YOLO(args.model)
    reid_encoder = PersonReIDEncoder(args.reid_weights)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Cannot open source: {args.source}")
        raise SystemExit(1)

    print(f"Tracking started with source={args.source}")
    print(f"Using model={args.model}")
    print(f"Using tracker={args.tracker}")
    print(f"Using reid_weights={args.reid_weights}")

    track_uuid_map = {}
    active_tracks = {}
    lost_tracks = {}
    identity_bank = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from source.")
            break

        now_ts = time.time()
        current_tracks = {}
        active_uuids = {info["uuid"] for info in active_tracks.values()}

        results = yolo.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=0.15,
            tracker=args.tracker,
            verbose=False,
        )

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.int().tolist()
            track_ids = result.boxes.id.int().tolist()
            confs = result.boxes.conf.tolist()
            cls_ids = result.boxes.cls.int().tolist()
            names_dict = result.names

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                if names_dict[cls_id] != "person":
                    continue

                clipped_box = clamp_box(box, frame.shape[1], frame.shape[0])
                if clipped_box is None:
                    continue

                x1, y1, x2, y2 = clipped_box
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                current_embedding = reid_encoder.encode(person_crop)
                quality_info = compute_reid_quality(conf, clipped_box, person_crop)
                reused_info = None
                if track_id not in track_uuid_map:
                    reused_uuid, reused_info, match_debug = match_lost_uuid(
                        clipped_box,
                        current_embedding,
                        lost_tracks,
                        identity_bank,
                        now_ts,
                        active_uuids,
                    )
                    track_uuid_map[track_id] = reused_uuid or str(uuid.uuid4())[:8]
                    if reused_uuid is not None:
                        log_track_event(
                            "reconnected "
                            f"track_id={track_id} -> uuid={reused_uuid} "
                            f"source={match_debug['source']} "
                            f"reason={match_debug['reason']} "
                            f"reid_max={fmt_debug_number(match_debug['max_similarity'])} "
                            f"reid_mean={fmt_debug_number(match_debug['mean_similarity'])} "
                            f"margin={fmt_debug_number(match_debug['margin'])} "
                            f"iou={fmt_debug_number(match_debug['iou'])} "
                            f"distance={fmt_debug_number(match_debug['distance'], precision=1)}/"
                            f"{fmt_debug_number(match_debug['center_limit'], precision=1)} "
                            f"lost_for={fmt_debug_number(match_debug['elapsed'], precision=2)}s "
                            f"score={fmt_debug_number(match_debug['score'])}"
                        )
                    else:
                        if match_debug is not None:
                            log_track_event(
                                "new_uuid "
                                f"track_id={track_id} -> uuid={track_uuid_map[track_id]} "
                                f"source={match_debug['source']} "
                                f"best_candidate={match_debug['candidate_uuid']} "
                                f"reason={match_debug['reason']} "
                                f"reid_max={fmt_debug_number(match_debug['max_similarity'])} "
                                f"reid_mean={fmt_debug_number(match_debug['mean_similarity'])} "
                                f"margin={fmt_debug_number(match_debug['margin'])} "
                                f"iou={fmt_debug_number(match_debug['iou'])} "
                                f"distance={fmt_debug_number(match_debug['distance'], precision=1)}/"
                                f"{fmt_debug_number(match_debug['center_limit'], precision=1)} "
                                f"lost_for={fmt_debug_number(match_debug['elapsed'], precision=2)}s"
                            )
                        else:
                            log_track_event(
                                f"new_uuid track_id={track_id} -> uuid={track_uuid_map[track_id]} reason=no_lost_candidate"
                            )

                person_uuid = track_uuid_map[track_id]
                previous_info = active_tracks.get(track_id) or reused_info
                previous_embedding = previous_info["embedding"] if previous_info else None
                stable_embedding = blend_embeddings(previous_embedding, current_embedding)

                previous_gallery = previous_info.get("gallery", []) if previous_info else []
                gallery = append_to_gallery(previous_gallery, stable_embedding)
                if should_use_embedding_update(quality_info):
                    gallery = append_to_gallery(gallery, current_embedding)
                elif track_id not in track_uuid_map:
                    log_track_event(
                        f"skip_gallery_update track_id={track_id} "
                        f"quality={quality_info['quality_score']:.3f} "
                        f"sharpness={quality_info['sharpness']:.1f} "
                        f"aspect={quality_info['aspect_ratio']:.2f}"
                    )

                previous_box = previous_info["bbox"] if previous_info else clipped_box
                previous_time = previous_info["last_seen"] if previous_info else now_ts
                velocity = estimate_velocity(previous_box, clipped_box, now_ts - previous_time)

                current_tracks[track_id] = {
                    "uuid": person_uuid,
                    "bbox": clipped_box,
                    "last_seen": now_ts,
                    "embedding": stable_embedding,
                    "gallery": gallery,
                    "velocity": velocity,
                    "quality_score": quality_info["quality_score"],
                }

                if quality_info["quality_score"] >= IDENTITY_BANK_MIN_QUALITY_SCORE:
                    update_identity_bank(identity_bank, person_uuid, current_tracks[track_id])

                display_text = f"UUID: {person_uuid} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    display_text,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        disappeared_track_ids = set(active_tracks) - set(current_tracks)
        for track_id in disappeared_track_ids:
            info = active_tracks[track_id]
            lost_tracks[info["uuid"]] = {
                "bbox": info["bbox"],
                "last_seen": info["last_seen"],
                "embedding": info["embedding"],
                "gallery": list(info.get("gallery", [])),
                "velocity": info.get("velocity", (0.0, 0.0)),
                "quality_score": info.get("quality_score", 0.0),
            }
            update_identity_bank(identity_bank, info["uuid"], info)
            log_track_event(
                f"lost track_id={track_id} uuid={info['uuid']} "
                f"gallery={len(info.get('gallery', []))} "
                f"bbox={info['bbox']}"
            )

        for person_uuid, info in list(lost_tracks.items()):
            if now_ts - info["last_seen"] > LOST_UUID_TTL_SECONDS:
                lost_tracks.pop(person_uuid, None)

        for person_uuid, info in list(identity_bank.items()):
            if now_ts - info["last_seen"] > IDENTITY_BANK_TTL_SECONDS:
                identity_bank.pop(person_uuid, None)

        active_tracks = current_tracks
        track_uuid_map = {track_id: info["uuid"] for track_id, info in active_tracks.items()}

        cv2.imshow(args.window_title, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")


if __name__ == "__main__":
    main()
