"""Transfer tracked RSL motion to a stylized, fixed-camera avatar."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageDraw


SIZE = 768
BACKGROUND = "#F1EEFA"
SHIRT = "#34477D"
COLLAR = "#5165A0"
SKIN = "#D9936C"
EDGE = "#8C4B39"
HAIR = "#2E263D"
FEATURE = "#422B36"
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109)
HANDS = ("left", "right")


def track(source, models, start, end):
    video = cv2.VideoCapture(str(source))
    fps = video.get(cv2.CAP_PROP_FPS)
    if not video.isOpened() or fps <= 0:
        raise ValueError(f"Cannot open {source}")
    first = round(start * fps)
    last = round(end * fps) if end is not None else int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.set(cv2.CAP_PROP_POS_FRAMES, first)

    frames = []
    try:
        trackers = Trackers(models)
        for _ in range(first, last):
            ok, frame = video.read()
            if not ok:
                break
            pose = trackers.pose(frame)
            face = trackers.face(frame)
            tracked = {"pose": pose, "face": face, "left": None, "right": None}
            if pose:
                tracked.update(assign_hands(pose, trackers.hands(frame)))
                for name, wrist, elbow in (("left", 15, 13), ("right", 16, 14)):
                    if tracked[name] is None:
                        tracked[name] = track_hand_crop(frame, pose, wrist, elbow, trackers.hands)
            frames.append(tracked)
    finally:
        video.release()

    if len(frames) < 12:
        raise ValueError("Too few source frames")
    for name in ("pose", "face", *HANDS):
        detections = [index for index, frame in enumerate(frames) if frame[name] is not None]
        print(f"{name}: {len(detections)}/{len(frames)} frames", flush=True)
        missing = [index for index, frame in enumerate(frames) if frame[name] is None]
        if missing:
            print(f"{name} missing at frames: {missing}", flush=True)
        for index, frame in enumerate(frames):
            if frame[name] is not None:
                continue
            before = max((item for item in detections if item < index), default=None)
            after = min((item for item in detections if item > index), default=None)
            if before is None or after is None or after - before > 3:
                raise ValueError(f"Missing {name} at frame {index}; choose a source with clear tracking")
            amount = (index - before) / (after - before)
            frame[name] = [between(a, b, amount) for a, b in zip(frames[before][name], frames[after][name])]
    return frames, fps


class Trackers:
    def __init__(self, models):
        # The reference OpenCV Zoo wrappers live next to their model files.
        sys.path.insert(0, str(models.resolve()))
        from mp_handpose import MPHandPose
        from mp_palmdet import MPPalmDet
        from mp_persondet import MPPersonDet
        from mp_pose import MPPose

        cv2.setNumThreads(2)
        self.person_detector = MPPersonDet(str(models / "person_detection_mediapipe_2023mar.onnx"))
        self.pose_tracker = MPPose(str(models / "pose_estimation_mediapipe_2023mar.onnx"))
        self.palm_detector = MPPalmDet(str(models / "palm_detection_mediapipe_2023feb.onnx"), scoreThreshold=0.35)
        self.hand_tracker = MPHandPose(str(models / "handpose_estimation_mediapipe_2023feb.onnx"), confThreshold=0.65)
        self.face_detector = cv2.FaceDetectorYN.create(
            str(models / "face_detection_yunet_2023mar.onnx"), "", (320, 320), score_threshold=0.7
        )
        self.face_mesh = Interpreter(model_path=str(models / "face_landmarks_detector.tflite"), num_threads=2)
        self.face_mesh.allocate_tensors()
        self.face_input = self.face_mesh.get_input_details()[0]["index"]
        outputs = self.face_mesh.get_output_details()
        self.face_points = next(output["index"] for output in outputs if output["name"] == "Identity")
        self.face_presence = next(output["index"] for output in outputs if output["name"] == "Identity_1")

    def pose(self, frame):
        people = self.person_detector.infer(frame)
        if not len(people):
            return None
        result = self.pose_tracker.infer(frame, people[0])
        return result[1][:33, :2].tolist() if result is not None else None

    def hands(self, frame):
        hands = []
        for palm in self.palm_detector.infer(frame)[:2]:
            result = self.hand_tracker.infer(frame, palm)
            if result is not None:
                hands.append(result[4:67].reshape(21, 3)[:, :2].tolist())
        return hands

    def face(self, frame):
        height, width = frame.shape[:2]
        reduction = min(1, 640 / max(width, height))
        small = cv2.resize(frame, (round(width * reduction), round(height * reduction)))
        self.face_detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.face_detector.detect(small)
        if faces is None:
            return None
        x, y, w, h = faces[0][:4] / reduction
        size = max(w, h) * 1.5
        left, top = x + (w - size) / 2, y + (h - size) / 2 - h * 0.08
        scale = 256 / size
        transform = np.float32([[scale, 0, -left * scale], [0, scale, -top * scale]])
        crop = cv2.warpAffine(frame, transform, (256, 256), borderMode=cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        self.face_mesh.set_tensor(self.face_input, rgb[np.newaxis])
        self.face_mesh.invoke()
        if self.face_mesh.get_tensor(self.face_presence).item() < 0:
            return None
        points = self.face_mesh.get_tensor(self.face_points).reshape(-1, 3)
        return (points[:, :2] / scale + [left, top]).tolist()


def assign_hands(pose, hands):
    left, right = pose[15], pose[16]
    if len(hands) == 2:
        first, second = hands
        if math.dist(first[0], left) + math.dist(second[0], right) > (
            math.dist(second[0], left) + math.dist(first[0], right)
        ):
            first, second = second, first
        return {"left": first, "right": second}
    if hands:
        hand = hands[0]
        return {"left" if math.dist(hand[0], left) < math.dist(hand[0], right) else "right": hand}
    return {}


def track_hand_crop(frame, pose, wrist, elbow, track_hands):
    x, y = between(pose[elbow], pose[wrist], 1.18)
    radius = max(50, math.dist(pose[11], pose[12]) * 0.64)
    left, right = max(0, round(x - radius)), min(frame.shape[1], round(x + radius))
    top, bottom = max(0, round(y - radius)), min(frame.shape[0], round(y + radius))
    if right <= left or bottom <= top:
        return None
    hands = [[(left + x, top + y) for x, y in points]
             for points in track_hands(frame[top:bottom, left:right])]
    if not hands:
        return None
    hand = min(hands, key=lambda points: math.dist(points[0], pose[wrist]))
    return hand if math.dist(hand[0], pose[wrist]) < radius * 0.6 else None


def between(a, b, amount):
    return (a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount)


def smooth(frames):
    # Centered filters add no timing offset. Keep the fingers on the shorter window
    # so brief bends and changes in the relative position of the fingers survive.
    for name, weights in (
        ("pose", (1, 4, 6, 4, 1)),
        ("face", (1, 4, 6, 4, 1)),
        ("left", (1, 4, 1)),
        ("right", (1, 4, 1)),
    ):
        points = np.array([frame[name] for frame in frames])
        if name in HANDS:
            wrist = points[:, :1]
            filtered = temporal_average(wrist, (1, 4, 6, 4, 1)) + temporal_average(points - wrist, weights)
        else:
            filtered = temporal_average(points, weights)
        for frame, landmarks in zip(frames, filtered.tolist()):
            frame[name] = landmarks


def temporal_average(points, weights):
    radius = len(weights) // 2
    padded = np.pad(points, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    return np.average(
        [padded[offset:offset + len(points)] for offset in range(len(weights))],
        axis=0,
        weights=weights,
    )


def double_frames(frames):
    for index, frame in enumerate(frames):
        yield frame
        following = frames[min(index + 1, len(frames) - 1)]
        yield {
            name: [between(a, b, 0.5) for a, b in zip(points, following[name])]
            for name, points in frame.items()
        }


def circle(draw, center, radius, color):
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def stroke(draw, points, width, color):
    width = max(1, round(width))
    draw.line(points, fill=color, width=width, joint="curve")
    for point in points:
        circle(draw, point, width / 2, color)


def camera(frames):
    visible = []
    for frame in frames:
        visible.extend(frame["face"][i] for i in FACE_OVAL)
        visible.extend(frame["pose"][i] for i in (11, 12, 13, 14, 23, 24))
        for name in HANDS:
            visible.extend(frame[name])
    left = min(x for x, _ in visible)
    right = max(x for x, _ in visible)
    top = min(y for _, y in visible)
    bottom = max(y for _, y in visible)
    scale = min((SIZE - 156) / (right - left), (SIZE - 136) / (bottom - top))
    center = (left + right) / 2
    return lambda p: (SIZE / 2 + (p[0] - center) * scale, 65 + (p[1] - top) * scale)


def draw_hand(draw, points):
    palm_width = math.dist(points[5], points[17])
    palm = [points[i] for i in (0, 1, 2, 5, 9, 13, 17)]
    fingers = ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20])
    draw.polygon(palm, fill=SKIN)
    draw.line(palm + [palm[0]], fill=EDGE, width=2, joint="curve")
    for finger, fraction in zip(fingers, (0.23, 0.19, 0.21, 0.18, 0.15)):
        width = max(4, palm_width * fraction)
        points_on_finger = [points[i] for i in finger]
        for index, (start, end) in enumerate(zip(points_on_finger, points_on_finger[1:])):
            stroke(draw, [start, end], width * (1 - index * 0.1) + 2, EDGE)
    for finger, fraction in zip(fingers, (0.23, 0.19, 0.21, 0.18, 0.15)):
        width = max(4, palm_width * fraction)
        points_on_finger = [points[i] for i in finger]
        for index, (start, end) in enumerate(zip(points_on_finger, points_on_finger[1:])):
            stroke(draw, [start, end], width * (1 - index * 0.1), SKIN)
    # The finger bases are part of the palm, so only retain the outer hand contour.
    draw.polygon(palm, fill=SKIN)
    draw.line([points[i] for i in (17, 0, 1)], fill=EDGE, width=2)


def draw_face(draw, face):
    contour = [face[i] for i in FACE_OVAL]
    x0 = min(x for x, _ in contour)
    x1 = max(x for x, _ in contour)
    y0 = min(y for _, y in contour)
    y1 = max(y for _, y in contour)
    width = x1 - x0
    height = y1 - y0
    draw.ellipse((x0 - width * 0.08, y0 - height * 0.13, x1 + width * 0.08, y1 - height * 0.16), fill=HAIR)
    draw.polygon(contour, fill=SKIN)
    draw.polygon([face[i] for i in (127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284,
                                     251, 389, 356, 301, 298, 333, 299, 151, 69, 104, 68, 71)], fill=HAIR)

    for outline, iris, brow in (
        ((33, 160, 158, 133, 153, 144), (33, 133), (70, 63, 105, 66, 107)),
        ((362, 385, 387, 263, 373, 380), (362, 263), (336, 296, 334, 293, 300)),
    ):
        draw.polygon([face[i] for i in outline], fill="#FFF7E8", outline=FEATURE)
        center = between(face[iris[0]], face[iris[1]], 0.5)
        opening = math.dist(face[outline[1]], face[outline[-1]])
        if opening > width * 0.019:
            circle(draw, center, min(width * 0.028, opening * 0.48), FEATURE)
        stroke(draw, [face[i] for i in brow], width * 0.032, FEATURE)

    stroke(draw, [face[i] for i in (168, 1, 2)], width * 0.016, EDGE)
    lips = [face[i] for i in (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91)]
    draw.polygon(lips, fill="#A84C43")
    draw.polygon([face[i] for i in (78, 81, 13, 311, 308, 402, 14, 178)], fill=FEATURE)


def render(frame, project):
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    pose = [project(point) for point in frame["pose"]]
    face = [project(point) for point in frame["face"]]
    shoulder_width = math.dist(pose[11], pose[12])
    neck = between(pose[11], pose[12], 0.5)
    chin = face[152]
    left_shoulder, right_shoulder = sorted((pose[11], pose[12]))
    left_hip, right_hip = sorted((pose[23], pose[24]))
    half_neck = shoulder_width * 0.117
    draw.polygon([
        (chin[0] - half_neck, chin[1] - shoulder_width * 0.12),
        (chin[0] + half_neck, chin[1] - shoulder_width * 0.12),
        (neck[0] + half_neck * 1.35, neck[1] + shoulder_width * 0.06),
        (neck[0] - half_neck * 1.35, neck[1] + shoulder_width * 0.06),
    ], fill=SKIN)
    collar_left = (neck[0] - half_neck * 1.28, neck[1] - shoulder_width * 0.045)
    collar_right = (neck[0] + half_neck * 1.28, neck[1] - shoulder_width * 0.045)
    collar_bottom = (neck[0], neck[1] + shoulder_width * 0.068)
    collar_curve = [between(between(collar_left, collar_bottom, t),
                            between(collar_bottom, collar_right, t), t) for t in np.linspace(0, 1, 12)]
    draw.polygon([
        (left_shoulder[0] - shoulder_width * 0.10, left_shoulder[1] + shoulder_width * 0.12),
        (left_shoulder[0], left_shoulder[1] - shoulder_width * 0.075),
        *collar_curve,
        (right_shoulder[0], right_shoulder[1] - shoulder_width * 0.075),
        (right_shoulder[0] + shoulder_width * 0.10, right_shoulder[1] + shoulder_width * 0.12),
        (right_hip[0] + shoulder_width * 0.13, SIZE),
        (left_hip[0] - shoulder_width * 0.13, SIZE),
    ], fill=SHIRT)
    stroke(draw, collar_curve, shoulder_width * 0.022, COLLAR)

    for hand, shoulder, elbow in (("left", 11, 13), ("right", 12, 14)):
        hand_points = [project(point) for point in frame[hand]]
        wrist = hand_points[0]
        sleeve = between(pose[shoulder], pose[elbow], 0.32)
        stroke(draw, [sleeve, pose[elbow]], shoulder_width * 0.17, SKIN)
        stroke(draw, [pose[elbow], wrist], shoulder_width * 0.13, SKIN)
        stroke(draw, [pose[shoulder], sleeve], shoulder_width * 0.24, SHIRT)

    draw_face(draw, face)
    for hand in HANDS:
        draw_hand(draw, [project(point) for point in frame[hand]])
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--track-json", type=Path)
    args = parser.parse_args()
    frames, fps = track(args.source, args.models, args.start, args.end)
    if args.track_json:
        args.track_json.write_text(json.dumps({"fps": fps, "frames": frames}), encoding="utf-8")
    smooth(frames)
    project = camera(frames)
    output_frames = double_frames(frames) if fps <= 30 else frames
    output_fps = fps * 2 if fps <= 30 else fps
    with tempfile.TemporaryDirectory(prefix="avatar-frames-") as directory:
        directory = Path(directory)
        for index, frame in enumerate(output_frames):
            render(frame, project).save(directory / f"{index:05}.png")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(output_fps), "-i", str(directory / "%05d.png"),
            "-vf", "scale=576:576:flags=lanczos",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output),
        ], check=True)


if __name__ == "__main__":
    main()
