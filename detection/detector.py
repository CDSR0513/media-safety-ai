from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple

# 사전학습 모델: 초경량 n 버전 (커스텀 학습 없이 시작)
_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

def read_image(file) -> np.ndarray:
    """file(bytes or file-like) -> BGR np.ndarray"""
    if isinstance(file, (str, Path)):
        img = cv2.imread(str(file))
        return img
    # Streamlit uploader의 BytesIO 대응
    pil = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def run_detection_bgr(bgr: np.ndarray):
    """BGR ndarray -> results(list of Boxes)"""
    model = load_model()
    results = model.predict(source=bgr, imgsz=640, conf=0.25, verbose=False)
    return results[0]  # first prediction

def draw_boxes(bgr: np.ndarray, result, label_map=None):
    """결과 bbox를 그려 BGR 반환"""
    if label_map is None:
        # YOLOv8 COCO 기본 라벨 맵 (간단 표기)
        label_map = lambda i: result.names.get(int(i), str(i))
    out = bgr.copy()
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        label = f"{label_map(cls)} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out

def summarize(result):
    """탐지 요약: 총개수 & 클래스 카운트 dict"""
    from collections import Counter
    cnt = Counter()
    for b in result.boxes:
        cnt[int(b.cls[0].item())] += 1
    return sum(cnt.values()), {result.names[k]: v for k, v in cnt.items()}
