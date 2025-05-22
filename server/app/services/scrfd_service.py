import os
import cv2
import numpy as np
import onnxruntime as ort

_det_sess = None
_rec_sess = None

def load_scrfd_model(det_path: str = None, rec_path: str = None):
    """
    初始化 SCRFD 偵測 Session 與辨識 Session (singleton)。
    """
    global _det_sess, _rec_sess
    # 預設模型目錄：server/models
    model_dir = os.getenv(
        "MODEL_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    )
    det_file = det_path or os.path.join(model_dir, "scrfd_person_2.5g.onnx")
    rec_file = rec_path or os.path.join(model_dir, "buffalo_l", "w600k_r50.onnx")
    if _det_sess is None:
        _det_sess = ort.InferenceSession(det_file)
    if _rec_sess is None:
        _rec_sess = ort.InferenceSession(rec_file)
    return _det_sess, _rec_sess


def non_max_suppression(boxes, scores, score_thresh=0.1, nms_thresh=0.4):
    """
    使用 OpenCV DNN NMS 對檢測框進行去重。
    boxes: list of (x1,y1,x2,y2), scores: list of float
    """
    # OpenCV NMSBoxes 接受 [x,y,w,h]
    bboxes = [[x, y, x2-x, y2-y] for x, y, x2, y2 in boxes]
    idxs = cv2.dnn.NMSBoxes(bboxes, scores, score_thresh, nms_thresh)
    if len(idxs) == 0:
        return []
    final = []
    for i in idxs:
        idx = int(i[0]) if hasattr(i, '__len__') else int(i)
        x1, y1, x2, y2 = boxes[idx]
        final.append((x1, y1, x2, y2, scores[idx]))
    return final


def detect_faces(img: np.ndarray, score_thresh=0.1, nms_thresh=0.4):
    """
    使用 SCRFD ONNX model 偵測影像中的人臉，
    回傳 list of (x1,y1,x2,y2,confidence)。
    """
    det_sess, _ = load_scrfd_model()
    h, w = img.shape[:2]
    input_name = det_sess.get_inputs()[0].name

    # 前處理：resize, normalization, NCHW
    resized = cv2.resize(img, (640, 640)).astype(np.float32)
    normalized = (resized - 127.5) / 128.0
    blob = normalized.transpose(2, 0, 1)[None, ...]

    # 推理
    outputs = det_sess.run(None, {input_name: blob})

    # 解析 outputs: confidence 與 bbox
    scores_list, boxes_list = [], []
    for o in outputs:
        if o.ndim == 2:
            C = o.shape[1]
            if C == 1:
                scores_list.append(o.reshape(-1))
            elif C == 4:
                boxes_list.append(o.reshape(-1, 4))
    if not scores_list or not boxes_list:
        return []

    # 合併所有尺度
    scores = np.concatenate(scores_list, axis=0)
    boxes  = np.concatenate(boxes_list, axis=0)

    # 閾值篩選
    idxs = np.where(scores > score_thresh)[0]
    sel_scores = scores[idxs]
    sel_boxes  = boxes[idxs]

    # 還原至原圖比例
    sx, sy = w / 640.0, h / 640.0
    filtered_boxes, filtered_scores = [], []
    for i, (x1, y1, x2, y2) in enumerate(sel_boxes):
        x1n = int(np.clip(x1 * sx, 0, w-1))
        y1n = int(np.clip(y1 * sy, 0, h-1))
        x2n = int(np.clip(x2 * sx, 0, w-1))
        y2n = int(np.clip(y2 * sy, 0, h-1))
        if x2n > x1n and y2n > y1n:
            filtered_boxes.append((x1n, y1n, x2n, y2n))
            filtered_scores.append(float(sel_scores[i]))

    # NMS 合併重疊框
    faces = non_max_suppression(filtered_boxes, filtered_scores, score_thresh, nms_thresh)
    return faces


def preprocess_face(img: np.ndarray, box):
    x1, y1, x2, y2, _ = box
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    blob = face.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return (blob - 127.5) / 128.0


def get_embedding(face_blob: np.ndarray):
    _, rec_sess = load_scrfd_model()
    out = rec_sess.run(None, {rec_sess.get_inputs()[0].name: face_blob})[0]
    return out[0]


def cosine_similarity(a, b):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    return float(np.dot(a, b))
