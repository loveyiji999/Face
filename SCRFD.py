import cv2
import numpy as np
import onnxruntime as ort

# 載入SCRFD-2.5G模型和人臉辨識模型，請確認路徑正確
det_sess = ort.InferenceSession(r"D:\Desktop\Face\model\scrfd_person_2.5g.onnx")
rec_sess = ort.InferenceSession(r"D:\Desktop\Face\model\buffalo_l\w600k_r50.onnx")


def non_max_suppression(boxes, scores, score_thresh=0.1, nms_thresh=0.4):
    # OpenCV NMSBoxes 要求格式 [x,y,w,h]
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


def detect_faces(img, score_thresh=0.1, nms_thresh=0.4):
    h, w = img.shape[:2]
    input_name = det_sess.get_inputs()[0].name

    # 前處理：640x640、浮點、normalization、NCHW
    resized = cv2.resize(img, (640, 640)).astype(np.float32)
    normalized = (resized - 127.5) / 128.0
    blob = normalized.transpose(2, 0, 1)[None, ...]

    # 推理
    outputs = det_sess.run(None, {input_name: blob})

    # 動態解析：shape[1]==1 為 confidence, shape[1]==4 為 bbox
    scores_list, boxes_list = [], []
    for o in outputs:
        if o.ndim == 2:
            C = o.shape[1]
            if C == 1:
                scores_list.append(o.reshape(-1))
            elif C == 4:
                boxes_list.append(o.reshape(-1, 4))
    if not scores_list or not boxes_list:
        print("輸出解析失敗，無scores或boxes，請檢查模型或輸入尺寸。")
        return []

    # 合併所有尺度
    scores = np.concatenate(scores_list, axis=0)
    boxes  = np.concatenate(boxes_list, axis=0)

    # 閾值篩選
    idx = np.where(scores > score_thresh)[0]
    sel_scores = scores[idx]
    sel_boxes  = boxes[idx]

    # 還原至原圖比例，並過濾合理框
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
    print(f"max score: {scores.max():.4f}, 最終框數: {len(faces)}")
    return faces


def preprocess_face(img, box):
    x1, y1, x2, y2, _ = box
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    blob = face.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return (blob - 127.5) / 128.0


def get_embedding(face_blob):
    input_name = rec_sess.get_inputs()[0].name
    out = rec_sess.run(None, {input_name: face_blob})[0]
    return out[0]


def cosine_similarity(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.dot(a, b))


if __name__ == '__main__':
    # 讀取圖片
    img1 = cv2.imread(r"D:\Desktop\Face\facedata\liu1.jpg")
    img2 = cv2.imread(r"D:\Desktop\Face\facedata\liu2.jpg")
    if img1 is None or img2 is None:
        print("讀不到圖，檢查路徑！"); exit()

    # 偵測並畫框
    faces1 = detect_faces(img1)
    for x1, y1, x2, y2, _ in faces1:
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.namedWindow("偵測結果1", cv2.WINDOW_NORMAL)
    cv2.imshow("偵測結果1", img1); cv2.waitKey(0); cv2.destroyAllWindows()

    faces2 = detect_faces(img2)
    for x1, y1, x2, y2, _ in faces2:
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.namedWindow("偵測結果2", cv2.WINDOW_NORMAL)
    cv2.imshow("偵測結果2", img2); cv2.waitKey(0); cv2.destroyAllWindows()

    # Embedding & 比對
    if faces1 and faces2:
        emb1 = get_embedding(preprocess_face(img1, faces1[0]))
        emb2 = get_embedding(preprocess_face(img2, faces2[0]))
        print(f"比對分數: {cosine_similarity(emb1, emb2):.3f}")
    else:
        print("偵測不到臉！")
