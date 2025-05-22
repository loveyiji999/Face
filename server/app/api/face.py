from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
from app.services.scrfd_service import detect_faces, preprocess_face, get_embedding, cosine_similarity

router = APIRouter()

@router.post("/detect")
async def detect_route(file: UploadFile = File(...)):
    """
    上傳影像檔，回傳偵測到的人臉 bounding boxes 與置信度。
    """
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="讀取圖片失敗")
    faces = detect_faces(img)
    return {"faces": [
        {"box": [x1, y1, x2, y2], "score": score}
        for x1, y1, x2, y2, score in faces
    ]}

@router.post("/compare")
async def compare_route(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    上傳兩張影像，取第一張臉計算 Embedding 相似度。
    """
    data1 = await file1.read()
    data2 = await file2.read()
    img1 = cv2.imdecode(np.frombuffer(data1, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(data2, np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise HTTPException(status_code=400, detail="讀取圖片失敗")
    f1 = detect_faces(img1)
    f2 = detect_faces(img2)
    if not f1 or not f2:
        raise HTTPException(status_code=400, detail="偵測不到臉")
    emb1 = get_embedding(preprocess_face(img1, f1[0]))
    emb2 = get_embedding(preprocess_face(img2, f2[0]))
    score = cosine_similarity(emb1, emb2)
    return {"similarity": score}
