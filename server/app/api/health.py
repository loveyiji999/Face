from fastapi import APIRouter

router = APIRouter()

@router.get("/health", summary="健康檢查")
def health_check():
    """
    回傳服務狀態，方便 CI/CD 與 Load Balancer 檢測
    """
    return {"status": "ok"}
