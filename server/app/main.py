from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.face import router as face_router

def create_app() -> FastAPI:
    app = FastAPI(title="Face Attendance System")
    app.include_router(health_router, prefix="/api")
    app.include_router(face_router, prefix="/api")
    return app
    
app = create_app()

#uvicorn app.main:app --reload
#uvicorn server.app.main:app --reload --reload-dir server
#venv\Scripts\activate
#compare
#curl -X POST -F "file1=@D:/Desktop/Face/data/jay.jpg" -F "file2=@D:/Desktop/Face/data/jay2.jpg" http://127.0.0.1:8000/api/compare
#detect
#curl -X POST -F "file=@D:/Desktop/Face/data/jay.jpg" http://127.0.0.1:8000/api/detect
