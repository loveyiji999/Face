from fastapi import FastAPI
from app.api.health import router as health_router

def create_app() -> FastAPI:
    app = FastAPI(title="Face Attendance System")
    app.include_router(health_router, prefix="/api")
    return app

app = create_app()

#uvicorn app.main:app --reload
#uvicorn server.app.main:app --reload --reload-dir server