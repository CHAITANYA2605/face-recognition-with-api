import newrelic.agent
import os as _os
newrelic.agent.initialize(_os.path.join(_os.path.dirname(__file__), "newrelic.ini"))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router
from app.middleware.stats import StatsMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(StatsMiddleware)

app.include_router(router, prefix=settings.API_V1_STR)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    if request.url.path == f"{settings.API_V1_STR}/register":
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    "Invalid registration request."
                )
            },
        )

    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request. Please check the submitted data."},
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

app = newrelic.agent.ASGIApplicationWrapper(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
