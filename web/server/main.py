import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference

from db.migrations import run_migrations
from errors.errors import PredictionAPIError
from routers import prediction, results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations on startup."""
    run_migrations()
    yield


app = FastAPI(
    title="Kronos Prediction Server",
    description="Fetch OHLCV data and predict future prices with selectable Kronos models.",
    version="0.1.0",
    lifespan=lifespan,
)

# Include Routers
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])
app.include_router(results.router, prefix="/api", tags=["Results"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "Invalid prediction request. Please check the submitted fields.",
                    "details": {"fields": exc.errors()},
                }
            }
        ),
    )


@app.exception_handler(PredictionAPIError)
async def prediction_api_exception_handler(request: Request, exc: PredictionAPIError):
    return JSONResponse(status_code=exc.status_code, content=exc.to_response())


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled prediction server error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Prediction failed because of an unexpected server error.",
            }
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,  # Your OpenAPI document
        scalar_proxy_url="https://proxy.scalar.com",  # Avoid CORS issues (optional)
    )
