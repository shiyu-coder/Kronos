import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference

from schemas.prediction import (
    PricePredictionRequest,
    PricePredictionResponse,
    PredictionOptionsResponse,
)
from errors.errors  import PredictionAPIError
from services.price_prediction import PricePredictionService
from services.batch_price_prediction_service import BatchPricePredictionService

app = FastAPI(
    title="Kronos Prediction Server",
    description="Fetch OHLCV data and predict future prices with selectable Kronos models.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = PricePredictionService()
batch_prediction_service = BatchPricePredictionService()
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
        # Your OpenAPI document
        openapi_url=app.openapi_url,
        # Avoid CORS issues (optional)
        scalar_proxy_url="https://proxy.scalar.com",
    )


@app.get("/prediction/options", response_model=PredictionOptionsResponse)
def prediction_options() -> PredictionOptionsResponse:
    return prediction_service.get_options()


@app.post("/prediction/price", response_model=PricePredictionResponse)
def predict_price(request: PricePredictionRequest) -> PricePredictionResponse:
    return prediction_service.predict_price(request)


@app.post("/prediction/price/batch", response_model=list[PricePredictionResponse])
def predict_price(
    request: list[PricePredictionRequest],
) -> list[PricePredictionResponse]:
    return batch_prediction_service.predict_batch(request)
