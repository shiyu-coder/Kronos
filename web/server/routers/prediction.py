from fastapi import APIRouter

from schemas.prediction import (PredictionOptionsResponse,
                                PricePredictionRequest,
                                PricePredictionResponse)
from services.prediction_service.batch_price_prediction import \
    BatchPricePredictionService
from services.prediction_service.price_prediction import PricePredictionService

router = APIRouter()

prediction_service = PricePredictionService()
batch_prediction_service = BatchPricePredictionService()


@router.get("/prediction/options", response_model=PredictionOptionsResponse)
def prediction_options() -> PredictionOptionsResponse:
    return prediction_service.get_options()


@router.post("/prediction/price", response_model=PricePredictionResponse)
def predict_price(request: PricePredictionRequest) -> PricePredictionResponse:
    return prediction_service.predict_price(request)


@router.post("/prediction/price/batch", response_model=list[PricePredictionResponse])
def predict_price(
    request: list[PricePredictionRequest],
) -> list[PricePredictionResponse]:
    return batch_prediction_service.predict_batch(request)
