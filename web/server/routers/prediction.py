from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, UploadFile

from schemas.prediction import (
    PredictionOptionsResponse,
    PricePredictionRequest,
    PricePredictionResponse,
    UploadedDataFileResponse,
)
from errors.errors import DataFormatError
from services.prediction_service.batch_price_prediction import (
    BatchPricePredictionService,
)
from services.prediction_service.price_prediction import PricePredictionService
from services.raw_data_service.ohlcv_data import OHLCVDataService

router = APIRouter()

prediction_service = PricePredictionService()
batch_prediction_service = BatchPricePredictionService()
data_validation_service = OHLCVDataService()

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads" / "local_data"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.get("/prediction/options", response_model=PredictionOptionsResponse)
def prediction_options() -> PredictionOptionsResponse:
    return prediction_service.get_options()


@router.post(
    "/prediction/local-data/upload",
    response_model=UploadedDataFileResponse,
)
async def upload_local_data_file(
    file: UploadFile = File(...),
) -> UploadedDataFileResponse:
    original_name = Path(file.filename or "").name
    if not original_name or Path(original_name).suffix.lower() != ".csv":
        raise DataFormatError(
            "Only CSV files can be uploaded as local data sources.",
            details={"filename": original_name},
        )

    contents = await file.read()
    size_bytes = len(contents)
    if size_bytes == 0:
        raise DataFormatError(
            "The uploaded CSV file is empty.",
            details={"filename": original_name},
        )
    if size_bytes > MAX_UPLOAD_BYTES:
        raise DataFormatError(
            "The uploaded CSV file is too large.",
            details={
                "filename": original_name,
                "size_bytes": size_bytes,
                "max_size_bytes": MAX_UPLOAD_BYTES,
            },
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}_{original_name}"
    stored_path = UPLOAD_DIR / stored_name
    stored_path.write_bytes(contents)

    try:
        df = pd.read_csv(stored_path)
        normalized = data_validation_service._normalize_columns(df)
    except DataFormatError:
        stored_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise DataFormatError(
            "Could not read the uploaded CSV file.",
            details={"filename": original_name},
        ) from exc

    return UploadedDataFileResponse(
        filename=original_name,
        stored_path=str(stored_path),
        size_bytes=size_bytes,
        row_count=len(normalized),
        columns=list(df.columns),
    )


@router.post("/prediction/price", response_model=PricePredictionResponse)
def predict_price(request: PricePredictionRequest) -> PricePredictionResponse:
    return prediction_service.predict_price(request)


@router.post("/prediction/price/batch", response_model=list[PricePredictionResponse])
def predict_price(
    request: list[PricePredictionRequest],
) -> list[PricePredictionResponse]:
    return batch_prediction_service.predict_batch(request)
