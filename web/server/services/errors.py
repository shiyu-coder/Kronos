class PredictionAPIError(Exception):
    """Base class for errors that should be shown to API clients."""

    status_code = 400
    code = "PREDICTION_ERROR"
    message = "Prediction request failed."

    def __init__(self, message: str | None = None, details: dict | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message
        self.details = details or {}

    def to_response(self):
        response = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            response["details"] = self.details
        return {"error": response}


class UnsupportedPredictionOptionError(PredictionAPIError):
    status_code = 400
    code = "UNSUPPORTED_OPTION"
    message = "One or more prediction options are not supported."


class DataProviderError(PredictionAPIError):
    status_code = 502
    code = "DATA_PROVIDER_ERROR"
    message = "The selected data provider could not complete the request."


class DataProviderTimeoutError(DataProviderError):
    status_code = 504
    code = "DATA_PROVIDER_TIMEOUT"
    message = "The selected data provider timed out."


class DataNotFoundError(PredictionAPIError):
    status_code = 404
    code = "DATA_NOT_FOUND"
    message = "The requested data could not be found."


class DataFormatError(PredictionAPIError):
    status_code = 400
    code = "DATA_FORMAT_ERROR"
    message = "The input data format is invalid."


class InsufficientDataError(PredictionAPIError):
    status_code = 400
    code = "INSUFFICIENT_DATA"
    message = "Not enough historical data is available for this prediction."


class ModelLoadError(PredictionAPIError):
    status_code = 503
    code = "MODEL_LOAD_ERROR"
    message = "The prediction model could not be loaded."


class PredictionRuntimeError(PredictionAPIError):
    status_code = 500
    code = "PREDICTION_RUNTIME_ERROR"
    message = "The prediction model failed while generating results."
