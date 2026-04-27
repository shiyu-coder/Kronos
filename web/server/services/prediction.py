import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer

from constants.available_models import AVAILABLE_MODELS


class KronosPredictionService:
    """Service wrapper for loading Kronos models and predicting future candles."""

    def __init__(
        self,
        model_name="kronos-mini",
        device="cpu",
    ):
        """Create the service with the selected model and compute device."""
        self.model_key = None
        self.model_name = None
        self.tokenizer_name = None
        self.max_context = None
        self.device = device
        self.predictor: KronosPredictor | None = None
        self.set_model(model_name)

    def set_model(self, model_name: str):
        """Select a Kronos model configuration and clear any loaded predictor."""
        if model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_config = AVAILABLE_MODELS[model_name]
        self.model_key = model_name
        self.model_name = model_config["model_id"]
        self.tokenizer_name = model_config["tokenizer_id"]
        self.max_context = model_config["context_length"]
        self.predictor = None

    def set_device(self, device: str):
        """Select the runtime device and clear any loaded predictor."""
        self.device = device
        self.predictor = None

    def configure(self, model_name: str | None = None, device: str | None = None):
        """Update model and/or device settings before the next prediction."""
        if model_name is not None:
            self.set_model(model_name)
        if device is not None:
            self.set_device(device)
        return self

    def load(self):
        """Load the tokenizer, model, and predictor for the current settings."""
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        model = Kronos.from_pretrained(self.model_name)

        self.predictor = KronosPredictor(
            model,
            tokenizer,
            device=self.device,
            max_context=self.max_context,
        )

        return self.predictor

    def predict(
        self,
        df,
        x_timestamp,
        y_timestamp,
        pred_len: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ):
        """Run candle prediction using the loaded model, loading it if needed."""
        if self.predictor is None:
            self.load()

        return self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ):
        """
        Run batch prediction on multiple time series in parallel.

        Args:
            df_list: List of DataFrames with OHLCVA columns
            x_timestamp_list: List of historical timestamp indices
            y_timestamp_list: List of future timestamp indices
            pred_len: Number of prediction steps (must be identical for all series)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling
            sample_count: Number of samples per series
            verbose: Show progress

        Returns:
            List of prediction DataFrames in same order as input

        Constraints:
            - All DataFrames must have same number of rows
            - All x_timestamp_list entries must have same length
            - All y_timestamp_list entries must have same length
        """
        if self.predictor is None:
            self.load()

        return self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_timestamp_list,
            y_timestamp_list=y_timestamp_list,
            pred_len=pred_len,
            T=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
