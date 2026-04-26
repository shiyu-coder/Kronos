import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

import main
from services.ohlcv_data import OHLCVDataService


class PredictionAPIErrorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def test_prediction_price_validation_error_has_stable_shape(self):
        response = self.client.post("/prediction/price", json={"lookback": 1})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error"]["code"], "REQUEST_VALIDATION_ERROR")
        self.assertIn("fields", body["error"]["details"])

    def test_prediction_price_requires_symbol_for_binance(self):
        response = self.client.post(
            "/prediction/price",
            json={"data_source": "binance", "symbol": None},
        )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["code"], "UNSUPPORTED_OPTION")
        self.assertEqual(body["error"]["details"]["field"], "symbol")

    def test_prediction_price_reports_missing_local_file(self):
        response = self.client.post(
            "/prediction/price",
            json={
                "data_source": "local",
                "local_path": "missing-file.csv",
                "lookback": 10,
            },
        )

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["error"]["code"], "DATA_NOT_FOUND")
        self.assertIn("missing-file.csv", body["error"]["message"])

    def test_prediction_price_rejects_unknown_model_name(self):
        response = self.client.post(
            "/prediction/price",
            json={"model_name": "bad-model"},
        )

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error"]["code"], "REQUEST_VALIDATION_ERROR")

    def test_ohlcv_service_reports_insufficient_valid_rows(self):
        service = OHLCVDataService()
        df = pd.DataFrame(
            {
                "timestamps": pd.date_range("2025-01-01", periods=2, freq="h"),
                "open": [1, 2],
                "high": [2, 3],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
            }
        )

        with self.assertRaises(Exception) as context:
            service._prepare_prediction_data(df, lookback=3, pred_len=1)

        self.assertEqual(context.exception.code, "INSUFFICIENT_DATA")
        self.assertEqual(
            context.exception.details,
            {"required_rows": 3, "available_rows": 2},
        )


if __name__ == "__main__":
    unittest.main()
