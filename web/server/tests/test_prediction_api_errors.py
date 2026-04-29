import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from services.raw_data_service.ohlcv_data import OHLCVDataService

import main


class PredictionAPIErrorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_prediction_price_validation_error_has_stable_shape(self):
        response = self.client.post("/api/prediction/price", json={"lookback": 1})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error"]["code"], "REQUEST_VALIDATION_ERROR")
        self.assertIn("fields", body["error"]["details"])

    def test_prediction_price_requires_symbol_for_binance(self):
        response = self.client.post(
            "/api/prediction/price",
            json={"data_source": "binance", "symbol": None},
        )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["code"], "UNSUPPORTED_OPTION")
        self.assertEqual(body["error"]["details"]["field"], "symbol")

    def test_prediction_price_reports_missing_local_file(self):
        response = self.client.post(
            "/api/prediction/price",
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
            "/api/prediction/price",
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

    def test_upload_local_data_rejects_non_csv_file(self):
        response = self.client.post(
            "/api/prediction/local-data/upload",
            files={"file": ("prices.txt", b"not,csv", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["code"], "DATA_FORMAT_ERROR")

    def test_upload_local_data_validates_required_columns(self):
        response = self.client.post(
            "/api/prediction/local-data/upload",
            files={
                "file": ("prices.csv", b"timestamps,open\n2025-01-01,1\n", "text/csv")
            },
        )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["code"], "DATA_FORMAT_ERROR")
        self.assertIn("missing_columns", body["error"]["details"])

    def test_upload_local_data_returns_stored_path(self):
        csv = (
            "timestamps,open,high,low,close,volume\n"
            "2025-01-01 00:00:00,1,2,0.5,1.5,100\n"
            "2025-01-01 01:00:00,2,3,1.5,2.5,110\n"
        )

        response = self.client.post(
            "/api/prediction/local-data/upload",
            files={"file": ("prices.csv", csv.encode("utf-8"), "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["filename"], "prices.csv")
        self.assertEqual(body["row_count"], 2)
        stored_path = Path(body["stored_path"])
        self.assertTrue(stored_path.exists())
        stored_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
