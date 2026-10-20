"""Offline timestamp regressions; no checkpoints or model inference required."""
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from model.kronos import KronosPredictor, calc_time_stamps


class TimestampInputsTest(unittest.TestCase):
    def test_datetime_index_matches_series(self):
        for tz in (None, 'UTC', 'America/New_York'):
            with self.subTest(tz=tz):
                timestamps = pd.date_range('2026-11-01', periods=5, freq='h', tz=tz)
                pd.testing.assert_frame_equal(
                    calc_time_stamps(timestamps),
                    calc_time_stamps(pd.Series(timestamps)),
                )

    def test_series_preserves_labels_and_order(self):
        timestamps = pd.Series(pd.to_datetime(['2026-02-03 14:25', '2025-12-31 23:59']), index=[7, 3])
        expected = pd.DataFrame({
            'minute': [25, 59], 'hour': [14, 23], 'weekday': [1, 2],
            'day': [3, 31], 'month': [2, 12],
        }, index=[7, 3], dtype='int32')
        pd.testing.assert_frame_equal(calc_time_stamps(timestamps), expected)

    def test_empty_datetime_index(self):
        pd.testing.assert_frame_equal(
            calc_time_stamps(pd.DatetimeIndex([])),
            calc_time_stamps(pd.Series([], dtype='datetime64[ns]')),
        )

    def test_predictor_boundaries_accept_datetime_index(self):
        predictor = KronosPredictor(Mock(), Mock(), device='cpu')
        x_timestamp = pd.date_range('2026-10-20 09:30', periods=3, freq='min', tz='Asia/Shanghai')
        y_timestamp = pd.date_range('2026-10-20 09:33', periods=2, freq='min', tz='Asia/Shanghai')
        df = pd.DataFrame({col: [1., 2., 3.] for col in predictor.price_cols})
        # Exercise real preprocessing/output assembly, replacing only model execution.
        predictor.generate = Mock(return_value=np.zeros((1, 2, 6), dtype=np.float32))
        single = predictor.predict(df, x_timestamp, y_timestamp, pred_len=2)
        np.testing.assert_array_equal(predictor.generate.call_args.args[1][0], calc_time_stamps(pd.Series(x_timestamp)).values)
        pd.testing.assert_index_equal(single.index, y_timestamp)
        predictor.generate = Mock(return_value=np.zeros((2, 2, 6), dtype=np.float32))
        batch = predictor.predict_batch([df, df], [x_timestamp, x_timestamp], [y_timestamp, y_timestamp], pred_len=2)
        for result in batch:
            pd.testing.assert_frame_equal(result, single)


if __name__ == '__main__':
    unittest.main()
