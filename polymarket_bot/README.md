# PolyMarket Finance Bot (Scaffold)

Khung xương bot tự động cho PolyMarket, dùng Kronos (`model.Kronos`, `model.KronosTokenizer`, `model.KronosPredictor`) để suy luận xác suất xu hướng ngắn hạn.

## Cấu trúc

- `data/collector.py`: lấy market snapshot + lịch sử giá (mock, sẵn sàng thay API thật)
- `model/kronos_adapter.py`: adapter cho Kronos inference
- `decision/engine.py`: EV + Kelly + risk caps
- `execution/executor.py`: place order abstraction + retry
- `monitoring/logger.py`: logging giao dịch và metrics
- `main.py`: orchestration pipeline end-to-end

## Chạy demo

```bash
python -m polymarket_bot.main
```

> Mặc định đang chạy chế độ `dry_run=True`, không gửi lệnh thật.
