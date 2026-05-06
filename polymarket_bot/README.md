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

### Cách 1 (khuyến nghị): từ thư mục root repo `Kronos/`

```bash
python -m polymarket_bot.main
```

### Cách 2: nếu đang đứng trong thư mục `Kronos/polymarket_bot/`

```bash
python -m main
```

## Kiểm tra compile

### Từ root repo `Kronos/`

```bash
python -m py_compile polymarket_bot/main.py polymarket_bot/data/collector.py polymarket_bot/model/kronos_adapter.py polymarket_bot/decision/engine.py polymarket_bot/execution/executor.py polymarket_bot/monitoring/logger.py polymarket_bot/config/settings.py
```

### Từ `Kronos/polymarket_bot/`

```bash
python -m py_compile main.py data/collector.py model/kronos_adapter.py decision/engine.py execution/executor.py monitoring/logger.py config/settings.py
```

> Mặc định đang chạy chế độ `dry_run=True`, không gửi lệnh thật.
