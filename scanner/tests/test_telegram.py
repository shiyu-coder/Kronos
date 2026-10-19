from scanner.alerts.telegram import send_telegram_message


class DummyLogger:
    def error(self, *args, **kwargs):
        return None


def test_telegram_send_fail(monkeypatch):
    import requests

    class Resp:
        status_code = 500
        text = "fail"

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    ok = send_telegram_message("token", "chat", "msg", DummyLogger())
    assert ok is False
