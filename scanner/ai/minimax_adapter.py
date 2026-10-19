from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

from ..config import (
    MINIMAX_BASE_URL,
    MINIMAX_MAX_OUTPUT_TOKENS,
    MINIMAX_MODEL,
    MINIMAX_TEMPERATURE,
    MINIMAX_TIMEOUT_SECONDS,
)


class MiniMaxAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.enabled = os.getenv("MINIMAX_ENABLED", "false").strip().lower() == "true"
        self.api_key = os.getenv("MINIMAX_API_KEY", "").strip()
        self.base_url = os.getenv("MINIMAX_BASE_URL", MINIMAX_BASE_URL).strip().rstrip("/")
        self.model = os.getenv("MINIMAX_MODEL", MINIMAX_MODEL).strip()

    def _safe_default(self, reason: str) -> dict:
        return {
            "enabled": self.enabled,
            "status": "skipped",
            "score_band": "N/A",
            "confidence": 0.0,
            "rationale": reason,
            "red_flags": [],
            "model": self.model,
        }

    def _extract_json(self, text: str) -> dict | None:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _extract_structured_fallback(self, text: str) -> dict:
        out = {
            "score_band": "N/A",
            "confidence": 0.0,
            "rationale": text[:220],
            "red_flags": [],
        }
        band_match = re.search(r"\b(A|B|C|REJECT)\b", text.upper())
        if band_match:
            out["score_band"] = band_match.group(1)
        conf_match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
        if conf_match:
            try:
                val = float(conf_match.group(1))
                out["confidence"] = min(max(val, 0.0), 1.0)
            except Exception:
                pass
        return out

    def _sanitize_text(self, text: str) -> str:
        lowered = text.lower()
        if "<think" in lowered:
            return "Reasoning redacted by adapter; use score/confidence/flags."
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        if not cleaned:
            cleaned = re.sub(r"<think>[\s\S]*", "", text, flags=re.IGNORECASE).strip()
        return cleaned[:220]

    def score_setup(self, payload: dict) -> dict:
        if not self.enabled:
            return self._safe_default("minimax disabled")
        if not self.api_key:
            return self._safe_default("missing MINIMAX_API_KEY")

        prompt = (
            "You are evaluating a Potter Box trade candidate.\n"
            "Return strict JSON only with keys: score_band, confidence, rationale, red_flags.\n"
            "score_band must be one of A,B,C,REJECT.\n"
            "confidence must be 0.0..1.0.\n"
            "rationale max 220 chars.\n"
            "red_flags must be an array of short strings.\n"
            f"Candidate data:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Output strict JSON only. Do not include reasoning tags, markdown, or prose."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.getenv("MINIMAX_TEMPERATURE", str(MINIMAX_TEMPERATURE))),
            "max_tokens": int(os.getenv("MINIMAX_MAX_OUTPUT_TOKENS", str(MINIMAX_MAX_OUTPUT_TOKENS))),
            "response_format": {"type": "json_object"},
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    json=body,
                    timeout=int(os.getenv("MINIMAX_TIMEOUT_SECONDS", str(MINIMAX_TIMEOUT_SECONDS))),
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    return {
                        "enabled": True,
                        "status": "error",
                        "score_band": "N/A",
                        "confidence": 0.0,
                        "rationale": f"minimax http {resp.status_code}",
                        "red_flags": [],
                        "model": self.model,
                    }

                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                parsed = self._extract_json(content)
                if not isinstance(parsed, dict):
                    parsed = self._extract_structured_fallback(content)
                if not isinstance(parsed, dict):
                    return {
                        "enabled": True,
                        "status": "ok_unparsed",
                        "score_band": "N/A",
                        "confidence": 0.0,
                        "rationale": str(content)[:220],
                        "red_flags": [],
                        "model": self.model,
                    }

                score_band = str(parsed.get("score_band", "N/A")).upper()
                if score_band not in {"A", "B", "C", "REJECT"}:
                    score_band = "N/A"
                try:
                    confidence = float(parsed.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                confidence = min(max(confidence, 0.0), 1.0)
                rationale = str(parsed.get("rationale", ""))[:220]
                rationale = self._sanitize_text(rationale)
                red_flags = parsed.get("red_flags", [])
                if not isinstance(red_flags, list):
                    red_flags = []
                red_flags = [str(x)[:60] for x in red_flags][:8]

                return {
                    "enabled": True,
                    "status": "ok",
                    "score_band": score_band,
                    "confidence": confidence,
                    "rationale": rationale,
                    "red_flags": red_flags,
                    "model": self.model,
                }
            except Exception as exc:
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                self.logger.error("MiniMax call failed: %s", exc)
                return {
                    "enabled": True,
                    "status": "error",
                    "score_band": "N/A",
                    "confidence": 0.0,
                    "rationale": f"minimax exception: {exc}",
                    "red_flags": [],
                    "model": self.model,
                }

        return self._safe_default("minimax exhausted retries")
