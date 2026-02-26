import os
import time
import logging
from pathlib import Path
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None)


class EmbeddingGenerator:

    _effective_batch_cache = None

    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.url_main = os.getenv("EMBEDDING_MODEL_URL")
        self.model = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
        self.timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

        if not self.api_key or not self.url_main:
            raise ValueError("Missing API_KEY or EMBEDDING_MODEL_URL in environment.")

        self.max_batch = max(1, int(os.getenv("EMBEDDING_MAX_BATCH", "17")))
        self.min_batch = max(1, int(os.getenv("EMBEDDING_MIN_BATCH", "1")))
        self.min_batch = min(self.min_batch, self.max_batch)

        self.url_fallback = os.getenv("EMBEDDING_FALLBACK_URL") or None
        self.model_fallback = os.getenv("EMBEDDING_FALLBACK_MODEL") or None
        self.url_main_alt = self._swap_host(self.url_main)

        logger.info("[Embedding] Using %s | model=%s", self.url_main, self.model)

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _swap_host(self, url: str) -> str:
        try:
            if "generative.engine.capgemini.com" in url:
                return (
                    url.replace("//openai.", "//api.")
                    if "//openai." in url
                    else url.replace("//api.", "//openai.")
                    if "//api." in url
                    else url
                )
        except Exception:
            pass
        return url

    def _is_proxy_403(self, msg: str) -> bool:
        if not msg:
            return False
        msg = msg.lower()
        return "proxy request failed" in msg and "403" in msg and "forbidden" in msg

    def _post_once(self, url: str, model: str, inp, label: str):
        payload = {"model": model, "input": inp}
        t0 = time.time()
        try:
            res = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
        except requests.exceptions.RequestException as e:
            logger.error("[Embedding] %s error (%s)", label, str(e)[:200])
            raise ValueError(str(e))

        if not isinstance(data, dict) or "data" not in data:
            logger.error("[Embedding] Invalid response: %s", str(data)[:300])
            raise ValueError(f"Bad response: {str(data)[:200]}")

        logger.info("[Embedding] %s OK in %.2fs (%d items)", label, time.time() - t0, len(data["data"]))
        return data

    def _request_with_failover(self, inp, label: str):
        try:
            return self._post_once(self.url_main, self.model, inp, label)
        except ValueError as e:
            if self._is_proxy_403(str(e)) and not (self.url_fallback or self.model_fallback):
                raise
            last_err = e

        if self.url_fallback or self.model_fallback:
            fb_url = self.url_fallback or self.url_main
            fb_model = self.model_fallback or self.model
            try:
                logger.warning("[Embedding] Trying fallback")
                return self._post_once(fb_url, fb_model, inp, label + " [fallback]")
            except ValueError as e2:
                last_err = e2

        if self.url_main_alt != self.url_main:
            try:
                logger.warning("[Embedding] Trying alt-host")
                return self._post_once(self.url_main_alt, self.model, inp, label + " [alt]")
            except ValueError as e3:
                last_err = e3

        raise last_err

    def generate_embedding(self, text: str):
        data = self._request_with_failover(text.strip(), "single")
        return data["data"][0]["embedding"]

    def generate_embeddings_batch(self, texts):
        if not texts:
            return []

        texts = [t.strip() for t in texts]

        if len(texts) <= self.min_batch:
            data = self._request_with_failover(texts, f"batch[{len(texts)}]")
            return [item["embedding"] for item in data["data"]]

        all_embeds = []
        current = EmbeddingGenerator._effective_batch_cache or self.max_batch
        i = 0

        while i < len(texts):
            batch = texts[i:i + current]
            label = f"batch[{len(batch)}]"
            try:
                data = self._request_with_failover(batch, label)
                all_embeds.extend([item["embedding"] for item in data["data"]])
                i += len(batch)
                EmbeddingGenerator._effective_batch_cache = current
            except ValueError:
                if current > self.min_batch:
                    current = max(self.min_batch, current - 1)
                    logger.warning("[Embedding] Reducing batch size to %d", current)
                else:
                    raise

        return all_embeds
