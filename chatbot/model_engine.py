import threading
import queue
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import tensorflow as tf
import tflearn

from config import MODEL_LOCK_TIMEOUT_SECONDS, MODEL_PREDICTION_TIMEOUT_SECONDS

tf.compat.v1.disable_eager_execution()

logger = logging.getLogger(__name__)


class IntentClassifier:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.model_path = base_dir / "model.tflearn"
        self.trained_data_path = base_dir / "trained_data"
        self.train_logs_dir = base_dir / "train_logs"
        self.model_ready = False
        self.model_lock = threading.Lock()

        self.words = []
        self.classes = []
        self._load_data()
        self.model = self._build_model()
        self._load_model()

    def _load_data(self) -> None:
        try:
            with open(self.trained_data_path, "rb") as f:
                data = pickle.load(f)
                self.words = data["words"]
                self.classes = data["classes"]
                self.train_x = data["train_x"]
                self.train_y = data["train_y"]
        except Exception as e:
            logger.exception("Failed to load trained data: %s", e)

    def _build_model(self) -> Any:
        if not hasattr(self, "train_x") or not hasattr(self, "train_y"):
            return None
        net = tflearn.input_data(shape=[None, len(self.train_x[0])])
        net = tflearn.fully_connected(net, 88)
        net = tflearn.fully_connected(net, 88)
        net = tflearn.fully_connected(net, len(self.train_y[0]), activation="softmax")
        net = tflearn.regression(net)
        return tflearn.DNN(net, tensorboard_dir=str(self.train_logs_dir))

    def _load_model(self) -> None:
        if self.model:
            try:
                self.model.load(str(self.model_path))
                self.model_ready = True
            except Exception as exc:
                logger.exception("Failed to load model: %s", exc)

    def classify_bag(self, bag_vector: np.ndarray) -> List[Tuple[str, float]]:
        if not self.model_ready:
            logger.error("Model is not ready yet.")
            return []

        if int(np.sum(bag_vector)) == 0:
            return []

        result_queue = queue.Queue(maxsize=1)

        def _predict_worker():
            acquired = self.model_lock.acquire(timeout=MODEL_LOCK_TIMEOUT_SECONDS)
            if not acquired:
                result_queue.put(("lock-timeout", None))
                return
            try:
                result_queue.put(("ok", self.model.predict([bag_vector])[0]))
            except Exception as exc:
                result_queue.put(("error", exc))
            finally:
                self.model_lock.release()

        worker = threading.Thread(target=_predict_worker, daemon=True)
        worker.start()
        worker.join(MODEL_PREDICTION_TIMEOUT_SECONDS)

        if worker.is_alive():
            logger.error(
                "Model prediction timed out after %.2fs.",
                MODEL_PREDICTION_TIMEOUT_SECONDS,
            )
            return []

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            logger.error("Model prediction returned no result.")
            return []

        if status == "lock-timeout":
            logger.error("Model lock timeout after %.2fs.", MODEL_LOCK_TIMEOUT_SECONDS)
            return []

        if status == "error":
            logger.exception("Model prediction failed: %s", payload)
            return []

        results = payload
        ranked_results = sorted(
            [(self.classes[i], float(score)) for i, score in enumerate(results)],
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked_results
