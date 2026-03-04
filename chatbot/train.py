import inspect
import json
import os
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
import tensorflow as tf
import tflearn
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer

tf.compat.v1.disable_eager_execution()

stemmer = GermanStemmer()
BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.2
MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 50
MIN_IMPROVEMENT = 1e-4


def get_path(file_name: str) -> str:
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(path, file_name).replace("\\", "/")


def ensure_nltk_data() -> None:
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.append(str(NLTK_DATA_DIR))
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for resource, name in {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
    }.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, download_dir=str(NLTK_DATA_DIR))


def get_json_path() -> str:
    return get_path("chat.json")


def build_training_data(dialogflow_data):
    stop_words = stopwords.words("german")
    ignore_words = ["?", ".", ","] + stop_words

    words = []
    classes = []
    documents = []

    for dialog in dialogflow_data["dialogflow"]:
        for pattern in dialog["synonym"]:
            tokenized = nltk.word_tokenize(pattern, language="german")
            words.extend(tokenized)
            documents.append((tokenized, dialog["intent"]))
            if dialog["intent"] not in classes:
                classes.append(dialog["intent"])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words] + [
        "weit",
        "and",
        "nicht",
    ]
    words = sorted(set(words))
    classes = sorted(set(classes))

    training = []
    output_empty = [0] * len(classes)

    for tokenized_pattern, intent in documents:
        pattern_words = [stemmer.stem(word.lower()) for word in tokenized_pattern]
        bag = [1 if w in pattern_words else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(intent)] = 1
        training.append([bag, output_row])

    random.shuffle(training)

    train_x = np.array([item[0] for item in training], dtype=np.float32)
    train_y = np.array([item[1] for item in training], dtype=np.float32)

    return words, classes, documents, train_x, train_y


def split_train_validation(train_x, train_y, validation_ratio=VALIDATION_RATIO):
    sample_count = len(train_x)
    if sample_count < 5:
        return train_x, train_y, np.array([]), np.array([])

    rng = random.Random(RANDOM_SEED)
    class_labels = np.argmax(train_y, axis=1)
    class_to_indices = {}
    for idx, label in enumerate(class_labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    train_idx = []
    val_idx = []
    for label in sorted(class_to_indices.keys()):
        class_indices = class_to_indices[label]
        rng.shuffle(class_indices)

        # Keep very small classes fully in training so no intent disappears there.
        if len(class_indices) <= 2:
            train_idx.extend(class_indices)
            continue

        proposed_val = max(1, int(len(class_indices) * validation_ratio))
        val_count = min(proposed_val, len(class_indices) - 1)
        val_idx.extend(class_indices[:val_count])
        train_idx.extend(class_indices[val_count:])

    if not val_idx:
        # Fallback for edge cases.
        all_indices = list(range(sample_count))
        rng.shuffle(all_indices)
        val_size = max(1, int(sample_count * validation_ratio))
        val_idx = all_indices[:val_size]
        train_idx = all_indices[val_size:]

    x_train = train_x[train_idx]
    y_train = train_y[train_idx]
    x_val = train_x[val_idx]
    y_val = train_y[val_idx]
    return x_train, y_train, x_val, y_val


def build_model(input_size, output_size):
    net = tflearn.input_data(shape=[None, input_size])
    net = tflearn.fully_connected(net, 88)
    net = tflearn.fully_connected(net, 88)
    net = tflearn.fully_connected(net, output_size, activation="softmax")
    net = tflearn.regression(net)
    return tflearn.DNN(net, tensorboard_dir=get_path("train_logs"))


def compute_validation_stats(model, features, labels):
    if len(features) == 0:
        return 0.0, float("inf")
    predictions = np.array(model.predict(features), dtype=np.float64)
    predictions = np.clip(predictions, 1e-9, 1.0 - 1e-9)
    prediction_labels = np.argmax(np.array(predictions), axis=1)
    target_labels = np.argmax(labels, axis=1)
    accuracy = float(np.mean(prediction_labels == target_labels))
    loss = float(-np.mean(np.sum(labels * np.log(predictions), axis=1)))
    return accuracy, loss


def train_with_early_stopping(model, x_train, y_train, x_val, y_val, model_path):
    best_metric = float("-inf")
    no_improvement_epochs = 0
    ran_epochs = 0

    batch_size = max(8, min(256, len(x_train)))

    for epoch in range(1, MAX_EPOCHS + 1):
        model.fit(
            x_train,
            y_train,
            n_epoch=1,
            batch_size=batch_size,
            show_metric=False,
            run_id="early-stop-run",
        )
        ran_epochs = epoch

        if len(x_val) > 0:
            val_accuracy, val_loss = compute_validation_stats(model, x_val, y_val)
        else:
            val_accuracy, val_loss = compute_validation_stats(model, x_train, y_train)

        # Lower validation loss is better; convert to maximization metric.
        metric = -val_loss

        if metric > best_metric + MIN_IMPROVEMENT:
            best_metric = metric
            no_improvement_epochs = 0
            model.save(model_path)
        else:
            no_improvement_epochs += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch:04d} val_acc={val_accuracy:.4f} val_loss={val_loss:.4f} "
                f"metric={metric:.4f} best={best_metric:.4f} "
                f"patience={no_improvement_epochs}/{EARLY_STOPPING_PATIENCE}"
            )

        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} (best_metric={best_metric:.4f})."
            )
            break

    if best_metric == float("-inf"):
        model.save(model_path)

    return ran_epochs, best_metric


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.set_random_seed(RANDOM_SEED)

    ensure_nltk_data()

    with open(get_json_path(), encoding="utf-8") as json_data:
        dialogflow_data = json.load(json_data)

    words, classes, documents, train_x, train_y = build_training_data(dialogflow_data)
    x_train, y_train, x_val, y_val = split_train_validation(train_x, train_y)

    print(f"{len(documents)} Docs")
    print(f"{len(classes)} Classes {classes}")
    print(f"{len(words)} Split words {words}")
    print(
        f"Train samples: {len(x_train)} | Validation samples: {len(x_val)} | "
        f"Max epochs: {MAX_EPOCHS} | Patience: {EARLY_STOPPING_PATIENCE}"
    )

    model = build_model(input_size=len(train_x[0]), output_size=len(train_y[0]))
    model_path = get_path("model.tflearn")

    ran_epochs, best_metric = train_with_early_stopping(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        model_path,
    )

    print(
        f"model created (epochs={ran_epochs}, best_metric={best_metric:.4f})"
    )

    pickle.dump(
        {
            "words": words,
            "classes": classes,
            "train_x": train_x.tolist(),
            "train_y": train_y.tolist(),
        },
        open(get_path("trained_data"), "wb"),
    )


if __name__ == "__main__":
    main()
