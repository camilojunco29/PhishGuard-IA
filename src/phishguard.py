import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEED = 42
PHRASES = ("verifique su cuenta", "actualice sus datos", "confirme su contraseña")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def normalize(text):
    text = re.sub(r"(?:https?://|www\.)\S+", " urltoken ", text.lower())
    return " ".join(text.split())


class KeywordRules(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.array([sum(phrase in text for phrase in PHRASES) for text in X])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def load_data(path):
    grouped = defaultdict(list)
    raw = Counter()

    with path.open(encoding="utf-8-sig", newline="") as stream:
        delimiter = csv.Sniffer().sniff(stream.readline(), delimiters=",;").delimiter
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter=delimiter)
        fields = {name.strip().lower(): name for name in reader.fieldnames or []}
        if not {"subject", "body", "label"} <= fields.keys():
            raise ValueError("El CSV debe contener subject, body y Label.")
        for row_id, row in enumerate(reader, 1):
            label = int(row[fields["label"]])
            if label not in (0, 1):
                raise ValueError(f"Etiqueta binaria inválida en el registro {row_id}")
            raw[label] += 1
            text = normalize(" ".join(row[fields[field]] or "" for field in ("subject", "body")))
            if text:
                grouped[text].append((row_id, label))

    unique = [(text, *items[0]) for text, items in grouped.items()
              if len({label for _, label in items}) == 1]
    if not unique:
        raise ValueError("El CSV no contiene textos utilizables.")

    # A shared character-shingle profile can reflect a repeated template.
    seen = set()
    retained = []
    profiles = []
    for text, row_id, label in unique:
        shingles = {text[i:i+5] for i in range(len(text)-4)}
        profile = frozenset(shingles)
        if not profile:
            retained.append((text, row_id, label))
            continue
        # Compare profiles before partitioning and drop both sides of cross-label matches.
        for i, (previous, previous_label) in enumerate(profiles):
            if min(len(profile), len(previous)) < 0.9 * max(len(profile), len(previous)):
                continue
            common = len(profile & previous)
            if common >= 0.9 * (len(profile) + len(previous) - common):
                seen.add(len(profiles))
                if previous_label != label:
                    seen.add(i)
        profiles.append((profile, label))
        retained.append((text, row_id, label))
    # Keep profile indexes aligned even when a short text has no shingles.
    indices = iter(range(len(profiles)))
    clean = [item for item in retained if len(item[0]) < 5 or next(indices) not in seen]
    return raw, clean


def experiment(path, output):
    from sklearn.metrics import get_scorer, make_scorer, precision_score

    raw, rows = load_data(path)
    texts, ids, labels = map(np.array, zip(*rows))
    train, test = train_test_split(np.arange(len(labels)), test_size=0.15,
                                   random_state=SEED, stratify=labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparison = {}
    candidates = {}
    scoring = {name: get_scorer(name) for name in METRICS}
    scoring["precision"] = make_scorer(precision_score, zero_division=0)
    models = {
        "mayoritario": (DummyClassifier(strategy="most_frequent"), {}),
        "reglas": (KeywordRules(), {}),
        "naive_bayes": (MultinomialNB(), {"model__alpha": [0.1, 0.5, 1.0]}),
        "regresion_logistica": (LogisticRegression(max_iter=2000, random_state=SEED),
                               {"model__C": [0.1, 1.0, 10.0]}),
    }
    for name, (model, grid) in models.items():
        if grid:
            pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("model", model)])
            grid.update({"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2]})
            search = GridSearchCV(pipeline, grid, scoring=scoring, cv=folds,
                                  refit="f1", error_score="raise")
            search.fit(texts[train], labels[train])
            model = search.best_estimator_
            index = search.best_index_
            summary = {m: [float(search.cv_results_[f"{stat}_test_{m}"][index])
                           for stat in ("mean", "std")] for m in METRICS}
            params = search.best_params_
            candidates[name] = model
        else:
            scores = cross_validate(model, texts[train], labels[train], cv=folds,
                                    scoring=scoring, error_score="raise")
            summary = {m: [float(scores[f"test_{m}"].mean()), float(scores[f"test_{m}"].std())]
                       for m in METRICS}
            params = {}
            model.fit(texts[train], labels[train])
        comparison[name] = {"validacion": summary, "parametros": params}
        models[name] = (model, grid)

    winner = max(candidates, key=lambda name: comparison[name]["validacion"]["f1"][0])
    for name, (model, _) in models.items():
        predicted = model.predict(texts[test])
        comparison[name]["prueba"] = {
            m: float(scorer(model, texts[test], labels[test])) for m, scorer in scoring.items()
        }
        comparison[name]["matriz"] = confusion_matrix(labels[test], predicted, labels=[0, 1]).tolist()

    best = candidates[winner]
    predictions = best.predict(texts[test])
    errors = [
        {"registro_csv": int(ids[index]), "real": int(labels[index]), "prediccion": int(pred)}
        for index, pred in zip(test, predictions) if pred != labels[index]
    ]
    metrics = comparison[winner]["prueba"]
    gain = metrics["f1"] - max(comparison[name]["prueba"]["f1"] for name in ("reglas", "mayoritario"))
    report = {
        "semilla": SEED, "corpus": dict(raw), "limpio": dict(Counter(map(int, labels))),
        "entrenamiento": dict(Counter(map(int, labels[train]))),
        "prueba": dict(Counter(map(int, labels[test]))), "modelos": comparison,
        "seleccion": winner, "ventaja_f1": gain,
        "cumple": bool(metrics["f1"] >= 0.85 and metrics["recall"] >= 0.85 and gain >= 0.05),
        "errores": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "resultados.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(best, output / "modelo.joblib")
    print(f"Resultados escritos en {output}")


def predict(model_path):
    model = joblib.load(model_path)
    text = normalize(input("Asunto: ") + " " + input("Cuerpo en una línea: "))
    if not text:
        raise SystemExit("Ingresa un asunto o un cuerpo para clasificar.")
    label = int(model.predict([text])[0])
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["model"]
    vector = vectorizer.transform([text]).tocsr()
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("El modelo no tiene las clases esperadas.")
    weights = (classifier.coef_[0] if isinstance(classifier, LogisticRegression)
               else classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0])
    contributions = vector.data * weights[vector.indices] * (1 if label else -1)
    terms = vectorizer.get_feature_names_out()
    strongest = [terms[vector.indices[i]] for i in np.argsort(-contributions)[:5]
                 if contributions[i] > 0]
    print("Predicción:", "phishing" if label else "legítimo")
    print("Términos que apoyan la predicción:", ", ".join(strongest) or "ninguno identificado")
    print("Usa este resultado como apoyo a la revisión del correo.")


def main():
    parser = argparse.ArgumentParser(description="Clasificación de correos en español")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("entrenar")
    train.add_argument("csv", type=Path)
    train.add_argument("--salida", type=Path, default=Path("experiments"))
    classify = commands.add_parser("clasificar")
    classify.add_argument("modelo", type=Path)
    args = parser.parse_args()
    if args.command == "entrenar":
        experiment(args.csv, args.salida)
    else:
        predict(args.modelo)


if __name__ == "__main__":
    main()

