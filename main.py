import json
import pickle
from pathlib import Path

from sklearn.metrics import f1_score

from src.data_ingation import data_loader
from src.data_modelbuilding import train_model
from src.modelevolation import evaluate_clustering_model, evaluate_model

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_risk_mapping(classifier, clustering_pipeline, label_encoder, X_test):
    """Return a stable risk mapping for downstream web or API consumers."""
    cluster_model = clustering_pipeline.named_steps["model"]
    centers = getattr(cluster_model, "cluster_centers_", [])
    cluster_risk = {
        int(index): float((index + 1) / max(len(centers), 1))
        for index in range(len(centers))
    }
    risk_mapping = {
        "Low": 0.33,
        "Medium": 0.66,
        "High": 0.90,
    }
    return risk_mapping, cluster_risk


def build_feature_schema(df, X_test):
    """Return column metadata that can be consumed by a front-end app."""
    return [
        {"name": column, "dtype": str(X_test[column].dtype)}
        for column in X_test.columns
    ]


def save_pickle(value, filename):
    with open(MODEL_DIR / filename, "wb") as model_file:
        pickle.dump(value, model_file)


def main():
    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    df = data_loader()
    print("\nData Loaded Successfully")
    print("Shape:", df.shape)

    # =========================================================
    # 2. TRAIN MODELS
    # =========================================================
    (
        rf_pipeline,
        xgb_pipeline,
        clustering_pipeline,
        label_encoder,
        X_test,
        y_test,
        rf_pred,
        xgb_pred,
        cluster_pred,
    ) = train_model(df)

    # =========================================================
    # 3. RANDOM FOREST EVALUATION
    # =========================================================
    print("\n========================================")
    print("RANDOM FOREST EVALUATION")
    print("========================================")
    evaluate_model(y_test, rf_pred)

    # =========================================================
    # 4. XGBOOST EVALUATION
    # =========================================================
    print("\n========================================")
    print("XGBOOST EVALUATION")
    print("========================================")
    evaluate_model(y_test, xgb_pred)

    # =========================================================
    # 5. CLUSTERING MODEL EVALUATION
    # =========================================================
    print("\n========================================")
    print("K-MEANS CLUSTERING EVALUATION")
    print("========================================")
    evaluate_clustering_model(clustering_pipeline, X_test)

    # =========================================================
    # 6. SAVE MODELS FOR THE WEBSITE
    # =========================================================
    rf_f1 = f1_score(y_test, rf_pred, average="weighted")
    xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
    if xgb_f1 >= rf_f1:
        best_name, best_classifier = "XGBoost", xgb_pipeline
    else:
        best_name, best_classifier = "Random Forest", rf_pipeline

    print(
        f"\nBest classifier for the website: {best_name} "
        f"(weighted F1: RF={rf_f1:.3f}, XGB={xgb_f1:.3f})"
    )

    risk_mapping, cluster_risk = build_risk_mapping(
        best_classifier,
        clustering_pipeline,
        label_encoder,
        X_test,
    )
    print("Cluster -> risk level:", risk_mapping)

    save_pickle(best_classifier, "classifier.pkl")
    save_pickle(label_encoder, "label_encoder.pkl")
    save_pickle(
        {
            "model": clustering_pipeline,
            "risk_mapping": risk_mapping,
            "cluster_risk_scores": {
                int(cluster_id): float(risk) for cluster_id, risk in cluster_risk.items()
            },
        },
        "clustering.pkl",
    )
    with open(MODEL_DIR / "features.json", "w", encoding="utf-8") as feature_file:
        json.dump({"columns": build_feature_schema(df, X_test)}, feature_file, indent=2)

    print(
        f"Saved classifier.pkl, label_encoder.pkl, clustering.pkl, features.json "
        f"to {MODEL_DIR}/"
    )


if __name__ == "__main__":
    main()