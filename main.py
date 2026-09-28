from src.data_ingation import data_loader
from src.data_modelbuilding import train_model
from src.modelevolation import (
    evaluate_model,
    evaluate_clustering_model
)


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
        cluster_pred
    ) = train_model(df)


    # =========================================================
    # 3. RANDOM FOREST EVALUATION
    # =========================================================

    print("\n========================================")
    print("RANDOM FOREST EVALUATION")
    print("========================================")

    rf_metrics = evaluate_model(
        y_test,
        rf_pred
    )


    # =========================================================
    # 4. XGBOOST EVALUATION
    # =========================================================

    print("\n========================================")
    print("XGBOOST EVALUATION")
    print("========================================")

    xgb_metrics = evaluate_model(
        y_test,
        xgb_pred
    )


    # =========================================================
    # 5. CLUSTERING MODEL EVALUATION
    # =========================================================

    print("\n========================================")
    print("K-MEANS CLUSTERING EVALUATION")
    print("========================================")

    clustering_metrics = evaluate_clustering_model(
        clustering_pipeline,
        X_test
    )




if __name__ == "__main__":
    main()