from src.data_ingation import data_loader
from src.data_modelbuilding import train_model
import src.modelevolation as evaluation
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib



def main():

    # Load data
    df = data_loader()

    # Train model
    (
        model_pipeline,
        X_test,
        y_test,
        y_pred,
        xgb_pipeline,
        xgb_pred,
        clustering_pipeline,
        cluster_pred,
    ) = train_model(df)

    print("Model training completed.")
    print("Test samples:", len(X_test))
    print("Predictions:", y_pred[:10])
    print("XGBoost predictions:", xgb_pred[:10])
    print("K-means clusters:", cluster_pred[:10])

    print("Random Forest evaluation:")
    evaluation.evaluate_model(
        y_test,
        y_pred
    )

    print("XGBoost evaluation:")
    evaluation.evaluate_model(
        y_test,
        xgb_pred
    )
# 5. CLUSTERING MODEL EVALUATION
    # =========================================================

    print("\n========================================")
    print("K-MEANS CLUSTERING EVALUATION")
    print("========================================")

    clustering_metrics = evaluation.evaluate_clustering_model(
        clustering_pipeline,
        X_test
    )

    

if __name__ == "__main__":
    main()

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model = joblib.load("model.pkl")  # Load the pre-trained model pipeline