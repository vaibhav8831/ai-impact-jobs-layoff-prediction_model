from src.data_ingation import data_loader
from src.data_modelbuilding import train_model
import src.modelevolation as evaluation


def main():

    # Load data
    df = data_loader()

    # Train model
    model_pipeline, X_test, y_test, y_pred, xgb_pipeline, xgb_pred = train_model(df)

    print("Model training completed.")
    print("Test samples:", len(X_test))
    print("Predictions:", y_pred[:10])
    print("XGBoost predictions:", xgb_pred[:10])

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


if __name__ == "__main__":
    main()