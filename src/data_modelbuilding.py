from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data_preproces import preprocessing


def train_model(df):

    # Get train/test data and preprocessor
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(df)

    # Complete ML pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=1
                )
            )
        ]
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = model_pipeline.predict(X_test)


    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1,
        eval_metric="logloss",
        n_jobs=-1
    )

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ]
    )

    # Train XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    xgb_pipeline.fit(X_train, y_train_encoded)

    # XGBoost prediction
    xgb_pred = label_encoder.inverse_transform(xgb_pipeline.predict(X_test))

    return model_pipeline, X_test, y_test, y_pred,xgb_pipeline, xgb_pred