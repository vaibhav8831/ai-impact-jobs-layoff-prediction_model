import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def preprocessing(df):
    """
    Clean data, split into train/test sets,
    and create the preprocessing pipeline.
    """

    # Work on a copy so the original dataframe is not modified
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove unnecessary columns
    df = df.drop(
        columns=["Age", "AI_Adoption_Level"],
        errors="ignore"
    )

    # Separate X and y
    X = df.drop(columns=["Layoff_Risk"])
    y = df["Layoff_Risk"]

    # Identify numerical and categorical columns
    numerical_data = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_data = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=1,
        stratify=y
    )

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # Combine numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical",
            numerical_pipeline,
            numerical_data),
            ("categorical",
            categorical_pipeline,
            categorical_data)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor