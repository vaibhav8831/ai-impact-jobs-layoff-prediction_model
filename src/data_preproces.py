# step 2 : Data Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocessing(df):
    """Split the data and fit a preprocessing/model pipeline.

    Returns the train/test partitions and predictions for the test partition.
    """
    if 'layoff_risk' not in df.columns:
        raise ValueError("DataFrame must contain a 'layoff_risk' column")

    df = df.drop_duplicates()

  # separate x and y 
  x = df.drop('layoff_risk', axis=1)
  y = df['layoff_risk']

  # identify categorical and numerical columns
  categorical_cols = x.select_dtypes(include=['object']).columns.tolist()
  numerical_cols = x.select_dtypes(exclude=['object']).columns.tolist()

  X_train, X_test, y_train, y_test = train_test_split(
      x, y, test_size=0.2, random_state=1
  )

  numerical_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median'))
  ])
  categorical_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
  ])
  preprocessor = ColumnTransformer(transformers=[
      ('numerical', numerical_pipeline, numerical_cols),
      ('categorical', categorical_pipeline, categorical_cols)
  ])
  model_pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', RandomForestClassifier(
          n_estimators=50, random_state=1, n_jobs=-1
      ))
  ])

  model_pipeline.fit(X_train, y_train)
  y_pred = model_pipeline.predict(X_test)

  return X_train,X_test,y_train,y_test