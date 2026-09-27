from sklearn import preprocessing

from src.data_ingation import data_loader
from src.data_preproces import preprocessing


def main():
    df = data_loader()
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(df.head())

    X_train,X_test,y_train,y_test = preprocessing(df)
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set: X={X_test.shape}, y={y_test.shape}")
    



    main()