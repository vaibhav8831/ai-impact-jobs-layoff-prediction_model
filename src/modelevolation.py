from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.sparse import issparse


# =========================================================
# CLASSIFICATION EVALUATION
# =========================================================

def evaluate_model(y_test, y_pred):

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("Classification Model Evaluation")
    print("--------------------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    return accuracy, precision, recall, f1


# =========================================================
# CLUSTERING EVALUATION
# =========================================================

def evaluate_clustering_model(
    clustering_pipeline,
    X_test
):

    # Predict clusters
    cluster_labels = clustering_pipeline.predict(X_test)

    # Apply fitted preprocessing
    X_test_transformed = clustering_pipeline.named_steps[
        "preprocessor"
    ].transform(X_test)
    if issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()

    # Clustering metrics
    silhouette = silhouette_score(
        X_test_transformed,
        cluster_labels
    )

    calinski = calinski_harabasz_score(
        X_test_transformed,
        cluster_labels
    )

    davies = davies_bouldin_score(
        X_test_transformed,
        cluster_labels
    )

    # K-Means model
    kmeans_model = clustering_pipeline.named_steps["model"]

    inertia = kmeans_model.inertia_

    print("\nClustering Model Evaluation")
    print("--------------------------------")
    print("Silhouette Score        :", silhouette)
    print("Calinski-Harabasz Score :", calinski)
    print("Davies-Bouldin Score    :", davies)
    print("Inertia                 :", inertia)

    return (
        silhouette,
        calinski,
        davies,
        inertia
    )