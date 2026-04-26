import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/231130651/Eksperimen_SML_Michelle-Anditio.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = '231130651'
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN', '')

train_df = pd.read_csv('credit_risk_preprocessing/credit_risk_train.csv')
test_df = pd.read_csv('credit_risk_preprocessing/credit_risk_test.csv')

X_train = train_df.drop(columns=['loan_status'])
y_train = train_df['loan_status']
X_test = test_df.drop(columns=['loan_status'])
y_test = test_df['loan_status']

mlflow.set_experiment("credit_risk_modelling")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')

    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    mlflow.log_artifact('feature_importance.png')

    mlflow.sklearn.log_model(model, "random_forest_model")
    print("Training selesai!")