# save_model_to_registry.py
import os
from argparse import ArgumentParser
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import wandb
import joblib
os.environ["WANDB_API_KEY"] = "local-86e9b65b97739dbc0a23bac403285be06b06887a"
os.environ["WANDB_BASE_URL"] = "http://localhost:8080" 
wandb.init(project="mlops", name="test")

# 1. get data
db_connect = psycopg2.connect(
    user="myuser",
    password="mypassword",
    host="localhost",
    port=5432,
    database="mydatabase",
)
df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connect)

X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 2. model development and train
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
wandb.log({"train_accuracy": train_acc, "valid_accuracy": valid_acc})

model_filename = "svm_model.pkl"
joblib.dump(model_pipeline, model_filename)
wandb.save(model_filename)

wandb.finish()