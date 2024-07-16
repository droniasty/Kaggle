import numpy as np
import pandas as pd
import os

import keras
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import ParameterGrid, train_test_split
from tensorflow_decision_forests.keras.core_inference import pd_dataframe_to_tf_dataset

# 0.77990


# Load the dataset.
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

pd.set_option('display.max_columns', None)

print(train_df.head(50))

# Preprocess the dataset.

# Drop the columns name and ticket.
train_df = train_df.drop(columns=["Name", "Ticket", "Cabin", "Fare"])
test_df = test_df.drop(columns=["Name", "Ticket", "Cabin", "Fare"])


# count and display the number of missing values in each column.
print(train_df.isnull().sum())

# fill the missing values in the column "Age" with the -1 value.
train_df["Age"] = train_df["Age"].fillna(-1)
test_df["Age"]  = test_df["Age"].fillna(-1)

# encode the sex column.
train_df["Sex"] = train_df["Sex"].map({"male":1, "female":0}).astype(int)
test_df["Sex"] = test_df["Sex"].map({"male":1, "female":0}).astype(int)


print(train_df.head(50))

#find unhashable collumns
for column in train_df.columns:
    # Getting unique types of all values in the column
    unique_types = set(type(value) for value in train_df[column])
    
    # If more than one type is found, report the column
    if len(unique_types) > 1:
        print(f"Column {column} has multiple types: {unique_types}")

# Split the dataset into a training and a validation dataset.
train_ds = train_df.sample(frac=0.8, random_state=0)
valid_ds = train_df.drop(train_ds.index)

# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds, label="Survived")
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds, label="Survived")

# Split the dataset into features and labels.
x_train, y_train, x_val, y_val = train_test_split(train_df.drop(columns=["Survived"]), train_df["Survived"], test_size=0.2, random_state=42)
x_train = pd_dataframe_to_tf_dataset(x_train)

model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)

param_grid = {
    "num_trees": range(1, 100),
    "max_depth": range(1, 10),
}

# find the best hyperparameters
best_score = float('-inf')
best_params = {}

for params in ParameterGrid(param_grid):
    model = tfdf.keras.RandomForestModel(**params)
    model.fit(train_ds)
    score = model.evaluate(valid_ds)
    
    if score > best_score:
        best_score = score
        best_params = params

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

predictions = model.predict(test_ds)
predictions = predictions.flatten()

print(predictions.shape)

kaggle_predictions = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": (predictions >= 0.5).astype(int)
})   

# Save the predictions to a CSV file.
kaggle_predictions.to_csv("submission.csv", index=False)



