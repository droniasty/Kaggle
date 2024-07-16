import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split


# Load the dataset.
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_df = train_df.drop(columns=["Name", "Ticket", "Cabin", "Fare",  "SibSp", "Parch", "PassengerId"])
test_df = test_df.drop(columns=["Name", "Ticket", "Cabin", "Fare", "SibSp", "Parch", ])

train_df["Age"] = train_df["Age"].fillna(-1)
test_df["Age"]  = test_df["Age"].fillna(-1)

# fill the missing values in the column "Age" with the -1 value.
train_df["Age"] = train_df["Age"].fillna(-1)
test_df["Age"]  = test_df["Age"].fillna(-1)

# encode the sex column.
train_df["Sex"] = train_df["Sex"].map({"male":1, "female":0}).astype(int)
test_df["Sex"] = test_df["Sex"].map({"male":1, "female":0}).astype(int)

# fill the missing values in the column "Embarked" with the most common value.
common_value = "S"
train_df["Embarked"] = train_df["Embarked"].fillna(common_value)

# convert the column "Embarked" to a numerical value.
train_df["Embarked"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
test_df["Embarked"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)


# Split the dataset into a training and a validation dataset.
train_ds = train_df.sample(frac=0.8, random_state=0)
valid_ds = train_df.drop(train_ds.index)

# create the xgboost model and train it.
model = xgb.XGBClassifier()

model.fit(train_ds.drop(columns=["Survived"]), train_ds["Survived"])

# evaluate the model.

predictions = model.predict(valid_ds.drop(columns=["Survived"]))
accuracy = np.mean(predictions == valid_ds["Survived"])

print(f"Accuracy: {accuracy}")

# predict the test dataset.
test_predictions = model.predict(test_df.drop(columns=["PassengerId"]))
test_df["Survived"] = test_predictions

test_df[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)