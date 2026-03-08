import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic/train.csv")

# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df.drop("Cabin", axis=1, inplace=True)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Verify no missing values remain
# print(df.isnull().sum())

# Pick features to train on
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Prepare test data
test = pd.read_csv("titanic/test.csv")
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode()[0])
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

test_predictions = model.predict(test[features])

submission = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": test_predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission file created!")
