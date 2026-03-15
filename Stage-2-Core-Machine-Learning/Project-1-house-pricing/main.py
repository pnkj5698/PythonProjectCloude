import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("train.csv")

# Before get_dummies
print("Before get_dummies:")
print(df[["MSZoning", "Street", "LotShape"]].head())
print("Shape:", df.shape)

# Drop columns with too many missing values
df.drop(
    ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis=1, inplace=True
)

# Fill numeric missing values with median
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())

# Fill categorical missing values with mode
for col in [
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
    "Electrical",
]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert text columns to numbers
df = pd.get_dummies(df, drop_first=True)

# After get_dummies
print("\nAfter get_dummies:")
print("Shape:", df.shape)

# Print all MSZoning related columns
mszoning_cols = [col for col in df.columns if "MSZoning" in col]
print("MSZoning columns:", mszoning_cols)

# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error: £{mae:,.0f}")
print(f"Average house price: £{y.mean():,.0f}")


# Project ends
