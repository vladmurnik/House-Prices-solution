import pandas as pd
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

# Load data
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

# Drop columns with little useful information
columns_to_drop = [
    "PoolQC", "MiscFeature", "Alley", "Fence",
    "MasVnrType", "FireplaceQu", "LotFrontage"
]
test = test.drop(columns=columns_to_drop)
train = train.drop(columns=columns_to_drop)

# Fill missing values in numeric columns
train = train.fillna(train.mean(numeric_only=True))
test = test.fillna(test.mean(numeric_only=True))

X_train, y_train = train.drop("SalePrice", axis=1), train["SalePrice"]
X_test = test.copy()

# Detect categorical features
cat_features = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

# Fill missing values in numeric features
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# Fill missing values in categorical features
X_train[cat_features] = X_train[cat_features].fillna("missing")
X_test[cat_features] = X_test[cat_features].fillna("missing")

# Train the model
pipeline = Pipeline([
    ("model", CatBoostRegressor(verbose=0))
])
model = pipeline.fit(X_train, y_train, model__cat_features=cat_features)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a submission file
pred_df = pd.DataFrame({
    "Id": X_test["Id"].values,
    "SalePrice": y_pred
})
pred_df.to_csv("out.csv", index=False)