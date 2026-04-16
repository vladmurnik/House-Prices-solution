# 🏠 House Prices: Advanced Regression Techniques — Solution

## 📌 Overview

This project contains a simple and effective solution for the Kaggle competition **“House Prices: Advanced Regression Techniques”**.
The goal is to predict house sale prices based on various features describing residential homes.

The solution uses:

* **CatBoostRegressor** for handling both numerical and categorical features
* Minimal preprocessing
* A clean and reproducible pipeline

---

## 📂 Project Structure

```
.
├── out.csv          # Generated submission file
├── main.py          # Training & prediction script
├── requirements.txt # Requirements for libraries
└── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/vladmurnik/House-Prices-solution
cd House-Prices-solution
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn catboost
```

---

## 🚀 Usage

Run the script:

```bash
python main.py
```

After execution, the file `out.csv` will be generated — ready for submission to Kaggle.

---

## 🧠 Approach

### 1. Data Loading

```python
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
```

---

### 2. Feature Cleaning

Columns with too many missing values or low importance are removed:

* PoolQC
* MiscFeature
* Alley
* Fence
* MasVnrType
* FireplaceQu
* LotFrontage

---

### 3. Missing Values Handling

#### Numerical features:

* Filled with **mean values**

#### Categorical features:

* Filled with `"missing"` label

---

### 4. Feature Separation

```python
X_train = train.drop("SalePrice", axis=1)
y_train = train["SalePrice"]
```

Categorical features are detected automatically:

```python
cat_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()
```

---

### 5. Model

We use **CatBoostRegressor**, which is well-suited for tabular data and handles categorical features natively.

```python
pipeline = Pipeline([
    ("model", CatBoostRegressor(verbose=0))
])
```

Training:

```python
model = pipeline.fit(
    X_train,
    y_train,
    model__cat_features=cat_features
)
```

---

### 6. Prediction

```python
y_pred = model.predict(X_test)
```

---

### 7. Submission File

```python
pred_df = pd.DataFrame({
    "Id": X_test["Id"].values,
    "SalePrice": y_pred
})
pred_df.to_csv("out.csv", index=False)
```

---

## 📊 Notes

* No feature scaling is required (CatBoost handles it internally)
* No encoding needed for categorical features
* Simple baseline model — can be improved with:

  * Hyperparameter tuning
  * Feature engineering
  * Cross-validation
  * Log-transform of target (`SalePrice`)

---

## 🔧 Possible Improvements

* Use `log1p(SalePrice)` for better RMSLE score
* Add cross-validation (`KFold`)
* Tune parameters:

  * depth
  * learning_rate
  * iterations
* Handle outliers explicitly
* Feature engineering (e.g., total area, age of house)

---

## 📈 Result

This baseline provides a solid starting point and achieves a reasonable score without heavy tuning.

---

## 🏁 Submission

Upload `out.csv` to Kaggle:

* Ensure correct format:

  * Columns: `Id`, `SalePrice`
  * No index column

---

## 📜 License

Free to use and modify for learning purposes.
