# Data Directory

This directory contains all datasets used in the wine quality classification project. The data is organized into two main subdirectories: `raw/` and `processed/`.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed data
│   └── winequality-red.csv
└── processed/              # Preprocessed datasets ready for model training
    ├── winequality-red-normalized-train.csv
    ├── winequality-red-normalized-test.csv
    ├── winequality-red-interactions-train.csv
    ├── winequality-red-interactions-test.csv
    ├── winequality-red-pca-train.csv
    ├── winequality-red-pca-test.csv
    └── winequality-red-oversampling-train.csv
```

## Raw Data

### `raw/winequality-red.csv`

The original dataset containing 1,599 wine samples with 11 physicochemical features and a target variable (quality score). This dataset is obtained from the UCI Machine Learning Repository (Cortez et al., 2009) and contains no missing values.

**Features:**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

**Target Variable:**
- Quality (discrete values from 3 to 8)

## Processed Data

All processed datasets are derived from the raw data and have been prepared for machine learning model training. To prevent data leakage, all preprocessing transformations (normalization, PCA, feature interactions, oversampling) are fitted **only on the training set** and then applied to both train and test sets. This ensures that test set statistics do not influence the preprocessing pipeline.

**Train-Test Split:** 80-20 split with stratified sampling to preserve class distribution.

### Normalized Dataset

**Baseline datasets** for all classification models.

- **`processed/winequality-red-normalized-train.csv`** (Training set)
- **`processed/winequality-red-normalized-test.csv`** (Test set)

These datasets contain:
- All 11 original features
- Outliers treated using the Interquartile Range (IQR) method (capped, not removed)
- Features standardized to have zero mean and unit variance (Z-score normalization)

**Outlier Treatment:** Outliers are identified using the IQR method and **capped** (not removed) to preserve sample size. Outlier bounds are calculated **only from the training set** and applied to both train and test sets.

**Normalization:** StandardScaler is fitted **only on the training set**. The mean and standard deviation from the training set are then used to transform both train and test sets, preventing data leakage.

This dataset serves as the baseline for comparing model performance across different algorithms (SVM, ANN, and Random Forest).

### Interactions Dataset

**Feature-engineered datasets** that extend the normalized dataset with interaction features.

- **`processed/winequality-red-interactions-train.csv`** (Training set)
- **`processed/winequality-red-interactions-test.csv`** (Test set)

These datasets contain:
- All 11 original standardized features from the normalized dataset
- 7 additional multiplicative interaction features based on strongly correlated pairs (|r| > 0.5):
  - Fixed acidity × Citric acid
  - Fixed acidity × Density
  - Total sulfur dioxide × Free sulfur dioxide
  - Fixed acidity × pH
  - Citric acid × Volatile acidity
  - Citric acid × pH
  - Alcohol × Density

**Total features:** 18 (11 original + 7 interactions)

**Why Interaction Features?** Strong pairwise correlations suggest that features may interact in ways that influence wine quality. Interaction features capture these synergistic relationships where the impact of one feature depends on the value of another.

### PCA Dataset

**Dimensionality-reduced datasets** created using Principal Component Analysis (PCA).

- **`processed/winequality-red-pca-train.csv`** (Training set)
- **`processed/winequality-red-pca-test.csv`** (Test set)

These datasets contain:
- Principal components derived from the normalized dataset
- 9 components retaining 95% of the total variance

PCA is fitted **only on the training set**. The fitted PCA model is then used to transform both train and test sets, ensuring that principal components are learned only from training data.

This dataset allows us to compare classification performance with and without dimensionality reduction.

### Oversampling Dataset

**Balanced training dataset** created using SMOTE (Synthetic Minority Oversampling Technique).

- **`processed/winequality-red-oversampling-train.csv`** (Training set)
- **Test set:** Use `winequality-red-normalized-test.csv` (unchanged)

This dataset contains:
- All 11 original standardized features
- Synthetic samples generated for minority classes to balance the class distribution
- SMOTE applied **only to the training set** after normalization

**Why Oversampling?** The dataset exhibits severe class imbalance (quality levels 3 and 8 represent less than 2% of samples). SMOTE generates synthetic samples for minority classes by interpolating between existing examples, helping models learn better decision boundaries for underrepresented classes.

**Important:** When using the oversampling training dataset (`winequality-red-oversampling-train.csv`), you **must** use the normalized test set (`winequality-red-normalized-test.csv`). The test set remains unchanged (no SMOTE applied) to provide an unbiased evaluation of model performance on the original class distribution. Using synthetic samples in the test set would invalidate the evaluation.

## Usage Notes

- **For baseline models:** Use normalized train/test datasets
- **For feature engineering experiments:** Use interactions train/test datasets
- **For dimensionality reduction experiments:** Use PCA train/test datasets
- **For class imbalance experiments:** Use `winequality-red-oversampling-train.csv` with `winequality-red-normalized-test.csv` (test set must be normalized, NOT oversampled)

All preprocessing steps are applied correctly to prevent data leakage, ensuring that test set statistics do not influence model training or preprocessing.

## Data Source

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.

UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/wine+quality
