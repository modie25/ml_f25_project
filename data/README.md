# Data Directory

This directory contains all datasets used in the wine quality classification project. The data is organized into two main subdirectories: `raw/` and `processed/`.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed data
│   └── winequality-red.csv
└── processed/              # Preprocessed datasets ready for model training
    ├── winequality-red-normalized.csv
    ├── winequality-red-interactions.csv
    └── winequality-red-pca.csv
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

All processed datasets are derived from the raw data and have been prepared for machine learning model training. Each dataset serves a specific purpose in our analysis.

### `processed/winequality-red-normalized.csv`

**Baseline dataset** for all classification models. This dataset contains:
- All 11 original features
- Outliers treated using the Interquartile Range (IQR) method (capped, not removed)
- Features standardized to have zero mean and unit variance (Z-score normalization)

**Outlier Treatment (Capped, Not Removed):**
Outliers are identified using the IQR method, where values outside the range [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are considered outliers. Rather than removing these outliers (which would reduce the sample size from 1,599), we **cap** them by setting extreme values to the boundary limits. This approach:
- Preserves the full sample size (1,599 samples) for model training
- Prevents outliers from distorting the mean and standard deviation used in standardization
- Maintains the dataset's representativeness while reducing the influence of extreme values
- Ensures all samples contribute to model training, which is particularly important given the class imbalance in the target variable

This dataset serves as the baseline for comparing model performance across different algorithms (SVM, ANN, and Random Forest). All features are on the same scale, ensuring fair comparison and preventing features with larger numerical ranges from dominating the learning process.

### `processed/winequality-red-interactions.csv`

**Feature-engineered dataset** that extends the normalized dataset with interaction features. This dataset contains:
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

**Why Interaction Features?**
The correlation analysis revealed several strong pairwise relationships between features, suggesting that these variables may interact in ways that influence wine quality. While linear models can capture individual feature effects, they may miss how the combination of features together affects the outcome. For example, the effect of fixed acidity on wine quality may depend on the level of citric acid present—the relationship between acidity and quality may change when both variables are considered together rather than independently. These interaction features allow us to capture these synergistic or conditional relationships, where the impact of one feature depends on the value of another.

This dataset allows us to evaluate whether explicitly modeling feature interactions improves classification performance compared to the baseline normalized dataset, particularly for algorithms that may benefit from these engineered features.

### `processed/winequality-red-pca.csv`

**Dimensionality-reduced dataset** created using Principal Component Analysis (PCA). This dataset contains:
- Principal components derived from the normalized dataset
- Components ordered by the amount of variance they explain

**Variance Retention:**
- **7 components:** Retain 90% of the total variance
- **9 components:** Retain 95% of the total variance

Both configurations (7 and 9 components) can be used for model training, depending on the desired balance between dimensionality reduction and variance retention. The PCA-transformed dataset allows us to compare classification performance with and without dimensionality reduction, assessing whether reducing the feature space improves or hinders model accuracy.

## Usage Notes

- **For baseline models:** Use `winequality-red-normalized.csv`
- **For feature engineering experiments:** Use `winequality-red-interactions.csv`
- **For dimensionality reduction experiments:** Use `winequality-red-pca.csv` (with either 7 or 9 components)

All processed datasets maintain the same target variable (quality) and sample size (1,599 samples) as the original dataset.

## Data Source

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.

UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/wine+quality
