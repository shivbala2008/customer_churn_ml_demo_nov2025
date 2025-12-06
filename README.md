# Customer Churn ML Demo

A comprehensive machine learning pipeline for predicting customer churn in subscription-based services. This project demonstrates a complete data science workflow from raw data cleaning to feature engineering, preparing data for predictive modeling.

## 📋 Table of Contents

- [Overview](#overview)
- [Use Case](#use-case)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Pipeline Process](#pipeline-process)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Feature Engineering Details](#feature-engineering-details)
- [Next Steps](#next-steps)

---

## 🎯 Overview

This project implements a **customer churn prediction pipeline** for a subscription-based business. Customer churn refers to when customers cancel their subscriptions or stop using a service. Understanding and predicting churn is critical for businesses to:

- Identify at-risk customers before they leave
- Implement targeted retention strategies
- Reduce revenue loss
- Improve customer lifetime value (CLV)

The pipeline processes raw customer data through systematic cleaning and feature engineering stages to create a machine learning-ready dataset.

---

## 💼 Use Case

### Business Problem

A subscription-based company (e.g., SaaS, streaming service, telecom) wants to:

1. **Predict which customers are likely to churn** (cancel their subscription)
2. **Understand the factors** that contribute to customer churn
3. **Take proactive measures** to retain high-value customers

### Solution Approach

This pipeline prepares customer data by:

- Cleaning messy real-world data (missing values, duplicates, inconsistencies)
- Engineering meaningful features that capture customer behavior patterns
- Creating risk indicators and engagement metrics
- Preparing data for machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)

---

## 📊 Dataset Description

### Source Data

**File**: `data/customer_churn_raw.csv`  
**Records**: 107 customers (including 1 header row = 106 data rows)  
**Features**: 21 original columns

### Data Fields

| Category                   | Fields                                                                                                               | Description                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Identifiers**            | `customer_id`                                                                                                        | Unique customer identifier                                                    |
| **Demographics**           | `age`, `gender`                                                                                                      | Customer demographic information                                              |
| **Subscription**           | `subscription_plan`, `tenure_months`, `contract_type`                                                                | Plan type (Basic/Standard/Premium), length of subscription, contract duration |
| **Financial**              | `monthly_charges`, `total_charges`, `payment_method`                                                                 | Revenue metrics and payment details                                           |
| **Usage & Engagement**     | `login_frequency_monthly`, `features_used`, `data_consumption_gb`, `engagement_score`, `days_since_last_activity`    | Customer activity and engagement patterns                                     |
| **Support & Satisfaction** | `billing_issues_count`, `plan_changes`, `support_tickets`, `avg_resolution_hours`, `satisfaction_score`, `nps_score` | Customer service and satisfaction metrics                                     |
| **Target Variable**        | `churned`                                                                                                            | Binary indicator (0 = Active, 1 = Churned)                                    |

### Data Quality Issues

The raw dataset intentionally contains real-world data quality problems:

- **Missing values** in multiple columns (age, gender, charges, scores)
- **Duplicate customer records**
- **Inconsistent formatting** (e.g., "Male" vs "male", "Credit Card" vs "CC")
- **Currency symbols** in numeric fields (e.g., "$29.99")
- **Outliers** (e.g., age = 150, negative charges, tenure = -10)
- **Invalid ranges** (e.g., satisfaction_score = 10.5, nps_score = -1)

---

## 📁 Project Structure

```
customer-churn-ml-demo/
│
├── data/                                    # Data directory
│   ├── customer_churn_raw.csv              # Raw input data (with quality issues)
│   ├── customer_churn_cleaned.csv          # Cleaned data (after pipeline step 1)
│   ├── customer_churn_featured.csv         # Feature-engineered data (after pipeline step 2)
│   ├── cleaning_summary.csv                # Summary of cleaning operations
│   └── feature_documentation.csv           # Catalog of all features
│
├── src/                                     # Source code
│   ├── 01_clean_data.py                    # Data cleaning pipeline
│   └── 02_feature_engineering.py           # Feature engineering pipeline
│
└── README.md                                # This file
```

---

## 🔄 Pipeline Process

The pipeline consists of **two main stages** that must be run sequentially:

### Stage 1: Data Cleaning (`01_clean_data.py`)

Transforms raw, messy data into a clean, validated dataset.

**Operations (in order)**:

1. **Remove Duplicates**: Identifies and removes duplicate customer records based on `customer_id`
2. **Clean Categorical Variables**: Standardizes values (e.g., "M" → "Male", "CC" → "Credit Card")
3. **Correct Data Types**: Removes currency symbols, converts strings to numeric types
4. **Handle Missing Values**: Imputes missing data (median for numeric, mode for categorical)
5. **Fix Outliers & Validate Ranges**: Clips values to valid ranges (age: 18-100, scores: 0-10)
6. **Finalize Data Types**: Converts to final int/float types after all cleaning
7. **Validate Final Data**: Ensures no missing values, duplicates, or invalid ranges remain

**Key Features**:

- Comprehensive data quality reports before and after cleaning
- Detailed logging of all cleaning operations
- Automatic validation checks
- Summary statistics saved to `cleaning_summary.csv`

### Stage 2: Feature Engineering (`02_feature_engineering.py`)

Creates derived features that capture customer behavior patterns and risk indicators.

**Feature Categories**:

1. **Customer Value Metrics**

   - `monthly_value_ratio`: Average revenue per user (ARPU)
   - `charge_per_feature`: Cost efficiency
   - `customer_lifetime_value`: Total value for active customers
   - `value_tier`: Customer value segmentation

2. **Engagement Indicators**

   - `engagement_velocity`: Engagement per month
   - `login_intensity`: Average daily logins
   - `data_per_login`: Data consumption patterns
   - `activity_recency_category`: Customer activity status
   - `features_utilization_rate`: Feature adoption rate

3. **Support Risk Features**

   - `support_rate_annual`: Annualized support ticket rate
   - `resolution_burden`: Total time spent on support
   - `satisfaction_gap`: Distance from perfect score
   - `billing_risk_flag`: Billing issues indicator
   - `nps_category`: Net Promoter Score classification

4. **Interaction Features**

   - `plan_tenure_mismatch`: Plan-tenure fit indicator
   - `usage_plan_mismatch`: Usage-plan alignment
   - `payment_stability`: Payment reliability metric
   - `contract_value_risk`: Contract-value alignment

5. **Tenure & Lifecycle Features**
   - `lifecycle_stage`: Customer maturity stage
   - `contract_tenure_ratio`: Contract renewal cycles
   - `engagement_growth_rate`: Engagement trend
   - `tenure_stability`: Long-term stability metric

**Total Features Created**: ~35 new derived features

---

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)

### Required Libraries

#### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install required packages
pip install pandas numpy
```

#### Option 2: Using requirements.txt (if available)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install from requirements file
pip install -r requirements.txt
```

> **Note**: Remember to activate the virtual environment before running the pipeline scripts.

### Verify Installation

```bash
python --version        # Should show Python 3.12+
python -c "import pandas; print(pandas.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

---

## ▶️ How to Run

### Quick Start

Navigate to the project directory and run the pipeline scripts in order:

```bash
# Navigate to project directory
cd customer-churn-ml-demo

# Step 1: Clean the raw data
python src/01_clean_data.py

# Step 2: Engineer features from cleaned data
python src/02_feature_engineering.py
```

### Step-by-Step Instructions

#### Step 1: Data Cleaning

```bash
python src/01_clean_data.py
```

**What it does**:

- Reads `data/customer_churn_raw.csv`
- Performs all cleaning operations
- Saves cleaned data to `data/customer_churn_cleaned.csv`
- Generates `data/cleaning_summary.csv`

**Expected Output**:

```
================================================================================
CUSTOMER CHURN DATA CLEANING
================================================================================
Loading raw data from: .../data/customer_churn_raw.csv
Loaded 106 records

============================================================
Initial Data Quality Report
============================================================
Total Records: 106
Total Columns: 21
Missing Values:
  - gender: 6 (5.66%)
  - subscription_plan: 5 (4.72%)
  ...
Duplicates:
  - Duplicate customer_ids: 2
...

1. REMOVING DUPLICATES
------------------------------------------------------------
✓ Removed 2 duplicate records
  Records remaining: 104

2. CLEANING CATEGORICAL VARIABLES
------------------------------------------------------------
✓ Standardized gender values: ['Female', 'Male']
✓ Standardized payment_method values: ['Bank Transfer', 'Credit Card', 'PayPal']
...

✓ ALL VALIDATION CHECKS PASSED

================================================================================
CLEANING COMPLETE
================================================================================
```

**Duration**: ~1-2 seconds

#### Step 2: Feature Engineering

```bash
python src/02_feature_engineering.py
```

**What it does**:

- Reads `data/customer_churn_cleaned.csv`
- Creates 35+ derived features
- Validates all features (handles inf/NaN)
- Saves featured data to `data/customer_churn_featured.csv`
- Generates `data/feature_documentation.csv`

**Expected Output**:

```
================================================================================
CUSTOMER CHURN FEATURE ENGINEERING
================================================================================
Loading cleaned data from: .../data/customer_churn_cleaned.csv
Loaded 104 records with 21 columns

1. CREATING CUSTOMER VALUE FEATURES
------------------------------------------------------------
✓ Created 'monthly_value_ratio' (ARPU)
✓ Created 'charge_per_feature'
✓ Created 'customer_lifetime_value'
✓ Created 'value_tier'

2. CREATING ENGAGEMENT FEATURES
------------------------------------------------------------
✓ Created 'engagement_velocity'
...

Top 15 features correlated with churn:
  1. days_since_last_activity: 0.8234
  2. billing_issues_count: 0.7456
  ...

================================================================================
FEATURE ENGINEERING COMPLETE
================================================================================
Records: 104
Total features: 56
Ready for model training!
```

**Duration**: ~2-3 seconds

---

## 📤 Output Files

After running the complete pipeline, you will find:

### 1. `customer_churn_cleaned.csv`

Clean, validated dataset ready for analysis.

- **Records**: 104 (2 duplicates removed)
- **Features**: 21 (original columns)
- **Quality**: No missing values, no duplicates, all ranges validated

### 2. `customer_churn_featured.csv`

Machine learning-ready dataset with engineered features.

- **Records**: 104
- **Features**: ~56 (21 original + 35 engineered)
- **Ready for**: Model training, exploratory analysis, predictions

### 3. `cleaning_summary.csv`

Summary statistics of the cleaning process.

| Metric                 | Value |
| ---------------------- | ----- |
| Initial Records        | 106   |
| Final Records          | 104   |
| Records Removed        | 2     |
| Initial Missing Values | ~45   |
| Final Missing Values   | 0     |

### 4. `feature_documentation.csv`

Catalog of all features with metadata.

| Feature             | Type       | Data Type | Missing | Unique |
| ------------------- | ---------- | --------- | ------- | ------ |
| customer_id         | Original   | object    | 0       | 104    |
| monthly_value_ratio | Engineered | float64   | 0       | 104    |
| ...                 | ...        | ...       | ...     | ...    |

---

## 🔧 Feature Engineering Details

### Why Feature Engineering?

Raw data often doesn't capture complex patterns. Feature engineering creates meaningful combinations and transformations that help ML models learn better.

### Example Features

**1. Engagement Velocity**

```python
engagement_velocity = engagement_score / tenure_months
```

_Interpretation_: Higher values indicate rapidly engaged customers; lower values may indicate declining interest.

**2. Plan-Tenure Mismatch**

```python
# Risk flag for customers on wrong plan
if premium plan AND tenure < 6 months:
    plan_tenure_mismatch = 1  # Risky: May have been oversold
```

**3. NPS Category**

```python
# Net Promoter Score classification
0-6:  Detractor  (likely to churn)
7-8:  Passive    (neutral)
9-10: Promoter   (advocates)
```

### Feature Validation

The pipeline automatically:

- Replaces infinite values (from division by zero) with 0
- Fills NaN values (numeric: 0, categorical: mode)
- Computes correlation with churn target
- Generates feature documentation

---

## 🔮 Next Steps

After completing this pipeline, you can:

### 1. Exploratory Data Analysis (EDA)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load featured data
df = pd.read_csv('data/customer_churn_featured.csv')

# Visualize churn distribution
df['churned'].value_counts().plot(kind='bar')
plt.title('Churn Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.show()
```

### 2. Train Machine Learning Models

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare data
X = df.drop(['customer_id', 'churned'], axis=1)
# Convert categorical to numeric (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)
y = df['churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 3. Model Comparison

Try multiple algorithms:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks

### 4. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

### 5. Feature Importance Analysis

```python
# Get feature importance from trained model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))
```

### 6. Deploy Predictions

Create a prediction pipeline:

```python
def predict_churn(customer_data):
    """Predict churn probability for new customers"""
    # Clean data
    cleaned = clean_pipeline(customer_data)
    # Engineer features
    featured = feature_pipeline(cleaned)
    # Predict
    prediction = model.predict_proba(featured)
    return prediction[:, 1]  # Probability of churn
```

---

## 📚 Learning Objectives

This project demonstrates:

✅ **Data Cleaning Best Practices**

- Handling missing values
- Removing duplicates
- Standardizing categorical variables
- Validating data quality

✅ **Feature Engineering Techniques**

- Creating ratio and interaction features
- Binning continuous variables
- Engineering domain-specific metrics
- Feature validation

✅ **Python Programming**

- Pandas data manipulation
- NumPy numerical operations
- Modular code organization
- Documentation and logging

✅ **Machine Learning Pipeline**

- Sequential processing stages
- Data validation and quality checks
- Reproducible workflows
- Output documentation

---

## 🤝 Contributing

To extend this project:

1. Add new data sources (customer feedback, web analytics)
2. Implement additional feature engineering techniques
3. Add model training scripts
4. Create visualization dashboards
5. Implement automated ML (AutoML) pipelines

---

## 📝 License

This is a training/demonstration project for educational purposes.

---

### 📞 Support

For questions or issues:

1. Review the console output for detailed error messages
2. Check that all required libraries are installed
3. Verify file paths are correct
4. Ensure Python 3.8+ is being used

---

**Happy Learning! 🚀**
