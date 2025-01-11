# Dimensionality Reduction Using K-Means

## Overview
This project leverages the **UCI Human Activity Recognition Dataset** to explore and classify human activities using smartphone sensor data. The code demonstrates techniques such as exploratory data analysis (EDA), feature scaling, dimensionality reduction, and machine learning model training using **Gaussian Naive Bayes** and **K-Means clustering**.

---

## Dataset
The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). It contains sensor readings (accelerometer and gyroscope) collected from smartphones while users performed daily activities such as walking, sitting, and standing.

**Data Characteristics:**
- Contains multiple features derived from sensor data.
- Includes a target variable indicating the type of activity performed.

---

## Steps in the Code
### 1. Data Download and Preprocessing
- The dataset is downloaded and extracted programmatically using `requests` and `zipfile`.
- The features (`X_train.txt`) and labels (`y_train.txt`) are read into Pandas DataFrames.

### 2. Exploratory Data Analysis (EDA)
- Basic statistics, data shape, and data types are analyzed.
- A correlation matrix is computed to understand feature relationships.

### 3. Data Preparation
- The target labels are encoded using `LabelEncoder`.
- Features are standardized using `StandardScaler`.
- The dataset is split into training and testing sets using an 80-20 ratio.

### 4. Model Training: Gaussian Naive Bayes
- A pipeline is created using **Gaussian Naive Bayes** for classification.
- The model is trained on the full set of features, and the accuracy and time taken for prediction are reported.

### 5. Dimensionality Reduction: K-Means Clustering
- **K-Means clustering** is applied to group similar features.
- A subset of representative features is selected from each cluster.

### 6. Model Training on Reduced Dataset
- The Gaussian Naive Bayes model is retrained using the reduced feature set.
- Accuracy and time taken for prediction are reported.

---

## Key Dependencies
- Python libraries: `requests`, `BeautifulSoup`, `zipfile`, `pandas`, `numpy`, `sklearn`

To install the required libraries, run:
```bash
pip install requests beautifulsoup4 pandas numpy scikit-learn
```

---

## Results
1. **Initial Model:**
   - Full feature set with Gaussian Naive Bayes.
   - Accuracy and time taken are reported.

2. **Dimensionality Reduction:**
   - Selected features using K-Means clustering.
   - Retrained Gaussian Naive Bayes on reduced features.
   - Accuracy and time taken for this reduced model are compared to the initial model.

---

## How to Run the Code
1. Clone this repository.
2. Install dependencies.
3. Run the Python script in your terminal:
   ```bash
   python script_name.py
   ```
4. View the output in your terminal, which includes:
   - Dataset statistics.
   - Model accuracy and time taken for both full and reduced feature sets.

---

## Future Work
- Experiment with other classifiers like **Random Forest** or **SVM**.
- Use PCA or LDA for dimensionality reduction.
- Visualize the clusters formed by K-Means.

---

## Acknowledgements
- **Dataset**: UCI Machine Learning Repository.
- **Python Libraries**: scikit-learn, pandas, numpy.

