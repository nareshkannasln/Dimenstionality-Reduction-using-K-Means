import requests
from bs4 import BeautifulSoup
import zipfile
import io
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import time

# Function to download and load dataset
def load_data():
    page_url = 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones'
    page_response = requests.get(page_url)
    if page_response.status_code == 200:
        soup = BeautifulSoup(page_response.content, 'html.parser')
        download_link = soup.select_one('a[href$=".zip"]')['href']
        full_download_url = 'https://archive.ics.uci.edu' + download_link
        response = requests.get(full_download_url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
                inner_zip_name = 'UCI HAR Dataset.zip'
                with outer_zip.open(inner_zip_name) as inner_zip_file:
                    with zipfile.ZipFile(io.BytesIO(inner_zip_file.read())) as inner_zip:
                        with inner_zip.open('UCI HAR Dataset/train/X_train.txt') as myfile:
                            df = pd.read_csv(myfile, delim_whitespace=True, header=None)
                        with inner_zip.open('UCI HAR Dataset/train/y_train.txt') as myfile_y:
                            y = pd.read_csv(myfile_y, delim_whitespace=True, header=None)
    else:
        raise Exception("Failed to download or parse the dataset.")
    return df, y

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import time

# Load dataset
df, y = load_data()

#TASK 1 - DO EDA and understand a little about the data.
#Only important thing is to know that it has a lot of features that don't make sense, just a
#bunch of readings from sensors.
#We think many of these features are redundant or irrelevant, and we want to find good features.
print(df.head())  # Display first few rows of the dataframe
print(df.shape)  # Print the shape of the dataframe (rows, columns)
print(df.describe())  # Generate descriptive statistics
print(df.info()) # Check data types and missing values
print(y.value_counts()) # Analyze the distribution of target variable 'y'

#Correlation Matrix
correlation_matrix = df.corr()

# Task 2: Encode class labels
# YOUR CODE HERE: Use LabelEncoder to encode class labels
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y[0])

# # Solution to Task 2: Encode class labels

# label_encoder = LabelEncoder()
# encoded_y = label_encoder.fit_transform(y.values.ravel())

# Task 3: Scale the features using StandardScaler
# # YOUR CODE HERE: Apply StandardScaler to df
# scaler = # YOUR CODE HERE
# df_scaled = # YOUR CODE HERE

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Task 4: Split the data into training and testing sets
# # YOUR CODE HERE: Use train_test_split to split the data
# X_train_full, X_test_full, y_train, y_test = # YOUR CODE HERE

X_train_full, X_test_full, y_train, y_test = train_test_split(df_scaled, encoded_y, test_size=0.2, random_state=42)

#TASK 5 - 1. Create a pipeline using Gaussian Naive Bayes
# #         2. Fit the model to the training data
# #         3. Predict values for test set
# #         4. Print accuracy score

from sklearn.naive_bayes import GaussianNB

import time
start_time = time.time()

# Create a pipeline with Gaussian Naive Bayes
pipeline = Pipeline([
    ('classifier', GaussianNB())
])

pipeline.fit(X_train_full, y_train)

y_pred = pipeline.predict(X_test_full)

accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
print(f"Accuracy: {accuracy}")

time_taken = end_time - start_time
print(f"{time_taken} seconds")

# TASK 7 - K-Means for dimensionality reduction
# n_clusters = #FILL
# kmeans = #FILL
# kmeans.fit(#FILL)  # Transpose to treat features as data points
# selected_features_indices = #FILL
# selected_features = #FILL

n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(df_scaled.T)  # Transpose to treat features as data points
selected_features_indices = [np.random.choice(np.where(kmeans.labels_ == i)[0]) for i in range(n_clusters)]
selected_features = df_scaled[:, selected_features_indices]

# TASK 8 - Train another model (GaussianNB) on the new dataset, and report time taken and accuracy

start_time = time.time()

# Use selected features from KMeans
X_train_selected = X_train_full[:, selected_features_indices]
X_test_selected = X_test_full[:, selected_features_indices]

# Create and train the GaussianNB model
gnb = GaussianNB()
gnb.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred_gnb = gnb.predict(X_test_selected)

# Calculate accuracy
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"GaussianNB Accuracy: {accuracy_gnb}")

end_time = time.time()
time_taken_gnb = end_time - start_time
print(f"GaussianNB Time Taken: {time_taken_gnb} seconds")