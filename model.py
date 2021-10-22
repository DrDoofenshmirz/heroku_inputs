import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle



# Load the csv file
df = pd.read_csv("heart.csv")

print(df.head())

# Select independent and dependent variable
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df["target"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)




classifier = LogisticRegression(random_state = 51, penalty = 'l2')
classifier.fit(X_train, y_train)



# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))