# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'improved_iris_app.py'.
# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating a Logistic Regression model. 
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

@st.cache()
def prediction(model ,SepalLengthCm,SepalWidthCm, PetalLengthCm, PetalWidthCm):

	y_pred = model.predict([[SepalLengthCm,SepalWidthCm, PetalLengthCm, PetalWidthCm]])

	if y_pred ==0:
		return 'Iris-setosa'
	elif y_pred ==1:
		return 'Iris-virginica'
	else:
		return 'Iris-versicolor'

#Design 
st.sidebar.title("Iris Flower Species Prediction Application")

l1 =['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
l2 = []
for i in range(4):

	l2.append(st.sidebar.slider(l1[i] , 0.00 , 10.00))

st.sidebar.subheader("Choose Classifier")
model_chosen = st.sidebar.selectbox("Classifier" , ("LogisticRegression", 'SVC', 'RandomForestClassifier'))
if model_chosen == 'LogisticRegression':
	model = log_reg
elif model_chosen == 'SVC':
	model = svc_model
elif model_chosen == 'RandomForestClassifier':
	model = rf_clf

if st.sidebar.button("Predict"):
	predicted_value = prediction(model ,l2[0],l2[1],l2[2],l2[3])

	st.write("Species predicted ", predicted_value)
	st.write("Accuracy of this model ", model.score(X_train,y_train))

