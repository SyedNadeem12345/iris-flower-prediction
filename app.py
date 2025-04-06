import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Iris Flower Species Prediction Web App")

st.write("Upload your Iris Flower Data and predict the Species using Machine Learning")

df = pd.read_csv('data/iris.csv')

sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width', 0.1, 2.5, 1.0)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = SVC(gamma='auto')
model.fit(X_train, y_train)

prediction = model.predict(input_data)

st.subheader("Predicted Species:")
st.write(prediction[0])

if st.button('Evaluate Model Accuracy'):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Accuracy: ", acc)
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
