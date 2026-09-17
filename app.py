import pandas as pd

df = pd.read_csv("/content/Iris.csv")

df

df.info()

df.describe

df.columns

df.head()

df.tail()

df.columns

#value_counts()=counts unique values
df['Species'].value_counts()

df.duplicated().sum()

df.describe()

x= df.drop(columns = ['Id','Species'])

y = df['Species']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=56)

x_train

x_test

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

model = knn.fit(x_train,y_train)

y_pred = model.predict(x_test)
display(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
display("Classification Report:")
display(classification_report(y_test,y_pred))

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)

cr = classification_report(y_test,y_pred)



import joblib

model = joblib.dump(model,"iris_model.pkl")

pip install streamlit

import streamlit as st

#MOdel_Deployment
#load_model
model = joblib.load("iris_model.pkl")

#page title
st.title("Machine Learning on Iris Dataset")

#input labels
sepal_length = st.number_input("sepal_lenght")
sepal_width = st.number_input("sepal_width")
petal_lenght = st.number_input("petal_lenght")
petal_width = st.number_input("Petal_width")

import numpy as np
import streamlit as st

#prediction
if st.button("predict"):
  input = np.array([[sepal_lenght,
                     sepal_width,
                     petal_lenght,
                     petal_width]]).astype(np.float64)
  prediction = model.predict(input)
  st.success(prediction)
