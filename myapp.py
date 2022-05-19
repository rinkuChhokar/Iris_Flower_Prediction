import streamlit as st
import pandas as pd
from sklearn import datasets
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

image = Image.open("main.webp")
st.image(image,use_column_width=True)

st.write("""

	# Simple Iris Flower Prediction App
	This app predicts the Iris flower type!
	""")

st.sidebar.header("User Input Parameters")


def user_input_features():

	sepal_length = st.sidebar.slider("Sepal length",4.3,7.9,5.4)
	sepal_width = st.sidebar.slider("Sepal width",2.0,4.4,3.4)
	petal_length = st.sidebar.slider("Petal length",1.0,6.9,5.4)
	petal_width = st.sidebar.slider("Petal width",0.1,2.5,1.0)

	data = {'sepal_length':sepal_length,
			'sepal_width':sepal_width,
			'petal_length':petal_length,
			'petal_width':petal_width
	}

	features = pd.DataFrame(data,index=[0])
	return features


df = user_input_features()

st.header("User Input parameters")
st.write(df)

iris = datasets.load_iris()
X = iris.data	
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)


st.subheader("Prediction")
if iris.target_names[prediction]=="setosa":
	st.write("Setosa")
	image = Image.open("setosa.jpg")
	st.image(image,use_column_width=True)


elif iris.target_names[prediction]=="versicolor":
	st.write("Versicolor")
	image = Image.open("versicolor.jpg")
	st.image(image,use_column_width=True)


else:
	st.write("Virginica")
	image = Image.open("virginica.jpg")
	st.image(image,use_column_width=True)

st.subheader("Prediction Probabiity")
st.write("Setosa-",int((prediction_prob[0][0])*100),"%")
st.write("Versicolor-",int((prediction_prob[0][1])*100),"%")
st.write("Virginica-",int((prediction_prob[0][2])*100),"%")

