import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import joblib

def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()

@st.cache_resource
def load_openml():
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist['data'][:10000]
    y = mnist['target'][:10000].astype(np.uint8)
    return X, y

def load_image_from_openml(index):
    X = load_openml()[0]
    some_digit = X[index]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    return plt.gcf()

@st.cache_resource
def load_image_classifier_model():
    model = joblib.load('kap415model.pk1')
    print("Loaded kap415model")
    return model

#Streamlit:
st.title("Digit Classifier")
st.markdown("Todo: add text")
if st.button("Show Example Image"):
    st.pyplot(load_image_from_openml(5))
