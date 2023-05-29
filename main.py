import streamlit as st
import numpy as np
from utils.preprocessor import preprocessing, vectorizer
# from utils.model import model1, model2, model3
from utils.model import model4

st.title("Sentimen Analisis")

inp = st.text_area(label="inp", label_visibility="hidden", placeholder="Please input text...", height=350)
btn_submit = st.button("submit")


if btn_submit:
    if inp == "":
        st.write("erro")
    else:
        prep = preprocessing(inp)
        
        transform = vectorizer.transform([" ".join(prep)])

        label = model4.predict(np.asarray(transform.todense()))
        st.write(label[0])  