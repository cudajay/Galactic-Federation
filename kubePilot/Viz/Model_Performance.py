import streamlit as st
import pandas as pd
st.title("Training")
import glob
import os
import time

st.cache(suppress_st_warning=True)
def grab_data():
    glb = glob.glob("data/logs/*")
    glb.sort(key=os.path.getmtime, reverse=True)
    g = glb[0]
    return pd.read_json(os.path.join(g,"training.json"))
try:
    chart_data = grab_data()
    st.header("Global Loss")
    st.line_chart(chart_data['global-loss'])
    chart_data = chart_data.drop('global-loss', axis=1)
    st.header("Training Loss across Federation")
    st.line_chart(chart_data.filter(regex=("-val")))
    st.header("Validation Loss across Federation")
    st.line_chart(chart_data.filter(regex=("-loss")))
except:
    st.header("Data is not yet ready")