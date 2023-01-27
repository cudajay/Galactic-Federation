import streamlit as st
import pandas as pd
st.title("Hello from Mr Juan !!!")
import glob
import os

glb = glob.glob("data/logs/*")
glb.sort(key=os.path.getmtime, reverse=True)
chart_data = pd.read_json(glb[0])

st.line_chart(chart_data['global-loss'])
chart_data = chart_data.drop('global-loss', axis=1)
st.line_chart(chart_data.filter(regex=("-val")))
st.line_chart(chart_data.filter(regex=("-loss")))