import streamlit as st
import pandas as pd
import glob
import os
import time
from streamlit_autorefresh import st_autorefresh

# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
count = st_autorefresh(interval=10000, limit=100000, key="fizzbuzzcounter")

st.title("Training")
st.cache(suppress_st_warning=True)
def grab_data():
    glb = glob.glob("data/logs/*")
    glb.sort(key=os.path.getmtime, reverse=True)
    g = glb[0]
    glb = glob.glob(g +"/*")
    glb.sort(key=os.path.getmtime, reverse=True)
    g = glb[0]
    st.header(g, anchor=None)
    return pd.read_json(os.path.join(g,"training.json"))
try:
    chart_data = grab_data()
    chart_data['target_loss'] =  0.07
    st.header("Global Loss")
    plt1_data = chart_data.loc[:,['global-loss', 'target_loss']]
    st.line_chart(plt1_data)
    chart_data = chart_data.drop('global-loss', axis=1)
    st.header("Training Loss across Federation")
    st.line_chart(chart_data.filter(regex=("-loss")))
except:
    st.header("Data is not yet ready")