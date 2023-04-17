import streamlit as st
import pandas as pd
import os
import glob
import numpy as np
import time
from bokeh.io import show, output_file
from bokeh.plotting import figure

from streamlit_autorefresh import st_autorefresh

# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
count = st_autorefresh(interval=10000, limit=100000, key="fizzbuzzcounter")

def grab_data():
    glb = glob.glob("data/logs/*")
    glb.sort(key=os.path.getmtime, reverse=True)
    g = glb[0]
    return pd.read_json(os.path.join(g, "misc.json"))

st.title("Communications")
chart_data = grab_data()
hist, edges = np.histogram(chart_data['latency'], density=True, bins=25)
p = figure()
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
st.header("Latency")
st.bokeh_chart(p, use_container_width=True)
hist2, edges2 = np.histogram(chart_data['avg_msg_size'], density=True, bins=25)
p2 = figure()
p2.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:], line_color="white")
st.header("Average Message Size")
st.bokeh_chart(p2, use_container_width=True)

