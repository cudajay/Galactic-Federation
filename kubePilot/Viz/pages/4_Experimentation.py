import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import glob
# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
count = st_autorefresh(interval=10000, limit=100000, key="fizzbuzzcounter")

st.title("Experimentation")


glb = glob.glob("data/logs/*")
options = st.multiselect("Select trials to compare",glb)
