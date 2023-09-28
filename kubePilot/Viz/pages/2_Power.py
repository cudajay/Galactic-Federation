import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
count = st_autorefresh(interval=10000, limit=100000, key="fizzbuzzcounter")

st.title("Power")

#Differential privacy measure