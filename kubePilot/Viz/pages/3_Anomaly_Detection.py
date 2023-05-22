import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
df = pd.read_csv("data/labeled_anomalies.csv")


def grab_data():
    glb = glob.glob("data/logs/*")
    glb.sort(key=os.path.getmtime, reverse=True)
    g = glb[0]
    with open(os.path.join(g, "config.yaml"), 'r') as file:
        cfg = yaml.safe_load(file)
    return (cfg, tf.keras.models.load_model(os.path.join(g, "model.h5")))
# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
count = st_autorefresh(interval=90000, limit=100000, key="fizzbuzzcounter")

st.title("Anomaly Detection")
"""
fig, ax = plt.subplots()
ax.plot(rolling_mean, 'r')
ax.plot(x, 'g')
ax.fill_between(np.arange(len(e_s)), rolling_mean-threshold_multiplier*rolling_std, rolling_mean+threshold_multiplier*rolling_std, alpha=0.2, color='g')
for i in df[df['chan_id'] == 'A-1']['anomaly_sequences']:
    j = int(i.replace("[", "").replace("]", "").split(",")[0])
    k = int(i.replace("[", "").replace("]", "").split(",")[1])
    ax.axvspan(j, k, color='red', alpha=0.5)
for i in anomaly_indices[0]:
    j = int(i)
    ax.axvspan(j, j + 1, color='blue', alpha=0.5)
st.pyplot(fig)
"""