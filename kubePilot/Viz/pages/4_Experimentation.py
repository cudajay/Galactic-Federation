import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import glob
import yaml
import json
import numpy as np
# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
count = st_autorefresh(interval=10000, limit=100000, key="fizzbuzzcounter")

st.title("Experimentation")


glb = glob.glob("data/logs/*")
glb.sort(reverse=True)
option = st.selectbox("Select Experiment to view",glb)
gexp = glob.glob(option + "/*")
st.write('You selected:', option)
data_list = []
col_list = []
len_list =[]
data_dct = {}
for exp in gexp:
    tmp = {}
    col_list.append(exp.split("/")[-1])
    with open(exp+"/config.yaml", "r") as filename:
        tmp['cfg'] = yaml.safe_load(filename)
    with open(exp+"/training.json", "r") as filename:
        dato = json.load(filename)
        tmp['td'] = [step['global-loss'] for step in dato]
        data_list.append(tmp['td'])
        len_list.append(len(tmp['td']))
    with open(exp+"/misc.json", "r") as filename:
        tmp["msc"] = json.load(filename)
    data_dct[exp.split("/")[-1]] =  tmp

option2 = st.multiselect("Select trials to compare",("Power", "loss"))
max_length = max(len_list)
fin_data = []
df = pd.DataFrame()
#Filling in NANs
for l,c in zip(data_list, col_list):
    df[c] =  np.pad(l, (0, max_length - len(l)), mode='constant', constant_values=np.nan)

st.write('You selected:', option2)


try:
    #chart_data['target_loss'] =  0.07
    st.header("Global Loss")
    st.line_chart(df)

except:
    st.header("Data is not yet ready")
