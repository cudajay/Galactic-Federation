import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
import yaml
import random
import os
import time
df = pd.read_csv("data/labeled_anomalies.csv")

def aggregate_predictions(y_hat_batch, cfg, method='first'):

    agg_y_hat_batch = np.array([])
    n_predictions = cfg["npreds"] #10

    for t in range(len(y_hat_batch)):

        start_idx = t - n_predictions
        start_idx = start_idx if start_idx >= 0 else 0

        # predictions pertaining to a specific timestep lie along diagonal
        y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

        if method == 'first':
            agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
        elif method == 'mean':
            agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

    return agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
    #y_hat = np.append(y_hat, agg_y_hat_batch)

def batch_predict(X_test, y_test, model, cfg):
    l_s = cfg["l_s"] #250
    batch_size = cfg["batch_size"] #70
    y_hat = np.array([])

    num_batches = int((y_test.shape[0] - l_s)
                        / batch_size)
    if num_batches < 0:
        raise ValueError('l_s ({}) too large for stream length {}.'
                            .format(l_s, y_test.shape[0]))

    # simulate data arriving in batches, predict each batch
    for i in range(0, num_batches + 1):
        prior_idx = i * batch_size
        idx = (i + 1) * batch_size

        if i + 1 == num_batches + 1:
            # remaining values won't necessarily equal batch size
            idx = y_test.shape[0]

        X_test_batch = X_test[prior_idx:idx]
        y_hat_batch = model.predict(X_test_batch)
        y_hat = np.append(y_hat, aggregate_predictions(y_hat_batch, cfg))

    y_hat =  np.reshape(y_hat, (y_hat.size,))
    return y_hat

def grab_data():
    glb = glob.glob("data/logs/*")
    glb.sort(key=os.path.getmtime, reverse=True)
    g = glb[0]
    with open(os.path.join(g, "config.yaml"), 'r') as file:
        cfg = yaml.safe_load(file)
    return (g, cfg, tf.keras.models.load_model(os.path.join(g, "model.h5")))


wrk_dir, cfg, model = grab_data()
dir_path = os.path.join(wrk_dir, "post_processing")

# Check if the directory exists
filename = os.path.join(dir_path, 'to_test.yaml')   
if not os.path.exists(dir_path):
    # Create the directory
    os.makedirs(dir_path)
    glb = glob.glob("data/25/test/*X.npy")
    with open(filename, 'w') as file:
        yaml.dump(glb, file)
    print(f'Created directory: {dir_path}')
    

with open(filename, 'r') as file:
    l = yaml.load(file, Loader=yaml.FullLoader)
if not l:
    print("No more test sets available")
    #TODO: find cleaner way to exit container loop
    while True:
        x = 1
with open(filename, 'w') as file:
    yaml.dump(l, file)
choice = l.pop()
x = np.load(choice)
y = np.load(choice.replace("X", "y"))
choice = choice.split("/")[-1].split("_")[0]
dir_path = os.path.join(dir_path, choice)
os.makedirs(dir_path)
print(f'Created directory: {dir_path}')

while True:
    
    yhat = batch_predict(x,y,model, cfg)
    batch_size = cfg["batch_size"]
    window_size = cfg["window_size"]
    smoothing_perc = cfg["smoothing_perc"]
    smoothing_window = int(batch_size * window_size
                                * smoothing_perc)
    e = np.array([abs(y_h-y_t[0]) for y_h, y_t in
                    zip(yhat, y)])
    e_s = pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten()
    # Calculate the moving average and standard deviation
    window_size = 500
    rolling_mean = np.convolve(e_s, np.ones(window_size)/window_size, mode='same')
    rolling_std = np.sqrt(np.convolve(np.square(e_s - rolling_mean), np.ones(window_size)/window_size, mode='same'))

    # Set the threshold as the rolling mean plus a multiple of the rolling standard deviation
    threshold_multiplier = 3
    threshold = rolling_mean + threshold_multiplier * rolling_std
    # Detect anomalies by comparing each data point to the threshold
    anomaly_mask = e_s> threshold
    anomaly_indices = np.where(anomaly_mask)

    np.save(os.path.join(dir_path, "yhat.npy"), yhat)
    np.save(os.path.join(dir_path, "e_s.npy"), e_s)
    np.save(os.path.join(dir_path, "rolling_mean.npy"), rolling_mean)
    np.save(os.path.join(dir_path, "threshold_multiplier.npy"), threshold_multiplier)
    np.save(os.path.join(dir_path, "rolling_std.npy"), rolling_std)

    rgs = [] 
    for i in df[df['chan_id'] == choice]['anomaly_sequences']:
        j = int(i.replace("[", "").replace("]", "").split(",")[0])
        k = int(i.replace("[", "").replace("]", "").split(",")[1])
        rgs.append(range(j,k))

    tp = 0
    for a in anomaly_indices[0]:
        for r in rgs:
            if a in r:
                tp += 1
                break
    fp = 0
    for a in anomaly_indices[0]:
        sm = 0
        for r in rgs:
            if a in r:
                sm += 1
        if not sm:
            fp += 1

    fn = anomaly_indices[0].shape[0] - (tp+fp)
    out = {}
    out['precision'] = tp/(tp+fp + 0.000001)
    out["recall"] = tp/(tp + fn + .0000001)
    out["f1"] = (2*out["precision"]*out["recall"])/(out["precision"]+out["recall"] + .0000001)


    with open(os.path.join(dir_path, "metrics.yaml"), 'w') as f:
        yaml.dump(out, f)
    time.sleep(60*5)