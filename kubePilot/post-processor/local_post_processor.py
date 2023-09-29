from errors import Errors
import numpy as np
import tensorflow as tf
import yaml
import pandas as pd
import glob
import random
import time
import sys

def evaluate_sequences(errors, label_row, result_tracker):
    """
    Compare identified anomalous sequences with labeled anomalous sequences.
    Args:
        errors (obj): Errors class object containing detected anomaly
            sequences for a channel
        label_row (pandas Series): Contains labels and true anomaly details
            for a channel
    Returns:
        result_row (dict): anomaly detection accuracy and results
    """

    result_row = {
        'false_positives': 0,
        'false_negatives': 0,
        'true_positives': 0,
        'fp_sequences': [],
        'tp_sequences': [],
        'num_true_anoms': 0
    }

    matched_true_seqs = []

    #label_row['anomaly_sequences'] = label_row['anomaly_sequences']
    lrd = label_row['anomaly_sequences']
    lrd = eval(lrd[lrd.keys()[0]])
    result_row['num_true_anoms'] += len(lrd)
    result_row['scores'] = errors.anom_scores

    if len(errors.E_seq) == 0:
        result_row['false_negatives'] = result_row['num_true_anoms']

    else:
        true_indices_grouped = [list(range(e[0], e[1]+1)) for e in lrd]
        true_indices_flat = set([i for group in true_indices_grouped for i in group])

        for e_seq in errors.E_seq:
            i_anom_predicted = set(range(e_seq[0], e_seq[1]+1))

            matched_indices = list(i_anom_predicted & true_indices_flat)
            valid = True if len(matched_indices) > 0 else False

            if valid:

                result_row['tp_sequences'].append(e_seq)

                true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                    len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                if not true_seq_index[0] in matched_true_seqs:
                    matched_true_seqs.append(true_seq_index[0])
                    result_row['true_positives'] += 1

            else:
                result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                result_row['false_positives'] += 1

        result_row["false_negatives"] = len(np.delete(lrd,
                                                        matched_true_seqs, axis=0))

    print('Channel Stats: TP: {}  FP: {}  FN: {}'.format(result_row['true_positives'],
                                                                result_row['false_positives'],
                                                                result_row['false_negatives']))

    for key, value in result_row.items():
        if key in result_tracker:
            result_tracker[key] += result_row[key]

    return result_row

class channel:
    def __init__(self,X_test, y_test, y_hat, id) -> None:
        self.id = id
        self.y_test = y_test
        self.y_hat = y_hat
        self.X_test = X_test 


assert(len(sys.argv) == 4)
start_time = time.time()
        
#test_dat_dir = "/home/jmeow/data/25/test/"
test_dat_dir = sys.argv[1]
#eval_data_dir = "/home/jmeow/data/logs/2023-04-19-lstm-fedAvg/"
eval_data_dir = sys.argv[2]
use_spec_models = sys.argv[3]
filename = eval_data_dir + "config.yaml"
with open(filename, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
model = tf.keras.models.load_model(eval_data_dir+"model.h5")
glb = glob.glob("/home/jmeow/data/25/test/*X.npy")
chs = [ g.split("/")[-1].split("_")[0] for g in glb]
for ch_id in chs:
    x = np.load(test_dat_dir + f"{ch_id}_X.npy")
    y = np.load(test_dat_dir + f"{ch_id}_y.npy")
    if use_spec_models:
        model = tf.keras.models.load_model("/home/jmeow/git/telemanom/data/2023-04-26_11.09.04/models/"+f"{ch_id}.h5")
    yhat =  model(x)
    yhat = yhat.numpy()
    ch = channel(x, y, yhat, ch_id)
    errors = Errors(ch, cfg)
    errors.process_batches(ch)

    result_row = {
        'chan_id': ch_id,
    #    'num_train_values': len(channel.X_train),
        'num_test_values': len(ch.X_test),
        'n_predicted_anoms': len(errors.E_seq),
        'normalized_pred_error': errors.normalized,
        'anom_scores': errors.anom_scores
    }
    print(result_row)

    df = pd.read_csv("/home/jmeow/data/labeled_anomalies.csv")
    lr  = df[df['chan_id'] == ch_id]
    evaluate_sequences(errors, lr, result_tracker)
print('Final Totals:')
print('-----------------')
print('True Positives: {}'
            .format(result_tracker['true_positives']))
print('False Positives: {}'
            .format(result_tracker['false_positives']))
print('False Negatives: {}\n'
            .format(result_tracker['false_negatives']))
try:
    print('Precision: {0:.2f}'
                .format(float(result_tracker['true_positives'])
                        / float(result_tracker['true_positives']
                                + result_tracker['false_positives'])))
    print('Recall: {0:.2f}'
                .format(float(result_tracker['true_positives'])
                        / float(result_tracker['true_positives']
                                + result_tracker['false_negatives'])))
except ZeroDivisionError:
    print('Precision: NaN')
    print('Recall: NaN')
    
end_time = time.time()
runtime = end_time - start_time

print("The program took", runtime, "seconds to run.")

