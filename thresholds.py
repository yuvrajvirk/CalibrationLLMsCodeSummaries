import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import json
import pickle
import os

def rescale(val):
    # [-1, 1] -> [0, 1]
    return (val+1)/2

def find_closest_index(array, value):
    """Find the index in 'array' closest to 'value'."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_roc_curve(metric_scores, true_int, metric_name):
    print('finding precision recall curve')
    precision, recall, pr_thresholds = precision_recall_curve(true_int, metric_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(f1_scores)  # Handle potential NaNs in f1_scores
    optimal_threshold = pr_thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    
    print('finding roc curve')
    fpr, tpr, roc_thresholds = roc_curve(true_int, metric_scores)
    roc_auc = auc(fpr, tpr)

    print('finding thresholds')
    # Find threshold for which precision is greater than 0.9 and recall is greater than 0.
    valid_precision_indices = np.where(precision > 0.9)[0]
    valid_recall_indices = np.where(recall > 0)[0]
    valid_indices = np.intersect1d(valid_precision_indices, valid_recall_indices)
    if len(valid_indices) > 0:
        precision_threshold_idx = valid_indices[np.argmax(recall[valid_indices])]
        if precision_threshold_idx >= len(pr_thresholds):
            precision_threshold_idx = len(pr_thresholds) - 1
        high_p_threshold = pr_thresholds[precision_threshold_idx]
        high_p_precision = precision[precision_threshold_idx]
        high_p_recall = recall[precision_threshold_idx]
    else:
        # If can't find threshold with precision > 0.9, look for precision > 0.85
        valid_precision_indices = np.where(precision > 0.8)[0]
        valid_indices = np.intersect1d(valid_precision_indices, valid_recall_indices)
        if len(valid_indices) > 0:
            precision_threshold_idx = valid_indices[np.argmax(recall[valid_indices])]
            if precision_threshold_idx >= len(pr_thresholds):
                precision_threshold_idx = len(pr_thresholds) - 1
            high_p_threshold = pr_thresholds[precision_threshold_idx]
            high_p_precision = precision[precision_threshold_idx]
            high_p_recall = recall[precision_threshold_idx]
        else:
            high_p_threshold = None
            high_p_precision = None
            high_p_recall = None

    # Find threshold for which recall is greater than 0.9 and precision is greater than 0.
    valid_recall_indices = np.where(recall > 0.9)[0]
    valid_precision_indices = np.where(precision > 0)[0]
    valid_indices = np.intersect1d(valid_precision_indices, valid_recall_indices)
    if len(valid_indices) > 0:
        recall_threshold_idx = valid_indices[np.argmax(precision[valid_indices])]
        if recall_threshold_idx >= len(pr_thresholds):
            recall_threshold_idx = len(pr_thresholds) - 1
        high_r_threshold = pr_thresholds[recall_threshold_idx]
        high_r_precision = precision[recall_threshold_idx]
        high_r_recall = recall[recall_threshold_idx]
    else:
        # If can't find threshold with recall > 0.9, look for recall > 0.85
        valid_recall_indices = np.where(recall > 0.8)[0]
        valid_indices = np.intersect1d(valid_precision_indices, valid_recall_indices)
        if len(valid_indices) > 0:
            recall_threshold_idx = valid_indices[np.argmax(precision[valid_indices])]
            if recall_threshold_idx >= len(pr_thresholds):
                recall_threshold_idx = len(pr_thresholds) - 1
            high_r_threshold = pr_thresholds[recall_threshold_idx]
            high_r_precision = precision[recall_threshold_idx]
            high_r_recall = recall[recall_threshold_idx]
        else:
            high_r_threshold = None
            high_r_precision = None
            high_r_recall = None
    
    optimal_idx_roc = find_closest_index(roc_thresholds, optimal_threshold)
    high_p_idx_roc = find_closest_index(roc_thresholds, high_p_threshold)
    high_r_idx_roc = find_closest_index(roc_thresholds, high_r_threshold)
    print('saving1')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})', zorder=1)
    plt.scatter(fpr[high_p_idx_roc], tpr[high_p_idx_roc], marker='o', color='red', s=100, label=f'High Precision Threshold: {high_p_threshold:.2f}', zorder=2)
    plt.scatter(fpr[optimal_idx_roc], tpr[optimal_idx_roc], marker='x', color='purple', s=100, label=f'Optimal F1 Threshold: {optimal_threshold:.2f}', zorder=2)
    # plt.scatter(fpr[high_r_idx_roc], tpr[high_r_idx_roc], marker='s', color='blue', s=100, label=f'High Recall Threshold : {high_r_threshold:.2f}', zorder=10)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if not os.path.exists('./results/roc_figs'):
        os.makedirs('./results/roc_figs')
    print('saving')
    plt.savefig(f"/home/ysvirk/calibration_code_summaries/results/roc_figs/{metric_name}_rocd.png")

    return {
        'roc_auc': roc_auc,
        'optimal_f1_threshold': optimal_threshold, 
        'optimal_f1_precision': optimal_precision, 
        'optimal_f1_recall': optimal_recall, 
        'high_p_threshold': high_p_threshold,
        'high_p_precision': high_p_precision,
        'high_p_recall': high_p_recall,
        'high_r_threshold': high_r_threshold,
        'high_r_precision': high_r_precision,
        'high_r_recall': high_r_recall,
    }

def calculate_f1_scores(metric_scores, true_int, metric_name):
    print(metric_name)
    precision, recall, pr_thresholds = precision_recall_curve(true_int, metric_scores)
    
    scores = {}
    for threshold in np.arange(0, 1.01, 0.01):
        # Find the index of the closest threshold
        idx = np.argmin(np.abs(pr_thresholds - threshold))
        
        if idx < len(precision) and idx < len(recall):
            p = precision[idx]
            r = recall[idx]
            if p + r > 0:  # Avoid division by zero
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0
        else:
            p = r = f1 = 0
        
        threshold_key = round(threshold, 2)
        scores[threshold_key] = {
            'f1': round(f1, 4),
            'precision': round(p, 4),
            'recall': round(r, 4)
        }

    return scores


if __name__ == "__main__":
    with open('/home/ysvirk/calibration_code_summaries/results/roc_figs/a.txt', 'w') as file:
        file.write('a.txt')
    haque_et_al_path = './data/haque_et_al/final_megafile.csv'
    df = pd.read_csv(haque_et_al_path)

    ## Similarity measures
    metric_name_to_col = {
        'BLEU-1': 'b1',
        'Infersent-CS': 'iS_cosine',
        'SentenceBert-CS': 'sb_cosine',
        'BERT Score-R': 'rbert'
    }
    metric_vals = {
        'ROUGE-1-P': [],
        'ROUGE-4-R': [],
        'ROUGE-W-R': [],
        'BLEU-1': [],
        'Infersent-CS': [],
        'SentenceBert-CS': [],
        'BERT Score-R': []
    }
    similarity_list = []
    rouge_scores_dict = pickle.load(open('./data/haque_et_al/rouge_score.pkl', 'rb'))

    fids = df['function_id'].unique()
    for i in range(len(fids)):
        fid = fids[i]
        df_baseline = df[(df['function_id'] == fid) & (df['source'] == 'baseline')]
        similarity_list.append(np.mean(df_baseline['similarity']))
        for metric_name, col_name in metric_name_to_col.items():
            metric_val = np.mean(df_baseline[col_name])
            if metric_name in ['Infersent-CS', 'SentenceBert-CS']:
                metric_val = rescale(metric_val)
            metric_vals[metric_name].append(metric_val)
        metric_vals['ROUGE-1-P'].append(float(rouge_scores_dict[fid][1][1]['p']))
        metric_vals['ROUGE-W-R'].append(float(rouge_scores_dict[fid][-1][1]['r']))
        metric_vals['ROUGE-4-R'].append(float(rouge_scores_dict[fid][4][1]['r']))

    good_similarity = (np.array(similarity_list) >= 3).astype(int)
    metric_threshold_stats = {}
    for metric_name, metric_scores in metric_vals.items():
        metric_threshold_stats[metric_name] = plot_roc_curve(metric_scores, good_similarity, metric_name)
        metric_threshold_stats[metric_name]= calculate_f1_scores(metric_scores, good_similarity, metric_name)

    # dump the stats
    # with open('./results/threshold_stats.json', 'w') as f:
    #     json.dump(metric_threshold_stats, f, indent=4)
    with open('./results/f1_scores_per_threshold.json', 'w') as f:
        json.dump(metric_threshold_stats, f, indent=4)
