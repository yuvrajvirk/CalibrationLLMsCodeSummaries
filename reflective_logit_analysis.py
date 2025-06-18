import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from reliability_plot import calc_moe

results_turbo_java_0 = json.load(open('./results/self_reflection_results/results_gpt-3.5-turbo_Java_0_T0_reflective_logit_False2.json')) 
results_turbo_python_0 = json.load(open('./results/self_reflection_results/results_gpt-3.5-turbo_Python_0_T0_reflective_logit_False2.json')) 
results_turbo_java_4 = json.load(open('./results/self_reflection_results/results_gpt-3.5-turbo_Java_4_T0_reflective_logit_False2.json')) 
results_turbo_python_4 = json.load(open('./results/self_reflection_results/results_gpt-3.5-turbo_Python_4_T0_reflective_logit_False2.json')) 
results_deepseek_java_0 = json.load(open('./results/self_reflection_results/results_deepseek-coder-33b-instruct_Java_0_T0_reflective_logit_False2.json')) 
results_deepseek_python_0 = json.load(open('./results/self_reflection_results/results_deepseek-coder-33b-instruct_Python_0_T0_reflective_logit_False2.json')) 
results_deepseek_java_4 = json.load(open('./results/self_reflection_results/results_deepseek-coder-33b-instruct_Java_4_T0_reflective_logit_False2.json')) 
results_deepseek_python_4 = json.load(open('./results/self_reflection_results/results_deepseek-coder-33b-instruct_Python_4_T0_reflective_logit_False2.json')) 
results_codellama_python_4 = json.load(open('./results/self_reflection_results/results_CodeLlama-70b-hf_Java_4_T0_reflective_logit_False2.json')) 
results_codellama_java_4 = json.load(open('./results/self_reflection_results/results_CodeLlama-70b-hf_Python_4_T0_reflective_logit_False2.json')) 
results_codellama_java_4_high_T = json.load(open('./results/self_reflection_results/results_CodeLlama-70b-hf_Java_4_T0.7_reflective_logit_False2.json')) 

results_list = [results_turbo_java_0, results_deepseek_java_0, results_turbo_python_0, results_deepseek_python_0, results_turbo_java_4, results_deepseek_java_4, results_turbo_python_4,  results_deepseek_python_4, results_codellama_java_4, results_codellama_python_4, results_codellama_java_4_high_T]
dfs = {}

def rescale(model_metric_scores):
    return (np.array(model_metric_scores) + 1) / 2

shots = 0
names = ["turbo_java_0", "deepseek_java_0", "turbo_python_0", "deepseek_python_0", "turbo_java_4", "deepseek_java_4", "turbo_python_4", "deepseek_python_4", "codellama_java_4", "codellama_python_4", "codellama_java_4_high_T"]
for i, result in enumerate(results_list):
    if i >= 4: shots = 4
    df = pd.DataFrame(result[str(shots)])  # Correctly create DataFrame from the result object
    df['LLM Rating'] = pd.to_numeric(df['LLM Rating'], errors='coerce').fillna(0).astype(int)
    dfs[names[i]] = df


def empircal_prob_correct(df, rating):
    SentenceBERT_CS = df[df['LLM Rating'] == rating]['SentenceBERT_CS']
    sample_size = len(SentenceBERT_CS)
    return (SentenceBERT_CS >= 0.8936890500000001).sum() / sample_size

def reliability_plot(df, save_name, save=False):
    save_path = "./results/self_reflection_results/verbalized_confidence_figs"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bins = [1, 2, 3, 4]
    index_x = [b - 0.05 for b in bins]

    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])  # Define a grid with 2 rows and 1 column, with different heights
    
    df = df[df['LLM Rating'] != 0]

    ax0 = plt.subplot(gs[0])
    ax0.set_ylim(0, 100)
    ax0.set_xticks([1, 2, 3, 4])
    ax0.set_ylabel("% of Samples", fontsize=24)
    samples_per_bin = [np.sum(df['LLM Rating'] == 1), np.sum(df['LLM Rating'] == 2), np.sum(df['LLM Rating'] == 3), np.sum(df['LLM Rating'] == 4)]
    bin_sizes = [((samples_per_bin[i] / len(df)) * 100) for i in range(len(samples_per_bin))] 
    ax0.bar([1, 2, 3, 4], bin_sizes, width=1.0, alpha=1, edgecolor='black')  # Adjusted to center bars and modified width for better visualization
    for i, j in zip(range(1, 5), bin_sizes):
        ax0.text(i, j, "%.1f" % (j), ha="center", va="bottom", fontsize=18)
    ax0.tick_params(labelsize=20)
    
    ax1 = plt.subplot(gs[1])
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Verbalized Confidence Rating", fontsize=24)
    ax1.set_ylabel("Model Accuracy", fontsize=24)
    ax1.set_axisbelow(True)
    ax1.grid(color='gray')

    bin_accs = [empircal_prob_correct(df, rating) for rating in bins]
    bars = ax1.bar(index_x, bin_accs, width=1.0, alpha=1, edgecolor='black', label='Output')
    for i, bar in enumerate(bars):
        moe = calc_moe(samples_per_bin[i], bin_accs[i])
        ax1.errorbar(bar.get_x() + bar.get_width() / 2, bin_accs[i], yerr=moe, fmt='o', color='black', linewidth=5)
    ax1.tick_params(labelsize=20)

    if save: plt.savefig(os.path.join(save_path, '{}-reliability.png'.format(save_name)), bbox_inches='tight')
       
for save_name, df in dfs.items(): reliability_plot(df, save_name, save=True)
