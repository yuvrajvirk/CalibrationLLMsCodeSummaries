import os
import json
import re
import math

# Based on Automatic Semantic Augmentation of Language Model Prompts code.
# https://zenodo.org/records/7793516

def extract_gold(gold_file_path: str):
    gold_summaries = {}
    with open(gold_file_path, "r") as file:
        for line in file:
            parts = line.split("\t")
            i = int(parts[0])
            text = parts[1]
            gold_summaries[i] = text
    return gold_summaries

def parse_line_echo(line: str, probs, token_position_cutoff=None):
    parts = line.split("\t\t")
    if len(parts) < 5:
        return None, None
    try:
        i = int(parts[0])
    except:
        print("Error on: ", line)
        exit()
    # text = parts[1]
    # average = float(parts[2])
    # total = float(parts[3])
    if probs: 
        log_probs = [math.log(float(x)) for x in parts[4].strip('[]\n').split(',')]
        prompt_log_probs = [math.log(float(x)) for x in parts[5].strip('[]\n').split(',')]
        if token_position_cutoff:
            log_probs = log_probs[:token_position_cutoff]
    else:     
        log_probs = [float(x) for x in parts[4].strip('[]\n').split(',')]
        prompt_log_probs = [float(x) for x in parts[5].strip('[]\n').split(',')]
        if token_position_cutoff:
            log_probs = log_probs[:token_position_cutoff]
    measures = {'index': i, 'prompt_log_probs': prompt_log_probs}

    return i, measures

def parse_line(line: str, probs, token_position_cutoff=None):
     # indicates whether source is probs or log_probs, out is log_probs. This gives probs
    parts = line.split("\t\t")
    if len(parts) < 4:
        return None, None
    try:
        i = int(parts[0])
    except:
        print("Error on: ", line)
        exit()
    text = parts[1]
    if text == "":
        return None, None
    average = float(parts[2])
    total = float(parts[3])
    if probs: 
        log_probs = [math.log(float(x)) for x in parts[4].strip('[]\n').split(',')]
        if token_position_cutoff:
            log_probs = log_probs[:token_position_cutoff]
    else:     
        log_probs = [float(x) for x in parts[4].strip('[]\n').split(',')]
        if token_position_cutoff:
            log_probs = log_probs[:token_position_cutoff]
    measures = {'index': i, 'generated_summary': text, 'avg_prob': average, 'total_log_prob': total, 'log_probs': log_probs}

    return i, measures

def extract_results(results_file: str, probs=False, token_position_cutoff=None):
    '''
    Results file is generated in the following format:
    str(i)+"\t\t"+text+"\t\t"+str(average)+"\t\t"+str(total)+"\t\t"+str(log_probs)+"\n"
    '''
    results = {}
    invalid_results = []
    with open(results_file, 'r') as file:
        for line in file:
            i, result = parse_line(line, probs, token_position_cutoff)
            if i is None:
                invalid_results.append({"Empty response": line})
                continue
            results[i] = result

    print("Number of results included:", len(results))
    print("Results ommited due to invalid format:", len(invalid_results))
    return results, invalid_results

import math
import numpy as np

def parse_bleu_scores(file_path) -> dict:
    with open(file_path, 'r') as file:
        bleu_scores = []
        for line in file:
            index, score = line.strip().split('\t')
            bleu_scores.append(float(score))
        return bleu_scores

def parse_use_scores(file_path):
    # return indices
    with open(file_path, 'r') as file:
        bleu_scores = []
        indices = []
        for line in file:
            index, score = line.strip().split('\t')
            bleu_scores.append(float(score))
            indices.append(index)
        return indices, bleu_scores

def bucket(values: list[float], interval: float = 0.05, omit_empty_intervals: bool = False) -> dict[float:list[int]]:
    """
    Given a list of values, this function returns a dictionary where the keys are intervals from 0 to 1 (with a step size defined by 'interval'),
    and the values are lists of indices from the original list where the corresponding values fall within the key's interval.
    """
    buckets = {}
    interval_count = int(1/interval)
    if not omit_empty_intervals:
        for i in range(interval_count):
            buckets[i*interval] = []
    for i, value in enumerate(values):
        bucket = int(value / interval)
        if bucket == interval_count: bucket = interval_count - 1
        if bucket*interval not in buckets.keys():
            buckets[bucket*interval] = []
        buckets[bucket*interval].append(i)
    return buckets
    

def count_above_threshold(bucket_indices, bleu, correct_threshold):
    correct_bleu_count = 0
    for i in bucket_indices:
        bleu_score = bleu[i]
        if bleu_score >= correct_threshold:
            correct_bleu_count += 1
    return correct_bleu_count

def check_avg_probs(results):
    count = 0
    for result in results.keys():
        if calc_avg_probs(result['log_probs']) != result["avg_prob"]:
            count += 1
    print(count, "/", len(results))

def get_aligned_lists(all_bleu_scores, results):
    included_bleu_scores = []
    for key in results.keys():
        included_bleu_scores.append(all_bleu_scores[key])
    return included_bleu_scores

def get_probs(results):
    probs = []
    for key in results.keys():
        log_probs_per_summary = results[key]["log_probs"]
        probs_per_summary = np.exp(log_probs_per_summary)
        probs.append(np.array(probs_per_summary))
    return probs

def get_avg_probs(results):
    probs = []
    for key in results.keys():
        avg_prob = results[key]["avg_prob"]
        probs.append(avg_prob)
    return probs

def get_geometric_mean_probs(results):
    probs_per_summary = get_probs(results)
    return [np.prod(probs) ** (1.0 / len(probs)) for probs in probs_per_summary]

def get_binary_scores(bleu_scores, correct_threshold):
    binary_scores = []
    for bleu_score in bleu_scores:
        if bleu_score >= correct_threshold:
            binary_scores.append(1)
        else:
            binary_scores.append(0)
    return np.array(binary_scores)

#### Token score calculations
def avg_prob(list_of_probs_lists: list[np.array]):
    return np.array([np.average(probs) for probs in list_of_probs_lists])

def max_prob(list_of_probs_lists: list[np.array]):
    return np.array([np.max(probs) for probs in list_of_probs_lists])

def median_prob(list_of_probs_lists: list[np.array]):
    return np.array([np.median(probs) for probs in list_of_probs_lists])

def avg_entropy(list_of_probs_lists: list[np.array]):
    avg_entropies = []
    for probs in list_of_probs_lists:
        avg_entropies.append(-np.average([p * np.log2(p) for p in probs if p > 0]))
    return avg_entropies

def max_entropy(list_of_probs_lists: list[np.array]):
    max_entropies = []
    for probs in list_of_probs_lists:
        max_entropies.append(-np.max([p * np.log2(p) for p in probs if p > 0]))
    return max_entropies

def avg_likelihood(list_of_probs_lists: list[np.array]):
    avg_likelihoods = []
    for probs in list_of_probs_lists:
        avg_likelihoods.append(np.average(-np.log(probs)))
    return avg_likelihoods

def max_likelihood(list_of_probs_lists: list[np.array]):
    max_likelihoods = []
    for probs in list_of_probs_lists:
        max_likelihoods.append(np.max(-np.log(probs)))
    return max_likelihoods

def median_likelihood(list_of_probs_lists: list[np.array]):
    median_likelihoods = []
    for probs in list_of_probs_lists:
        median_likelihoods.append(np.median(-np.log(probs)))
    return median_likelihoods

def calc_intersection_probs(log_probs: np.array):
    conditional_probs = np.exp(log_probs)
    # P(next token|prev tokens) = P(next token and prev tokens) / P(prev tokens)'
    # = P(next token)/