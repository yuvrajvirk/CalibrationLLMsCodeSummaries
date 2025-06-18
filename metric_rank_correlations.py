import json
import os
from data_utils import parse_bleu_scores, extract_results
import numpy as np
import scipy

metric_correlations = {} # language -> model_name -> {metric: score}
for language in ["Java", "Python"]:
    metric_correlations[language] = {}
    gold_file_path = f"./data/{language}/prompting_data/gold.txt"
    results_dir = f"./data/{language}/model_outputs/"
    metric_paths = {
        "Sentence BERT": f"./data/{language}/metrics_results/sbert/",
        "BLEU-1": f"./data/{language}/metrics_results/bleu-1/",
        "Infersent-CS": f"./data/{language}/metrics_results/infersent/",
        "BERT Score": f"./data/{language}/metrics_results/bert-score-recall/",
        "ROUGE-1-P": f"./data/{language}/metrics_results/rouge-1-p/",
        "ROUGE-4-R": f"./data/{language}/metrics_results/rouge-4-r/",
        "ROUGE-W-R": f"./data/{language}/metrics_results/rouge-w-r/",
    }

    model_files = ["gpt-3.5-turbo_ASAP.txt", "gpt-3.5-turbo_BM25.txt", "CodeLlama-70b-hf_ASAP.txt", "CodeLlama-70b-hf_BM25.txt", "deepseek-coder-33b-instruct_ASAP.txt", "deepseek-coder-33b-instruct_BM25.txt"]
    for model_filename in model_files:
        model_name = model_filename[:-9]
        prompting_method = model_filename[-8:-4]
        if model_name not in metric_correlations[language]: metric_correlations[language][model_name] = {}
        metric_correlations[language][model_name][prompting_method] = {}
        for metric, metric_path in metric_paths.items():
            metric_correlations[language][model_name][prompting_method][metric] = {"Avg Logit": {}, "Sequence Logit": {}}
            results_path = os.path.join(results_dir, model_filename)
            print(model_filename)
            if (model_filename == "gpt-3.5-turbo_ASAP.txt" or model_filename == "gpt-3.5-turbo_BM25.txt") and language == "Java":
                model_results, invalid_results = extract_results(results_path, probs=False)
            else:
                model_results, invalid_results = extract_results(results_path, probs=True)
            log_probs_list = [result['log_probs'] for result in model_results.values()]
            avg_logits = [np.prod(log_probs) ** (1 / len(log_probs)) for log_probs in log_probs_list]
            sequence_logits = [np.prod(log_probs) for log_probs in log_probs_list]
            model_metric_path = os.path.join(metric_path, model_filename)
            model_metric_scores = parse_bleu_scores(model_metric_path)
            # Calculate spearman, kendalls, and pearson on between avg logit, sequence and metric score
            length = min(5000, len(model_metric_scores))
            metric_correlations[language][model_name][prompting_method][metric]["Avg Logit"]["Spearman"] = scipy.stats.spearmanr(avg_logits[:length], model_metric_scores[:length])
            metric_correlations[language][model_name][prompting_method][metric]["Avg Logit"]["Kendalls"] = scipy.stats.kendalltau(avg_logits[:length], model_metric_scores[:length])
            metric_correlations[language][model_name][prompting_method][metric]["Avg Logit"]["Pearson"] = scipy.stats.pearsonr(avg_logits[:length], model_metric_scores[:length])
            metric_correlations[language][model_name][prompting_method][metric]["Sequence Logit"]["Spearman"] = scipy.stats.spearmanr(sequence_logits[:length], model_metric_scores[:length])
            metric_correlations[language][model_name][prompting_method][metric]["Sequence Logit"]["Kendalls"] = scipy.stats.kendalltau(sequence_logits[:length], model_metric_scores[:length])
            metric_correlations[language][model_name][prompting_method][metric]["Sequence Logit"]["Pearson"] = scipy.stats.pearsonr(sequence_logits[:length], model_metric_scores[:length])

# Dump the benchmarks to a JSON file
with open(f"./results/metric_correlations.json", "w") as f:
    json.dump(metric_correlations, f, indent=4)
