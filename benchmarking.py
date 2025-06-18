import json
import os
from data_utils import parse_bleu_scores, extract_results
import numpy as np

benchmarks = {} # language -> model_name -> {metric: score}
for language in ["Java", "Python"]:
    benchmarks[language] = {}
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
        if model_name not in benchmarks[language]: benchmarks[language][model_name] = {}
        benchmarks[language][model_name][prompting_method] = {}
        for metric, metric_path in metric_paths.items():
            benchmarks[language][model_name][prompting_method][metric] = {}
            results_path = os.path.join(results_dir, model_filename)
            model_results, invalid_results = extract_results(results_path)
            model_metric_path = os.path.join(metric_path, model_filename)
            model_metric_scores = parse_bleu_scores(model_metric_path)
            benchmarks[language][model_name][prompting_method][metric]["Number Invalid"] = len(invalid_results)
            length = min(5000, len(model_metric_scores))
            benchmarks[language][model_name][prompting_method][metric]["Exclude Invalid"] = np.sum(model_metric_scores[:length])/length
            benchmarks[language][model_name][prompting_method][metric]["Include Invalid"] = np.sum(model_metric_scores[:length])/(length+len(invalid_results))

# Dump the benchmarks to a JSON file
with open(f"./results/benchmarks.json", "w") as f:
    json.dump(benchmarks, f, indent=4)
