import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from data_utils import *
from calibration_metrics import get_calibration_metrics_raw, get_scaled_metrics_repo_split_kfold, rescale
import os 
from reliability_plot import draw_reliability_graph

raw_evaluations = {} 
scaled_evaluations = {}
threshold_stats = json.loads(open("./results/threshold_stats.json", "r").read())
for language in ["Java", "Python"]:
    metric_paths = {
        "SentenceBert-CS": f"./data/{language}/metrics_results/sbert/",
        "BLEU-1": f"./data/{language}/metrics_results/bleu-1/",
        "Infersent-CS": f"./data/{language}/metrics_results/infersent/",
        "BERT Score-R": f"./data/{language}/metrics_results/bert-score-recall/",
        "ROUGE-1-P": f"./data/{language}/metrics_results/rouge-1-p/",
        "ROUGE-4-R": f"./data/{language}/metrics_results/rouge-4-r/",
        "ROUGE-W-R": f"./data/{language}/metrics_results/rouge-w-r/",
    }
    raw_evaluations[language] = {}
    scaled_evaluations[language] = {}
    gold_file_path = f"./data/{language}/prompting_data/gold.txt"
    results_dir = f"./data/{language}/model_outputs/"
    # Change with desited model files
    model_files = [f"./results/self_reflection_results/results_gpt-3.5-turbo_{language}_0_T0_reflective_logit_True2.json"]
    for i, model_filename in enumerate(model_files):
        model_name="gpt-3.5-turbo"
        metrics_filename="gpt-3.5-turbo_ASAP.txt"
        results_path = os.path.join(results_dir, metrics_filename)
        if model_name not in raw_evaluations[language]: 
            raw_evaluations[language][model_name] = {}
            scaled_evaluations[language][model_name] = {}
        model_results = json.loads(open(model_filename, "r").read())
        df = pd.DataFrame(model_results['0'])
        rating_logprobs = df['Rating logprobs']
        probs = rating_logprobs.apply(lambda x: np.exp(np.array(x).squeeze()))
        model_og_results, invalid_results = extract_results(results_path)
        probs_dict = dict(zip(model_og_results.keys(), probs))
        for metric, metric_path in metric_paths.items():
            model_metric_path = os.path.join(metric_path, metrics_filename)
            indices, model_metric_scores = parse_use_scores(model_metric_path)
            # Model metric scores and probs align onto the same set of indices
            # because both use results parsed through extract_results - so same
            # set of invalid results removed.
            model_metric_scores = model_metric_scores[:len(probs)]
            model_metric_scores_dict = dict(zip(model_og_results.keys(), model_metric_scores))
            raw_evaluations[language][model_name][metric] = {"Probs": {}}
            
            if metric in ['SentenceBERT_CS']:
                model_metric_scores = rescale(model_metric_scores)
            
            thresholds = threshold_stats[metric] 
            calibration_metrics, calibration_metrics_json = get_calibration_metrics_raw(predicted_correctness=probs, actual_correctness=model_metric_scores, thresholds=thresholds, model_filename=model_filename)
            calibration_metrics_json['spearman_r'] = spearmanr(model_metric_scores, probs)
            raw_evaluations[language][model_name][metric] = calibration_metrics_json
            draw_reliability_graph(calibration_metrics["optimal_f1_threshold"]["ECE"], 
                    calibration_metrics["optimal_f1_threshold"]["p_correct"],
                    calibration_metrics["optimal_f1_threshold"]["Skill score"], 
                    calibration_metrics["optimal_f1_threshold"]["Brier Score"],
                    calibration_metrics["optimal_f1_threshold"]["Samples Per Bin"],
                    calibration_metrics["optimal_f1_threshold"]["Unskilled Brier Score"],
                    calibration_metrics["optimal_f1_threshold"]["Bins"], 
                    calibration_metrics["optimal_f1_threshold"]["Bin Accuracies"],
                    calibration_metrics["optimal_f1_threshold"]["Bin Confs"], 
                    calibration_metrics["optimal_f1_threshold"]["Bin Sizes"], 
                    f"./results/self_reflection_results/raw_reflective_reliability_figs/{language}/", 
                    f"{metrics_filename}_{metric}_avg_logit",
                    raw=True)
            calibration_metrics, calibration_metrics_json = get_scaled_metrics_repo_split_kfold(predicted_scores=probs_dict,
                actual_scores=model_metric_scores_dict,
                thresholds=thresholds,
                test_data=f"./data/{language}/prompting_data/test.jsonl",
                num_bins = 10,
                model_filename=model_filename)
            calibration_metrics_json['spearman_r'] = spearmanr(model_metric_scores, probs)
            scaled_evaluations[language][model_name][metric] = calibration_metrics_json
            draw_reliability_graph(calibration_metrics["optimal_f1_threshold"]["ECE"], 
                                calibration_metrics["optimal_f1_threshold"]["p_correct"],
                                calibration_metrics["optimal_f1_threshold"]["Skill score"], 
                                calibration_metrics["optimal_f1_threshold"]["Brier Score"],
                                calibration_metrics["optimal_f1_threshold"]["Samples Per Bin"],
                                calibration_metrics["optimal_f1_threshold"]["Unskilled Brier Score"],
                                calibration_metrics["optimal_f1_threshold"]["Bins"], 
                                calibration_metrics["optimal_f1_threshold"]["Bin Accuracies"],
                                np.array(calibration_metrics["optimal_f1_threshold"]["Bin Confs"]), 
                                np.array(calibration_metrics["optimal_f1_threshold"]["Bin Sizes"]), 
                                f"./results/self_reflection_results/rescaled_reflective_reliability_figs/{language}/", 
                                f"{metrics_filename}_{metric}_avg_prob")
        
        with open(f"./results/self_reflection_results/reflective_logit_calibration.json", "w") as f:
            json.dump(raw_evaluations, f, indent=4)
        with open(f"./results/self_reflection_results/rescaled_reflective_logit_calibration.json", "w") as f:
            json.dump(scaled_evaluations, f, indent=4)