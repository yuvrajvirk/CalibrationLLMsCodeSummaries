import matplotlib.pyplot as plt
import numpy as np
import os
import json
from data_utils import extract_results, parse_use_scores, get_binary_scores, get_avg_probs, get_geometric_mean_probs
from platt_scale import platt_rescale, isotonic_rescale, beta_rescale, platt_rescale_no_log_odds
from reliability_plot import draw_reliability_graph
from sklearn import metrics

def calculate_calibration_metrics(predicted_probabilities, 
                                true_labels,
                                num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
 
    # get binary class predictions from confidences
    predicted_labels = np.ones_like(predicted_probabilities)

    # get a boolean list of correct/false predictions
    accuracies = predicted_labels == true_labels
    bin_accuracies = []
    samples_per_bin = []
    binned_pred_probs = []
    binned_true_labels = []

    ece = 0  
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(
            predicted_probabilities > bin_lower.item(),
            predicted_probabilities <= bin_upper.item(),
        )
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        probability_in_bin = in_bin.astype(float).mean()

        if probability_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            bin_accuracies.append(accuracy_in_bin)

            labels_in_bin = true_labels[in_bin]
            binned_pred_probs.append(predicted_probabilities[in_bin])
            binned_true_labels.append(labels_in_bin)
            samples_per_bin.append(len(labels_in_bin))
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = predicted_probabilities[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            bin_ece = (
                np.abs(avg_confidence_in_bin - accuracy_in_bin) * probability_in_bin
            )
            ece += bin_ece
        else:
            bin_accuracies.append(np.nan)
            binned_pred_probs.append([])
            binned_true_labels.append([])
            samples_per_bin.append(0)

    brier_score = metrics.brier_score_loss(
        true_labels,
        predicted_probabilities,
    )
    p_correct = np.average(true_labels)
    unskilled_brier_score = p_correct * (1 - p_correct)
    skill_score = (unskilled_brier_score - brier_score) / unskilled_brier_score

    return {
        "ECE": ece,
        "p_correct": p_correct,
        "Brier Score": brier_score, 
        "Unskilled Brier Score": unskilled_brier_score, 
        "Skill score": skill_score,
        "Bins": np.linspace(0, 1, num_bins + 1),
        "Bin Confs": [np.mean(probs) for probs in binned_pred_probs],
        "Bin Accuracies": bin_accuracies,
        "Bin Sizes": np.array(samples_per_bin)/len(predicted_probabilities),
        "Num Samples": len(predicted_probabilities),
        "Samples Per Bin": samples_per_bin,
    }

def rescale(model_metric_scores):
    return (np.array(model_metric_scores) + 1) / 2

def scores_with_repo(use_scores_dict, test_data_path: str):
    repo_to_scores = {}
    test_json = []
    for line in open(test_data_path, 'r', encoding="utf-8"):
        test_json.append(json.loads(line))

    for i, score in use_scores_dict.items():
        repo = test_json[int(i)]['repo']
        if repo not in repo_to_scores:
            repo_to_scores[repo] = []
        repo_to_scores[repo].append({"index": i, "score": score})

    return repo_to_scores

def train_test_split_by_repo_balanced(repo_to_scores, num_folds=5):
    # Flatten the scores to calculate total number of samples
    total_samples = sum(len(scores) for scores in repo_to_scores.values())
    target_samples_per_fold = total_samples / num_folds
    
    # Convert repo_to_scores to a list of items, sorted by the number of samples per repo
    items = sorted(repo_to_scores.items(), key=lambda x: len(x[1]), reverse=True)
    
    folds = []
    tests_used = []
    for fold in range(num_folds):
        train_sample, test_sample = [], []
        train_metadata, test_metadata = {}, {}
        test_samples_count = 0
        
        # Dynamically allocate repositories to train or test to balance the number of samples
        for repo, scores in items:
            # If adding this repo to test won't exceed the target number of samples for the fold, add it to test
            if (repo not in tests_used) and (test_samples_count + len(scores) <= target_samples_per_fold):
                test_sample += scores
                test_metadata[repo] = len(scores)
                test_samples_count += len(scores)
                tests_used.append(repo)
            else:
                # Otherwise, add it to train
                train_sample += scores
                train_metadata[repo] = len(scores)
        
        # Save metadata and samples for the fold
        # with open(f'train_metadata_{fold}.json', 'w') as file:
        #     json.dump(train_metadata, file)
        # with open(f'test_metadata_{fold}.json', 'w') as file:
        #     json.dump(test_metadata, file)
        
        print(f"Fold {fold}")
        print(f"Train Length: {len(train_sample)}")
        print(f"Train Num Repos: {len(train_metadata.keys())}")
        print(f"Test Length: {len(test_sample)}")
        print(f"Test Num Repos: {len(test_metadata.keys())}")
        
        folds.append((train_sample, test_sample))
    
    return folds

def get_scaled_metrics_repo_split_kfold(predicted_scores, 
                                        actual_scores,
                                        thresholds,
                                        test_data,
                                        num_bins,
                                        model_filename,
                                        filter_in=None):
    fig_scaled, ax_scaled = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)   
    repo_to_scores = scores_with_repo(actual_scores, test_data)
    folds = train_test_split_by_repo_balanced(repo_to_scores, 5) 
    for i in range(5):
        print(f"Fold {i} train scores length:", len(folds[i][0]))
        print(f"Fold {i} test scores length:", len(folds[i][1]))
    combined_test_scores = []
    for _, test in folds:
        for test_sample in test:
            test_i = int(test_sample["index"])
            combined_test_scores.append(test_i)

    scaled_evaluations_json = {}
    scaled_evaluations = {}
    thresholds["optimal_f1_threshold"]=0.89
    for threshold_label, threshold in thresholds.items():
        if threshold_label not in ("optimal_f1_threshold"): continue        
        true_binary_scores = {int(index): actual_scores[index]>threshold for index in actual_scores.keys()}
        all_scaled_test_probs = np.array([])
        all_test_true_labels = np.array([])
        for train, test in folds:
            train_probs = np.array([])
            test_probs = np.array([])
            train_true_labels = np.array([])
            test_true_labels = np.array([])
            for train_sample in train:
                train_i = int(train_sample["index"])
                train_probs = np.append(train_probs, predicted_scores[train_i])
                train_true_labels = np.append(train_true_labels, true_binary_scores[train_i])            
            for test_sample in test:
                test_i = int(test_sample["index"])
                if filter_in is not None and test_i not in filter_in:
                    continue
                test_probs = np.append(test_probs, predicted_scores[test_i])
                test_true_labels = np.append(test_true_labels, true_binary_scores[test_i])

            test_scaled_probs = platt_rescale(train_probs, 
                                            train_true_labels, 
                                            test_probs, 
                                            test_true_labels)
            all_scaled_test_probs = np.append(all_scaled_test_probs, test_scaled_probs)
            all_test_true_labels = np.append(all_test_true_labels, test_true_labels)
        
        scaled_results = calculate_calibration_metrics(predicted_probabilities=np.array(all_scaled_test_probs), 
        true_labels=all_test_true_labels,
        num_bins=10)
        scaled_results["Threshold"] = threshold
        scaled_evaluations[threshold_label] = scaled_results
        scaled_results["Bin Confs"] = list(scaled_results["Bin Confs"])
        scaled_results["Bin Sizes"] = list(scaled_results["Bin Sizes"])

        scaled_evaluations_json[threshold_label] = {
            "ECE": scaled_results["ECE"],
            "p_correct": scaled_results["p_correct"],
            "Brier Score": scaled_results["Brier Score"], 
            "Unskilled Brier Score": scaled_results["Unskilled Brier Score"], 
            "Skill score": scaled_results["Skill score"],
            "Threshold": threshold
        } 
        
    return scaled_evaluations, scaled_evaluations_json
                
def get_calibration_metrics_raw(predicted_correctness: list[float], 
                                actual_correctness: list[float],
                                thresholds: list,
                                model_filename: str,
                                avg: bool = False):
    raw_evaluations = {}
    raw_evaluations_json = {}
    thresholds["optimal_f1_threshold"] = 0.49
    for threshold_label, threshold in thresholds.items():
        if threshold_label not in ("optimal_f1_threshold"): continue        
        true_binary_scores = get_binary_scores(actual_correctness, threshold)
        
        # raw_results = ExperimentResults(predicted_probabilities=np.array(predicted_correctness), 
        #                                 true_labels=true_binary_scores,
        #                                 bin_strategy=BinStrategy.Uniform,
        #                                 num_bins=10)
        raw_results = calculate_calibration_metrics(predicted_probabilities=np.array(predicted_correctness), 
        true_labels=true_binary_scores,
        num_bins=10)
        raw_results["Threshold"] = threshold
        raw_evaluations[threshold_label] = raw_results

        raw_evaluations_json[threshold_label] = {
            "ECE": raw_results["ECE"],
            "p_correct": raw_results["p_correct"],
            "Brier Score": raw_results["Brier Score"], 
            "Unskilled Brier Score": raw_results["Unskilled Brier Score"], 
            "Skill score": raw_results["Skill score"],
            "Threshold": threshold
        } 
    return raw_evaluations, raw_evaluations_json

def aligned_lists_from_dictionary(dict1, dict2):
    common_keys = dict1.keys() & dict2.keys()
    values_from_dict1 = [dict1[key] for key in common_keys]
    values_from_dict2 = [dict2[key] for key in common_keys]
    return values_from_dict1, values_from_dict2

def odds(probs):
    odds=[]
    for prob in probs:
        if prob == 1:
            odds.append(1)
        else:
            odds.append(prob / (1 - prob))
    return odds

if __name__ == "__main__":
    raw_evaluations = {} # language -> model_name -> {metric: score}
    scaled_evaluations = {} # language -> model_name -> {metric: score}
    token_position_cutoff = 100
    for language in ["Python", "Java"]:
        raw_evaluations[language] = {}
        scaled_evaluations[language] = {}
        gold_file_path = f"./data/{language}/prompting_data/gold.txt"
        results_dir = f"./data/{language}/model_outputs/"
        metric_paths = {
            # "SentenceBert-CS": f"./data/{language}/metrics_results/sbert/",
            # "BLEU-1": f"./data/{language}/metrics_results/bleu-1/",
            # "Infersent-CS": f"./data/{language}/metrics_results/infersent/",
            "BERT Score-R": f"./data/{language}/metrics_results/bert-score-recall/",
            # "ROUGE-1-P": f"./data/{language}/metrics_results/rouge-1-p/",
            # "ROUGE-4-R": f"./data/{language}/metrics_results/rouge-4-r/",
            # "ROUGE-W-R": f"./data/{language}/metrics_results/rouge-w-r/",
        }

        model_files = ["gpt-3.5-turbo_ASAP.txt", "gpt-3.5-turbo_BM25.txt", "CodeLlama-70b-hf_ASAP.txt", "CodeLlama-70b-hf_BM25.txt", "deepseek-coder-33b-instruct_ASAP.txt", "deepseek-coder-33b-instruct_BM25.txt"]
        # model_files = ["CodeLlama-70b-hf_BM25.txt"]
        for model_filename in model_files:
            model_name = model_filename[:-9]
            prompting_method = model_filename[-8:-4]

            if model_name not in raw_evaluations[language]: 
                raw_evaluations[language][model_name] = {}
                scaled_evaluations[language][model_name] = {}
            raw_evaluations[language][model_name][prompting_method] = {}
            scaled_evaluations[language][model_name][prompting_method] = {}
            results_path = os.path.join(results_dir, model_filename)
            decontaminated_results_path = f"data/{language}/model_outputs/{model_filename}_filtered_contaminants.txt"

            # Some log probs are actually probs in the data
            # get them in log probs and convert all into probs here
            if (model_filename == "gpt-3.5-turbo_ASAP.txt" or model_filename == "gpt-3.5-turbo_BM25.txt") and language == "Java":
                model_results, invalid_results = extract_results(results_path, probs=False, token_position_cutoff=token_position_cutoff)
                decontaminated_model_results, decontaminated_invalid_results = extract_results(decontaminated_results_path, probs=False, token_position_cutoff=token_position_cutoff)
            else:
                model_results, invalid_results = extract_results(results_path, probs=True, token_position_cutoff=token_position_cutoff)
                decontaminated_model_results, decontaminated_invalid_results = extract_results(decontaminated_results_path, probs=True, token_position_cutoff=token_position_cutoff)

            # model_results = decontaminated_model_results

            probs_per_summary = [np.exp(result['log_probs']) for result in model_results.values()]
            odds_per_summary = [odds(probs) for probs in probs_per_summary]
            log_probs_per_summary = [result['log_probs'] for result in model_results.values()]
            
            avg_probs = [np.mean(probs) for probs in probs_per_summary]
            avg_probs_dict = {int(result['index']): np.mean(np.exp(result['log_probs'])) for result in model_results.values()}

            # Get the list of indices from decontaminated_model_results
            # decontaminated_indices = [int(result['index']) for result in decontaminated_model_results.values()]

            median_probs_dict = {int(result['index']): np.median(np.exp(result['log_probs'])) for result in model_results.values()}

            avg_log_probs = [np.mean(log_probs) for log_probs in log_probs_per_summary]
            avg_log_probs_dict = {int(result['index']): avg_log_prob for result, avg_log_prob in zip(model_results.values(), avg_log_probs)}

            geom_means_probs = [np.prod(probs) ** (1.0 / len(probs)) for probs in probs_per_summary]
            geom_means_probs_dict = {int(result['index']): geom_mean for result, geom_mean in zip(model_results.values(), geom_means_probs)}
            
            a=1
            geom_means_odds = [np.prod(odds) ** (1.0 / len(odds)) for odds in odds_per_summary]
            geom_means_odds_prob = [(geom_mean_odds**a)/(1 + (geom_mean_odds**a)) for geom_mean_odds in geom_means_odds]
            geom_means_odds_prob = [0 if np.isnan(geom_mean_prob) else geom_mean_prob for geom_mean_prob in geom_means_odds_prob]
            geom_means_odds_prob_dict = {int(result['index']): geom_mean_prob for result, geom_mean_prob in zip(model_results.values(), geom_means_odds_prob)}

            sequence_probs = [np.prod(prob) for prob in probs_per_summary]
            sequence_probs_dict = {int(result['index']): np.prod(result['log_probs']) for result in model_results.values()}

            aggergate_probs_dict = geom_means_probs_dict
            aggregate_measure = f"geometric mean"

            for metric, metric_path in metric_paths.items():
                raw_evaluations[language][model_name][prompting_method][metric] = {"Avg Logit": {}, "Sequence Logit": {}}
                scaled_evaluations[language][model_name][prompting_method][metric] = {"Avg Logit": {}, "Sequence Logit": {}}

                model_metric_path = os.path.join(metric_path, model_filename)
                indices, model_metric_scores = parse_use_scores(model_metric_path)
                length = min(5000, len(model_metric_scores))
                model_metric_scores = model_metric_scores[:length]
                model_metric_scores_dict = {int(index): score for index, score in zip(indices, model_metric_scores) if int(index) in aggergate_probs_dict.keys()}

                if metric in ['Infersent-CS', 'SentenceBert-CS']:
                    model_metric_scores = rescale(model_metric_scores)
                
                threshold_stats = json.loads(open("./results/threshold_stats.json", "r").read())
                thresholds = threshold_stats[metric] 
                predicted_correctness, actual_correctness = aligned_lists_from_dictionary(aggergate_probs_dict, model_metric_scores_dict)
                
                if not os.path.exists(f"./results/raw_reliability_figs/{language}/{aggregate_measure}/{metric}/token_position_cutoff_{token_position_cutoff}"):
                    os.makedirs(f"./results/raw_reliability_figs/{language}/{aggregate_measure}/{metric}/token_position_cutoff_{token_position_cutoff}")
    
                calibration_metrics, calibration_metrics_json = get_calibration_metrics_raw(predicted_correctness=predicted_correctness,
                actual_correctness=actual_correctness,
                thresholds=thresholds,
                model_filename=model_filename)
                raw_evaluations[language][model_name][prompting_method][metric]['Avg Logit'] = calibration_metrics_json
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
                                    f"./results/raw_reliability_figs/{language}/{aggregate_measure}/{metric}/token_position_cutoff_{token_position_cutoff}", 
                                    f"high_p_{model_filename[:-4]}_{aggregate_measure}",
                                    raw=True)
                # draw_reliability_graph(calibration_metrics["experimental"]["ECE"], 
                #                     calibration_metrics["experimental"]["p_correct"],
                #                     calibration_metrics["experimental"]["Skill score"], 
                #                     calibration_metrics["experimental"]["Brier Score"],
                #                     calibration_metrics["experimental"]["Samples Per Bin"],
                #                     calibration_metrics["experimental"]["Unskilled Brier Score"],
                #                     calibration_metrics["experimental"]["Bins"], 
                #                     calibration_metrics["experimental"]["Bin Accuracies"],
                #                     calibration_metrics["experimental"]["Bin Confs"], 
                #                     calibration_metrics["experimental"]["Bin Sizes"], 
                #                     f"./results/raw_reliability_figs/{language}/{aggregate_measure}/{metric}/experimental_thresh_{token_position_cutoff}", 
                #                     f"{model_filename[:-4]}_{aggregate_measure}",
                #                     raw=True)
                
                calibration_metrics, calibration_metrics_json = get_scaled_metrics_repo_split_kfold(predicted_scores=aggergate_probs_dict,
                                                actual_scores=model_metric_scores_dict,
                                                thresholds=thresholds,
                                                test_data=f"./data/{language}/prompting_data/test.jsonl",
                                                num_bins = 10,
                                                model_filename=model_filename)
                scaled_evaluations[language][model_name][prompting_method][metric]['Avg Logit'] = calibration_metrics_json

                if not os.path.exists(f"./results/rescaled_reliability_figs/{language}/{aggregate_measure}/{metric}/experimental_thresh_{token_position_cutoff}"):
                    os.makedirs(f"./results/rescaled_reliability_figs/{language}/{aggregate_measure}/{metric}/experimental_thresh_{token_position_cutoff}")

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
                                    f"./results/rescaled_reliability_figs/{language}/{aggregate_measure}/{metric}/token_position_cutoff_{token_position_cutoff}", 
                                    f"high_p_{model_filename[:-4]}_{aggregate_measure}")
                # draw_reliability_graph(calibration_metrics["experimental"]["ECE"], 
                #                     calibration_metrics["experimental"]["p_correct"],
                #                     calibration_metrics["experimental"]["Skill score"], 
                #                     calibration_metrics["experimental"]["Brier Score"],
                #                     calibration_metrics["experimental"]["Samples Per Bin"],
                #                     calibration_metrics["experimental"]["Unskilled Brier Score"],
                #                     calibration_metrics["experimental"]["Bins"], 
                #                     calibration_metrics["experimental"]["Bin Accuracies"],
                #                     np.array(calibration_metrics["experimental"]["Bin Confs"]), 
                #                     np.array(calibration_metrics["experimental"]["Bin Sizes"]), 
                #                     f"./results/rescaled_reliability_figs/{language}/{aggregate_measure}/{metric}/experimental_thresh_{token_position_cutoff}", 
                #                     f"{model_filename[:-4]}_{aggregate_measure}")
    with open(f"./results/raw_evaluations_{aggregate_measure}_experimental_thresh_{token_position_cutoff}_high_r.json", "w") as f:
        json.dump(raw_evaluations, f, indent=4)
    with open(f"./results/scaled_evaluations_{aggregate_measure}_experimental_thresh_{token_position_cutoff}_high_r.json", "w") as f:
        json.dump(scaled_evaluations, f, indent=4)
        print("\nSkill Scores for BERT Score-R, optimal_f1_threshold, Avg Logit:")

    print("Raw Evaluations:")
    for language in raw_evaluations:
        print(f"\n{language}:")
        for model_name in raw_evaluations[language]:
            print(f"  {model_name}:")
            for prompting_method in raw_evaluations[language][model_name]:
                skill_score = raw_evaluations[language][model_name][prompting_method]["BERT Score-R"]["Avg Logit"]["optimal_f1_threshold"]["Skill score"]
                print(f"    {prompting_method}: {skill_score:.6f}")

    print("Scaled Evaluations:")
    for language in scaled_evaluations:
        print(f"\n{language}:")
        for model_name in scaled_evaluations[language]:
            print(f"  {model_name}:")
            for prompting_method in scaled_evaluations[language][model_name]:
                skill_score = scaled_evaluations[language][model_name][prompting_method]["BERT Score-R"]["Avg Logit"]["optimal_f1_threshold"]["Skill score"]
                print(f"    {prompting_method}: {skill_score:.6f}")
