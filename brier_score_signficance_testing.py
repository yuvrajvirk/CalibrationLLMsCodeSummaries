import json
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import fdrcorrection

def brier_score_significance_testing(errors_1, errors_2):
    """
    Perform significance testing on the Brier scores using the paired t-test.
    """
    # Convert inputs to numpy arrays
    errors_1 = np.array(errors_1)
    errors_2 = np.array(errors_2)
    
    # Create a mask for valid pairs (non-NaN values in both arrays)
    valid_mask = ~(np.isnan(errors_1) | np.isnan(errors_2))
    
    # Apply the mask to both arrays
    errors_1_clean = np.power(errors_1[valid_mask], 2)
    errors_2_clean = np.power(errors_2[valid_mask], 2)
    
    if len(errors_1_clean) == 0:
        return "No valid data", "No valid data"
    
    if np.all(errors_1_clean == errors_1_clean[0]) and np.all(errors_2_clean == errors_2_clean[0]):
        return "Identical data", "Identical data"
    
    try:
        print("Error Lengths:", len(errors_1_clean), len(errors_2_clean))
        print("MSE 1:", np.mean(errors_1_clean), "MSE 2:", np.mean(errors_2_clean))
        t_statistic, p_value = stats.ttest_rel(errors_1_clean, errors_2_clean)
        return t_statistic, p_value, np.mean(errors_1_clean), np.mean(errors_2_clean)
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}", "No valid data", "No valid data"

def interpret_t_test_results(t_statistic, p_value):
    """
    Interpret the results of the t-test.
    """
    alpha = 0.05
    if p_value < alpha:
        return "The difference in MSE between the two methods is statistically significant."
    else:
        return "There is not enough evidence to conclude that the difference in MSE is statistically significant."

if __name__ == "__main__":
    cutoff_results = "./results/calibration_metrics/scaled_evaluations_optimal_token_cutoff_geometric_mean_with_cutoff.json"
    no_cutoff_results = "./results/calibration_metrics/scaled_evaluations_optimal_token_cutoff_no_geometric_mean_with_cutoff.json"
    with open(cutoff_results, "r") as f:
        cutoff_data = json.load(f)
    with open(no_cutoff_results, "r") as f:
        no_cutoff_data = json.load(f)
    p_values = []
    t_statistics = []
    skill_score_cutoff = []
    skill_score_no_cutoff = []
    brier_score_cutoff = []
    brier_score_no_cutoff = []
    for language in cutoff_data:
        for model in cutoff_data[language]:
            cutoff_errors = [entry["errors"] for entry in cutoff_data[language][model]["optimal"]]
            no_cutoff_errors = [entry["errors"] for entry in no_cutoff_data[language][model]["optimal"]]
            t_statistic, p_value, mse_1, mse_2 = brier_score_significance_testing(cutoff_errors, no_cutoff_errors)
            p_values.append(p_value)
            t_statistics.append(t_statistic)
            skill_score_cutoff.append(cutoff_data[language][model]["optimal"][0]["Skill score"])
            skill_score_no_cutoff.append(no_cutoff_data[language][model]["optimal"][0]["Skill score"])
            brier_score_cutoff.append(mse_1)
            brier_score_no_cutoff.append(mse_2)
            print(f"Language: {language}, Model: {model}")
            print(f"T-statistic: {t_statistic}, P-value: {p_value}")
        
    _, p_values_adjusted = fdrcorrection(p_values)
    
    print("\nResults after FDR correction:")
    for i, (language, model) in enumerate([(lang, mod) for lang in cutoff_data for mod in cutoff_data[lang]]):
        if p_values_adjusted[i] < 0.05:
            print(f"Language: {language}, Model: {model}")
            print(f"Adjusted P-value: {p_values_adjusted[i]}")
            print(f"T-statistic: {t_statistics[i]}")
            print(f"Skill Score Difference: {skill_score_cutoff[i] - skill_score_no_cutoff[i]}, {(skill_score_cutoff[i] - skill_score_no_cutoff[i])/skill_score_no_cutoff[i]*100}%")
            print(f"Brier Score Difference: {brier_score_cutoff[i] - brier_score_no_cutoff[i]}, {(brier_score_cutoff[i] - brier_score_no_cutoff[i])/brier_score_no_cutoff[i]*100}%")
            # print(f"Skill Score Cutoff: {skill_score_cutoff[i]}")
            # print(f"Skill Score No Cutoff: {skill_score_no_cutoff[i]}")
            # print(f"Brier Score Cutoff: {brier_score_cutoff[i]}")
            # print(f"Brier Score No Cutoff: {brier_score_no_cutoff[i]}")
            # print("Interpretation:", "Significant" if p_values_adjusted[i] < 0.05 else "Not significant")
            print()