import os
from matplotlib import pyplot as plt
# from torch.nn import functional as F
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import numpy as np

def calc_moe(num_samples, sample_prop):
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    if num_samples> 0:
        standard_error = np.sqrt((sample_prop * (1 - sample_prop)) / num_samples)
    else:
        standard_error = 0
    margin_of_error = z_score * standard_error
    return margin_of_error

def draw_reliability_graph(ECE, ACC, SS, Brier, samples_per_bin, unskilled_brier_score, bins, bin_accs, bin_confs, bin_sizes, save_path, save_name, raw=False):
    bins = bins[1:]
    index_x = [b - 0.05 for b in bins]

    fig = plt.figure(figsize=(8, 12)) # Create a figure with two subplots vertically
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])  # Define a grid with 2 rows and 1 column, with different heights

    # Sample Distribution
    ax0 = plt.subplot(gs[0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("% of Samples", fontsize=24)
    bin_sizes = bin_sizes * 100
    ax0.bar(index_x, bin_sizes, width=0.1, alpha=1, edgecolor='black')
    for i, j, k in zip(index_x, bin_sizes, samples_per_bin):
        ax0.text(i, j, "%.1f" % (j), ha="center", va="bottom", fontsize=18)
    ax0.tick_params(labelsize=20)

    # Reliability plot
    ax1 = plt.subplot(gs[1])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Model Confidence", fontsize=24)
    ax1.set_ylabel("Model Accuracy", fontsize=24)
    ax1.set_axisbelow(True)
    # Draw bars and identity line
    ax1.grid(color='gray')
    bars = ax1.bar(index_x, bin_accs, width=0.1, edgecolor="black", alpha=1,label='Output')
    for i, bar in enumerate(bars):
        moe = calc_moe(samples_per_bin[i], bin_accs[i])
        ax1.errorbar(bar.get_x() + bar.get_width() / 2, bin_accs[i], yerr=moe, fmt='o', color='black', linewidth=5)

    ax1.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    ax1.set_aspect('equal', adjustable='box')

    # Output legend
    textstr = 'ECE = {:.2f}'.format(ECE) +\
            '\n' + 'Brier Score = {:.2f}'.format(Brier) +\
            '\n' + 'Skill Score = {:.2f}'.format(SS) +\
            '\n' + 'Success Rate = {:.2f}'.format(ACC)
    # textstr = 'Brier Score = {:.2f}'.format(Brier) +\
    #         '\n' + 'Skill Score = {:.2f}'.format(SS) +\
    #         '\n' + 'Success Rate = {:.2f}'.format(ACC)
    props = dict(boxstyle='round', alpha=0.5, facecolor='white', edgecolor='black')
    # if raw:
    #     ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=25,
    #     verticalalignment='top', horizontalalignment='left', bbox=props)
    # else:
    #     ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=25,
    #     verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=25,
    # verticalalignment='top', horizontalalignment='left', bbox=props)
    ax1.tick_params(labelsize=20)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # plt.savefig(os.path.join(save_path, '{}-reliability.png'.format(save_name)), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, '{}-reliability.png'.format(save_name)), dpi=600, bbox_inches="tight", transparent=True)
    else:
        plt.show()

def perfect_reliability_graph():
    """Generate a reliability plot for a perfectly calibrated model."""
    
    # Create 10 confidence bins from 0.1 to 1.0
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Midpoint of each bin
    
    # For perfect calibration: accuracy = confidence for each bin
    bin_confs = bin_centers
    bin_accs = bin_centers  # Perfect calibration means accuracy equals confidence
    
    # Generate realistic sample distribution (more samples in middle confidence ranges)
    total_samples = 1000
    # Sample distribution that's higher in middle ranges (typical for real models)
    sample_distribution = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.18, 0.15, 0.12, 0.08, 0.05])
    samples_per_bin = (sample_distribution * total_samples).astype(int)
    bin_sizes = sample_distribution  # Proportion of total samples in each bin
    
    # Calculate metrics for perfect calibration
    ECE = 0.0  # Expected Calibration Error is 0 for perfect calibration
    
    # Overall accuracy (weighted average of bin accuracies)
    ACC = np.sum(bin_accs * sample_distribution)
    
    # Brier score for perfect calibration
    # For a perfectly calibrated model, Brier score = base_rate * (1 - base_rate)
    # where base_rate is the overall accuracy
    Brier = ACC * (1 - ACC)
    
    # Unskilled Brier score (for a model that always predicts the base rate)
    unskilled_brier_score = ACC * (1 - ACC)
    
    # Skill Score (Brier Skill Score) = 1 - (Brier / Brier_ref)
    # For perfect calibration, this equals 0 since Brier equals the reference
    SS = 0.0
    
    # Generate the plot
    save_path = "."  # Save in current directory
    save_name = "perfect_calibration"
    
    print("Generating perfect calibration reliability plot...")
    print(f"ECE: {ECE:.3f}")
    print(f"Accuracy: {ACC:.3f}")
    print(f"Brier Score: {Brier:.3f}")
    print(f"Skill Score: {SS:.3f}")
    
    draw_reliability_graph(
        ECE=ECE,
        ACC=ACC,
        SS=SS,
        Brier=Brier,
        samples_per_bin=samples_per_bin,
        unskilled_brier_score=unskilled_brier_score,
        bins=bins,
        bin_accs=bin_accs,
        bin_confs=bin_confs,
        bin_sizes=bin_sizes,
        save_path=save_path,
        save_name=save_name,
        raw=False
    )

def poorly_calibrated_reliability_graph():
    """Generate a reliability plot for a poorly calibrated model with no confidence-accuracy relationship."""
    
    # Create 10 confidence bins from 0.1 to 1.0
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Midpoint of each bin
    
    # For poor calibration: confidence varies but accuracy stays roughly constant
    bin_confs = bin_centers
    # Accuracy is roughly constant around 0.6, with some small random variations
    # This shows no relationship between confidence and accuracy
    base_accuracy = 0.6
    accuracy_noise = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, 0.01, -0.01, 0.02])
    bin_accs = np.clip(base_accuracy + accuracy_noise, 0.0, 1.0)  # Keep within [0,1]
    
    # Generate realistic sample distribution (more samples in higher confidence ranges for overconfident model)
    total_samples = 1000
    # Overconfident model: more samples in high confidence bins
    sample_distribution = np.array([0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.20, 0.17, 0.12])
    samples_per_bin = (sample_distribution * total_samples).astype(int)
    bin_sizes = sample_distribution  # Proportion of total samples in each bin
    
    # Calculate metrics for poor calibration
    # ECE: Expected Calibration Error = sum over bins of |accuracy - confidence| * proportion
    ECE = np.sum(np.abs(bin_accs - bin_confs) * sample_distribution)
    
    # Overall accuracy (weighted average of bin accuracies)
    ACC = np.sum(bin_accs * sample_distribution)
    
    # Brier score for poor calibration - higher than perfect calibration
    # For simplicity, we'll estimate it as worse than the perfect case
    perfect_brier = ACC * (1 - ACC)
    Brier = perfect_brier + 0.1  # Add penalty for poor calibration
    
    # Unskilled Brier score (for a model that always predicts the base rate)
    unskilled_brier_score = ACC * (1 - ACC)
    
    # Skill Score (Brier Skill Score) = 1 - (Brier / Brier_ref)
    # This will be negative since Brier > unskilled_brier_score
    SS = 1 - (Brier / unskilled_brier_score)
    
    # Generate the plot
    save_path = "."  # Save in current directory
    save_name = "poor_calibration"
    
    print("Generating poor calibration reliability plot...")
    print(f"ECE: {ECE:.3f}")
    print(f"Accuracy: {ACC:.3f}")
    print(f"Brier Score: {Brier:.3f}")
    print(f"Skill Score: {SS:.3f}")
    
    draw_reliability_graph(
        ECE=ECE,
        ACC=ACC,
        SS=SS,
        Brier=Brier,
        samples_per_bin=samples_per_bin,
        unskilled_brier_score=unskilled_brier_score,
        bins=bins,
        bin_accs=bin_accs,
        bin_confs=bin_confs,
        bin_sizes=bin_sizes,
        save_path=save_path,
        save_name=save_name,
        raw=False
    )

def generate_probability_similarity_plots():
    """Generate scatterplots showing relationship between model probabilities and similarity scores."""
    
    # Generate synthetic data for different correlation scenarios
    n_samples = 1000
    
    # Scenario 1: Strong positive correlation (well-calibrated model)
    np.random.seed(42)
    probs_strong = np.random.beta(2, 2, n_samples)  # Beta distribution for probabilities
    noise_strong = np.random.normal(0, 0.1, n_samples)
    similarity_strong = np.clip(0.3 + 0.6 * probs_strong + noise_strong, 0, 1)
    
    # Scenario 2: Weak/no correlation (poorly calibrated model)
    probs_weak = np.random.beta(3, 1.5, n_samples)  # More confident predictions
    similarity_weak = np.random.beta(2, 2, n_samples)  # Independent similarity scores
    
    # Scenario 3: Negative correlation (overconfident model)
    probs_negative = np.random.beta(1.5, 1, n_samples)  # High confidence
    noise_negative = np.random.normal(0, 0.15, n_samples)
    similarity_negative = np.clip(0.8 - 0.4 * probs_negative + noise_negative, 0, 1)
    
    scenarios = [
        (probs_strong, similarity_strong, "Strong Positive Correlation", "strong_correlation"),
        (probs_weak, similarity_weak, "Weak Correlation", "weak_correlation"),
        (probs_negative, similarity_negative, "Negative Correlation", "negative_correlation")
    ]
    
    for probs, similarity, title, filename in scenarios:
        # Calculate correlation metrics
        pearson_r, pearson_p = stats.pearsonr(probs, similarity)
        spearman_r, spearman_p = stats.spearmanr(probs, similarity)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatterplot
        scatter = ax.scatter(probs, similarity, alpha=0.6, s=20, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(probs, similarity, 1)
        p = np.poly1d(z)
        ax.plot(probs, p(probs), "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope={z[0]:.3f})')
        
        # Formatting
        ax.set_xlabel('Model Probability', fontsize=16)
        ax.set_ylabel('Similarity Score', fontsize=16)
        ax.set_title(f'{title}\nProbability vs. Similarity', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add correlation statistics text box
        stats_text = f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.3e})\n'
        stats_text += f'Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3e})\n'
        stats_text += f'N = {len(probs)} samples'
        
        # Determine text box position based on correlation
        if pearson_r > 0:
            bbox_x, bbox_y = 0.05, 0.95
            va, ha = 'top', 'left'
        else:
            bbox_x, bbox_y = 0.95, 0.95
            va, ha = 'top', 'right'
            
        props = dict(boxstyle='round', alpha=0.8, facecolor='white', edgecolor='black')
        ax.text(bbox_x, bbox_y, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment=va, horizontalalignment=ha, bbox=props)
        
        # Add legend
        ax.legend(loc='lower right' if pearson_r > 0 else 'lower left', fontsize=12)
        
        # Save plot
        plt.tight_layout()
        save_path = "."
        plt.savefig(os.path.join(save_path, f'{filename}_scatterplot.png'), 
                   dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        
        # Print results
        print(f"\n{title}:")
        print(f"  Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.3e}")
        print(f"  Spearman correlation: ρ = {spearman_r:.3f}, p = {spearman_p:.3e}")
        print(f"  Saved as: {filename}_scatterplot.png")

if __name__ == "__main__":
    perfect_reliability_graph()
    print("\n" + "="*50 + "\n")
    poorly_calibrated_reliability_graph()
    print("\n" + "="*50 + "\n")
    generate_probability_similarity_plots()