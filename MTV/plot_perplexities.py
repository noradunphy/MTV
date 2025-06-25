import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_perplexities():
    """
    Plot mean perplexities for each dialogue act from eval_results_text_*.json files.
    Creates a bar plot with pairs of bars (clean and intervention) for each act.
    """
    # Find all eval results files
    json_files = glob.glob("eval_results_text_*.json")
    
    # Dictionaries to store results
    clean_ppls = {}
    clean_stds = {}
    intervention_ppls = {}
    intervention_stds = {}
    
    # Process each file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Extract dialogue act from filename using regex
        match = re.search(r'text_swda_([^_]+)', json_file)
        if match:
            act = match.group(1)
            
            # Store mean and std values
            clean_ppls[act] = data['summary']['clean_perplexity_mean']
            clean_stds[act] = data['summary']['clean_perplexity_std']
            intervention_ppls[act] = data['summary']['intervention_perplexity_mean']
            intervention_stds[act] = data['summary']['intervention_perplexity_std']
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Set up bar positions
    acts = sorted(clean_ppls.keys())  # Sort acts for consistent ordering
    x = np.arange(len(acts))
    width = 0.35
    
    # Create bars
    bars1 = plt.bar(x - width/2, [clean_ppls[act] for act in acts], width, 
            label='Clean', color='skyblue', alpha=0.7,
            yerr=[clean_stds[act] for act in acts], capsize=5)
    bars2 = plt.bar(x + width/2, [intervention_ppls[act] for act in acts], width, 
            label='Intervention', color='lightcoral', alpha=0.7,
            yerr=[intervention_stds[act] for act in acts], capsize=5)
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Customize plot
    plt.xlabel('Dialogue Acts', fontsize=14)
    plt.ylabel('Mean Perplexity (log scale)', fontsize=14)
    plt.title('Clean vs Intervention Perplexities by Dialogue Act', fontsize=16, pad=20)
    plt.xticks(x, acts, fontsize=12)
    plt.legend(fontsize=12)
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            # Position labels above the error bars
            y_pos = height + (height * 0.1)  # 10% above bar height
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.0f}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with high resolution
    plt.savefig('perplexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_perplexities() 