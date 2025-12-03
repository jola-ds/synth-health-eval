import pandas as pd
import os

def summarize_results(input_path, output_path):
    """
    Reads raw Master Loop results and calculates aggregate metrics (Mean/Std).
    """
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} (File not found)")
        return

    print(f"Processing: {input_path}...")
    df = pd.read_csv(input_path)
    
    # Group by Scenario and Model
    summary = df.groupby(['Scenario', 'Model']).agg({
        'Accuracy': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'AUC': ['mean', 'std']
    }).reset_index()
    
    # Flatten columns
    summary.columns = ['Scenario', 'Model', 'Acc_Mean', 'Acc_Std', 'F1_Mean', 'F1_Std', 'AUC_Mean', 'AUC_Std']
    
    # Sort for easier reading
    summary = summary.sort_values(['Model', 'Scenario'])
    
    print(f"Saving summary to: {output_path}")
    summary.to_csv(output_path, index=False)
    print(summary.to_string())
    print("-" * 50)

if __name__ == "__main__":
    # Define paths relative to project root or absolute
    base_dir = "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/4_data_analysis/results"
    
    # 1. Process Gaussian Copula
    summarize_results(
        input_path=os.path.join(base_dir, "results_master_loop_copula.csv"),
        output_path=os.path.join(base_dir, "results_summary_copula.csv")
    )
    
    # 2. Process CTGAN
    summarize_results(
        input_path=os.path.join(base_dir, "results_master_loop_ctgan.csv"),
        output_path=os.path.join(base_dir, "results_summary_ctgan.csv")
    )
