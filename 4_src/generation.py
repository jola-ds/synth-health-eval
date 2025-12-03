import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class GeneratorWrapper:
    """
    Wrapper to unify SDV's GaussianCopula and CTGAN
    under a single interface for the Master Loop.
    """
    def __init__(self, model_type='copula'):
        self.model_type = model_type
        self.model = None
        self.metadata = None
        self.columns = None

    def fit(self, data):
        self.columns = data.columns
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)
        
        if self.model_type == 'copula':
            self.model = GaussianCopulaSynthesizer(self.metadata)
        elif self.model_type == 'ctgan':
            self.model = CTGANSynthesizer(self.metadata, epochs=300, verbose=True)
            
        self.model.fit(data)

    def sample(self, num_rows):
        return self.model.sample(num_rows=num_rows)

def train_generator(data, model_type='copula'):
    """
    Trains a generator (Copula or CTGAN) on the provided data.
    """
    gen = GeneratorWrapper(model_type)
    gen.fit(data)
    return gen

def generate_data(synthesizer, num_rows):
    """
    Generates synthetic data using the trained synthesizer.
    """
    return synthesizer.sample(num_rows=num_rows)

def calculate_dcr(real_data, synthetic_data):
    """
    Calculates the Distance to Closest Record (DCR) for each synthetic point.
    Returns the minimum DCR (privacy risk) and the mean DCR (utility/diversity).
    """
    # We need to scale data because features have different ranges (Age vs BP)
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_data)
    syn_scaled = scaler.transform(synthetic_data)
    
    # Find nearest neighbor in Real data for each Synthetic point
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(real_scaled)
    distances, indices = nbrs.kneighbors(syn_scaled)
    
    min_dcr = np.min(distances)
    mean_dcr = np.mean(distances)
    
    return min_dcr, mean_dcr

if __name__ == "__main__":
    # Sanity Check Script
    print("Loading data...")
    # Load the MICE-imputed data we just created
    df = pd.read_csv("c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/1_datasets/imputed_data.csv")
    
    print(f"Training CTGAN on {len(df)} rows...")
    model = train_generator(df, model_type='ctgan')
    
    print("Generating 1000 synthetic rows...")
    syn_data = generate_data(model, 1000)
    
    print("Calculating Privacy Metrics (DCR)...")
    min_dist, mean_dist = calculate_dcr(df, syn_data)
    
    print("-" * 30)
    print(f"Min DCR (Privacy Risk): {min_dist:.4f}")
    print(f"Mean DCR (Diversity):   {mean_dist:.4f}")
    print("-" * 30)
    
    if min_dist < 0.01:
        print("WARNING: Synthetic data is too close to real data (Potential Leakage!)")
    else:
        print("Privacy Check: PASSED (No exact matches found)")
        
    # Save a sample
    output_path = "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/1_datasets/synthetic_sample_ctgan.csv"
    syn_data.to_csv(output_path, index=False)
    print(f"Sample synthetic data saved to: {output_path}")
