import os
import pandas as pd
import matplotlib.pyplot as plt

def load_election_data(path: str) -> pd.DataFrame:
    #load US election data into a pandas DataFrame
    
    #read correct delimiter
    df = pd.read_csv(path, sep=";", engine='python')

    #clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df

def plot_fraction_histogram(df: pd.DataFrame, column: str, save_path: str = None) -> None:
    #plot histogram of fraction of votes for a given column
    plt.figure(figsize=(10,6))
    plt.hist(df[column], bins=30, edgecolor='blue', color='blue', alpha=0.7)
    plt.title(f"Distribution of Fraction of Votes (US Election)") 
    plt.xlabel('Fraction of Votes')
    plt.ylabel('Number of States')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
