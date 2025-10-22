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
    plt.hist(df[column].dropna(), bins=30, edgecolor='blue', color='blue', alpha=0.7)
    plt.title(f"Distribution of Candidate Vote Shares (All Counties)") 
    plt.xlabel('Candidate vote share in county (fraction)')
    plt.ylabel('Count of county-candidate pairs')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

def print_sanity_checks(df: pd.DataFrame) -> None:
    #sum of fractions per county should ~1
    county_sum = (
        df.groupby(["state", "county"], as_index=False)["fraction_votes"].sum()
        .rename(columns={"fraction_votes": "sum_fraction"})
    )
    print("Sanity Check: Sum of fractions per county (should be ~1):")
    print(county_sum["sum_fraction"].describe())