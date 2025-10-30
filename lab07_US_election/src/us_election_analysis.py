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

def get_candidate_df(df: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    #filter DataFrame for a specific candidate
    candidate_df = df[df['candidate'].str.lower() == candidate_name.strip().lower()].copy()
    return candidate_df

def ensure_dir(path: str) -> None:
    #ensure directory exists
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def plot_fraction_histogram(df: pd.DataFrame, 
                            column: str, 
                            save_path: str | None = None,
                            title_suffix: str = "") -> None:
    #plot histogram of fraction of votes for a given column
    plt.figure(figsize=(10,6))
    plt.hist(df[column].dropna(), 
             bins=30, 
             edgecolor='blue', 
             color='blue', 
             alpha=0.7
    )

    plt.title(f"Distribution of Candidate Vote Shares {title_suffix}".strip())
    plt.xlabel('Candidate vote share in county (fraction)')
    plt.ylabel('Count of county-candidate pairs')

    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_winner_share_hist(df: pd.DataFrame,
                           save_path: str | None = None,
                           title_suffix: str = "") -> None:
    """
    plot histogram of winner's vote share per county

    If df is the FULL dataset:
        - "winner" means whoever had the highest fraction_votes in that county.

    If df is ALREADY FILTERED to a single candidate:
        - We first find counties where that candidate is the winner,
          then plot that candidate's share in ONLY those counties.
    """
    #sort so top row per (state, county) is the winner
    sorted_df = df.sort_values(['state', 'county', 'fraction_votes'], ascending=[True, True, False])

    winners = (
        sorted_df
          .groupby(['state', 'county'], as_index=False)
          .first()[['state', 'county', 'candidate', 'party', 'fraction_votes']]
          .rename(columns={'fraction_votes': 'winner_share'})
    )

    plt.figure(figsize=(10,6))
    plt.hist(winners['winner_share'].dropna(), 
             bins=30, 
             edgecolor='green', 
             color='green', 
             alpha=0.7)
    plt.title(f"Distribution of County Winner Vote Shares {title_suffix}".strip())
    plt.xlabel("Winner's share in county (fraction)")
    plt.ylabel('Number of counties')

    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Winner share histogram saved to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_candidate_win_share_hist(candidate_df: pd.DataFrame,
                                  full_df: pd.DataFrame,
                                  save_path:str | None = None,
                                  candidate_name: str = "") -> None:
    """
    
    If df is the FULL dataset:
        - "winner" means whoever had the highest fraction_votes in that county.

    If df is ALREADY FILTERED to a single candidate:
        - We first find counties where that candidate is the winner,
          then plot that candidate's share in ONLY those counties.
    """
    #find per country who won overall (using full_df)
    full_sorted = full_df.sort_values([
        'state', 'county', 'fraction_votes'], 
        ascending=[True, True, False]
        )
    county_winner = (
        full_sorted
          .groupby(['state', 'county'], as_index=False)
          .first()[['state', 'county', 'candidate']]
          .rename(columns={'candidate': 'winner_candidate'})
    )

    #merg candidate row with county_winner -> keep only winning rows
    merged = candidate_df.merge(
        county_winner,
        on=['state', 'county'],
        how='inner'
    )
    wins = merged[
        merged["candidate"].str.lower()
        == merged["winner_candidate"].str.lower()
    ]

    if wins.empty:
        print(f"[INFO] {candidate_name} did not win in any county.")
        return
    
    plt.figure(figsize=(10,6))
    plt.hist(
        wins['fraction_votes'].dropna(),
        bins=30,
        edgecolor='purple',
        color='purple',
        alpha=0.7
    )
    plt.title(
        f"{candidate_name}: Vote Share in Counties Won"
    )
    plt.xlabel(f"{candidate_name}'s share in won counties (fraction)")
    plt.ylabel('Number of counties')

    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{candidate_name} win share histogram saved to {save_path}")
    else:
        plt.show()

    plt.close()

def print_sanity_checks(df: pd.DataFrame) -> None:
    #sanity check: within each (state, county), sum of all candidates
    #sum of fractions per county should ~1
    county_sum = (
        df.groupby(["state", "county"], as_index=False)["fraction_votes"].sum()
        .rename(columns={"fraction_votes": "sum_fraction"})
    )
    print("Sanity Check: Sum of fractions per county (should be ~1):")
    print(county_sum["sum_fraction"].describe())

def party_summary(df):
    #print basic descriptive stats for parties
    g = df.groupby("party")["fraction_votes"]
    out = (
        g.agg(mean_share="mean", median_share="median", n_rows="count")
        .sort_values("mean_share", ascending=False)
    )
    print("\n Average candidate vote shares by party:")
    print(out.round(3))


def candidate_summary(candidate_df: pd.DataFrame, 
                      candidate_name: str) -> None:
    #print basic descriptive stats for a candidate
    if candidate_df.empty:
        print(f"[WARN] No data available for candidate: {candidate_name}")
        return
    
    desc = candidate_df["fraction_votes"].describe()
    print(f"\n Candidate summary for {candidate_name}:")
    print(desc.to_string())
    print(
        f"Total counties with data for {candidate_name}: "
        f"{candidate_df.shape[0]}"
    )
