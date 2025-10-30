import os
from lab07_US_election.src.us_election_analysis import (
    load_election_data,
    get_candidate_df,
    plot_fraction_histogram,
    plot_candidate_win_share_hist,
    print_sanity_checks,   
    plot_winner_share_hist,
    party_summary,
    candidate_summary,
)


#load csv from data folder
csv_path = "lab07_US_election/data/us_election.csv"
out_path = "lab07_US_election/artifacts"


def main():
    #load full dataset
    df = load_election_data(csv_path)

    #basic inspection
    print("[INFO] Head:")
    print(df.head(8))
    print("[INFO] Info:")
    print(df.info())
    print("[INFO] Fraction Votes Description:")
    print(df["fraction_votes"].describe())

    print_sanity_checks(df)
    party_summary(df)

    #ask user which candidate to analyze
    candidate_name = input(
        "Enter candidate name to analyze (e.g., 'JOE BIDEN'): "
    ).strip()

    #filter df for candidate
    cand_df = get_candidate_df(df, candidate_name)

    if cand_df.empty:
        print(f"[WARN] No data found for candidate '{candidate_name}'. Exiting.")
        return
    
    #print candidate summary
    candidate_summary(cand_df, candidate_name)

    #make cand specific artifact directory
    safe_name = candidate_name.lower().replace(" ", "_")
    cand_out_path = os.path.join(out_path, safe_name)
    os.makedirs(cand_out_path, exist_ok=True)

    #plot candidate fraction histogram
    plot_fraction_histogram(
        cand_df, 
        column="fraction_votes", 
        save_path=os.path.join(
            cand_out_path, 
            f"{safe_name}_vote_share_hist.png"
        ),
        title_suffix=f"for {candidate_name}"
    )

    #plot candidate distribution shares
    plot_candidate_win_share_hist(
        candidate_df=cand_df,
        full_df=df,
        save_path=os.path.join(
            cand_out_path,
            f"{safe_name}_winner_share_hist.png"
        ),
        candidate_name=candidate_name,
    )   

    #overall winner share histogram for ref
    #stored at root artifacts dir
    global_artifact_dir = out_path
    os.makedirs(global_artifact_dir, exist_ok=True)

    plot_fraction_histogram(
        df, 
        column="fraction_votes", 
        save_path=os.path.join(
            global_artifact_dir,
            "all_candidates_vote_share_hist.png"
        ),
    title_suffix="(All Candidates)"
    )
    
    plot_winner_share_hist(
        df, 
        save_path=os.path.join(
            global_artifact_dir,
            "all_candidates_winner_share_hist.png"
        ),
        title_suffix="(All Counties)"
    )


if __name__ == "__main__":
    main()