from lab07_US_election.src.us_election_analysis import (
    load_election_data,
    plot_fraction_histogram,
    print_sanity_checks,    
)


#load csv from data folder
csv_path = "lab07_US_election/data/us_election.csv"
out_path = "lab07_US_election/artifacts/histogram_fraction_votes.png"

df = load_election_data(csv_path)

#inspect
print(df.head(8))
print(df.info())
print(df["fraction_votes"].describe())
print_sanity_checks(df)

#plot histogram
plot_fraction_histogram(df, column="fraction_votes", save_path=out_path)