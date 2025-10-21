from lab07_US_election.src.us_election_analysis import (
    load_election_data,
    plot_fraction_histogram,    
)


#load csv from data folder
csv_path = "lab07_US_election/data/us_election.csv"
out_path = "lab07_US_election/artifacts/histogram_fraction_votes.png"

df = load_election_data(csv_path)

#inspect
print(df.head())
print(df.info())

#plot histogram
plot_fraction_histogram(df, column="fraction_votes", save_path=out_path)