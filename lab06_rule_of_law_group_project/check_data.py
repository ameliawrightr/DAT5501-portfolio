import pandas as pd

df = pd.read_csv("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw/rule_of_law.csv")
print(df.head())
print(df.columns)
print(df["Entity"].unique()[:10])
print(df.describe())