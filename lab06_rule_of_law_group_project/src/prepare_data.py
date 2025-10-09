from pathlib import Path
import pandas as pd

base = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw")

rol_file = base / "rule_of_law.csv" 
dem_file = base / "democracy_index.csv"
gdp_file = base / "gdp_per_capita.csv"

rol_col = "Rule of Law index (central estimate)"
dem_col = "Liberal democracy index (central estimate)"
gdp_col = "GDP per capita, PPP (constant 2021 international $)"

targets = ["Germany", "West Germany", "East Germany", "Russia"]
#best way to handle three germany, remove the line from the blank 
#"Germany", have two labelled lines showing West and East during their time 

rol = pd.read_csv(rol_file)[["Entity", "Code", "Year", rol_col]]
dem = pd.read_csv(dem_file)[["Entity", "Code", "Year", dem_col]]
gdp = pd.read_csv(gdp_file)[["Entity", "Code", "Year", gdp_col]]

#keep only Russia + German variants
rol = rol[rol["Entity"].isin(targets)].copy()
dem = dem[dem["Entity"].isin(targets)].copy()
gdp = gdp[gdp["Entity"].isin(targets)].copy()

print("ROL rows:", len(rol), "DEM rows:", len(dem), "GDP rows:", len(gdp))
