import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

artifacts_dir = Path('/Users/amelia/DAT5501-portfolio/lab08_fitting_and_forecasting/artifacts')
artifacts_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('/Users/amelia/DAT5501-portfolio/lab08_fitting_and_forecasting/data/worldpopulation.csv')
df = df[df["Entity"] == "World"].copy()
df = df[['Year', 'all years']].rename(columns={'all years': 'Population'})
df = df.sort_values("Year").reset_index(drop=True)

print(df.head())
print(df.tail())


#PLOT 1: check data set plots - whole world series
fig = plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Population'] / 1e9, marker='o')
plt.title('World Population Over Time')
plt.xlabel('Year')
plt.ylabel('Population (Billions)')
plt.tight_layout()
plt.grid(True)

fig.savefig(artifacts_dir / 'PLOT1_world_population_over_time_full_series.png', dpi=300)
plt.close(fig)  


#split data into training and testing sets
last_year = df['Year'].max()
cutoff_year = last_year - 10

train = df[df['Year'] <= cutoff_year]
test = df[df['Year'] > cutoff_year]

print("Last year in dataset:", last_year)
print("Cutoff year for training/testing split:", cutoff_year)
print("Training set years:", train['Year'].min(), "to", train['Year'].max())
print("Testing set years:", test['Year'].min(), "to", test['Year'].max())

#convert to NumPy arrays for fitting
x_train = train['Year'].values
y_train = train['Population'].values

x_test = test['Year'].values
y_test = test['Population'].values


#fit polynomials
degrees = range(1,10)
polynomials = {}

for deg in degrees:
    coeffs = np.polyfit(x_train, y_train, deg)
    poly = np.poly1d(coeffs)
    polynomials[deg] = poly
    print(f"Fitted polynomial of degree {deg}:")
    print(poly)
    print()

#forecast 10 years into future
future_years = np.arange(last_year + 1, last_year + 11)
print("Future years:", future_years)

#evaluate polynomials on test set and future years
for deg, poly in polynomials.items():
    y_test_pred = poly(x_test)
    y_future_pred = poly(future_years)

    mse = np.mean((y_test_pred - y_test) ** 2)
    print(f"Degree {deg} polynomial: MSE on last 10 years {mse:.3e}")


#PLOT 2: fits on historical data
fig = plt.figure(figsize=(10, 6))

#actual data
plt.scatter(df['Year'], df['Population'] / 1e9, label = "Actual data", s=20, color='black')

#smooth line of years within historical range for drawing curves
years_smooth = np.linspace(df['Year'].min(), df['Year'].max() + 10, 500)

for deg, poly in polynomials.items():
    plt.plot(years_smooth, poly(years_smooth) / 1e9, label=f'Degree {deg} fit')

plt.xlabel('Year')
plt.ylabel("World Population (Billions)")
plt.title("Polynomial Fits to World Population Data")
plt.legend()
plt.grid(True)
plt.tight_layout()

#save to artifacts folder
fig.savefig(artifacts_dir / 'PLOT2_world_population_polynomial_fits.png', dpi=300)
plt.close(fig)

#PLOT 3: forecasts for future years
years_with_future = np.concatenate([df['Year'].values, future_years])

fig = plt.figure(figsize=(10, 6))

#actual history
plt.scatter(df['Year'], df['Population'] / 1e9, label = "Actual data", s=20, color='black')

#forecast curvs
for deg, poly in polynomials.items():
    plt.plot(years_with_future, poly(years_with_future) / 1e9, label=f'Degree {deg} forecast')

plt.xlabel('Year')
plt.ylabel("World Population (Billions)")
plt.title("Polynomial Forecasts of World Population Data")
plt.legend()
plt.grid(True)
plt.tight_layout()

#save to artifacts folder
fig.savefig(artifacts_dir / 'PLOT3_world_population_polynomial_forecasts.png', dpi=300)
plt.close(fig)