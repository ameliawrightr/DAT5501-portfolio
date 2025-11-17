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
plt.title('World Population Over Time (1950-2023)')
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
years_smooth = np.linspace(df['Year'].min(), df['Year'].max(), 500)

for deg, poly in polynomials.items():
    plt.plot(years_smooth, poly(years_smooth) / 1e9, label=f'Degree {deg} fit')

plt.xlabel('Year')
plt.xticks(np.arange(df['Year'].min(), last_year + 5, 5))
plt.ylabel("World Population (Billions)")
plt.title("Polynomial Fits to World Population Data (1950-2023)")
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
plt.scatter(df['Year'], df['Population'] / 1e9, label = "Actual Data", s=20, color='black')

#forecast curves
for deg, poly in polynomials.items():
    plt.plot(years_with_future, poly(years_with_future) / 1e9, label=f'Degree {deg} forecast')


plt.xlabel('Year')
plt.ylabel("World Population (Billions)")
plt.title("Polynomial Forecasts of World Population Data (1950-2033)\nProjected to 2033")
plt.xticks(np.arange(df['Year'].min(), last_year + 15, 5))
#add line at 2023 (cutoff)
plt.axvline(x=last_year, color='grey', linestyle='--', label='Forecast Start (2023)')
plt.legend()
plt.grid(True)
plt.tight_layout()

#save to artifacts folder
fig.savefig(artifacts_dir / 'PLOT3_world_population_polynomial_forecasts.png', dpi=300)
plt.close(fig)


#chisquared per degree of freedom
chi2_dof_results = [] #list of (degree, chi2/dof)
chi2_bic_results = [] #list of (degree, chi2, BIC)

for deg, poly in polynomials.items():
    #model prediction on training data
    y_model = poly(x_train)
    #residuals
    residuals = y_train - y_model
    #chisquared assuming same error for every points (sigma = 1)
    chi2 = np.sum(residuals ** 2)

    #degrees of freedom = N - number_of_parameters
    N = len(x_train)
    p = deg + 1
    dof = N - p

    #calculate bayesian information criterion (BIC)
    bic = N * np.log(chi2 / N) + p * np.log(N)

    chi2_bic_results.append((deg, chi2, bic))
    chi2_per_dof = chi2 / dof

    chi2_dof_results.append((deg, chi2_per_dof / dof))
    chi2_bic_results.append((deg, chi2, bic))

    print(f"Degree {deg}: chi^2 = {chi2:.3e}, dof = {dof}, chi^2/dof = {chi2_per_dof:.3e}")
    print(f"Degree {deg}: BIC = {bic:.3e}")
    print()

#PLOT 4: chi2 per dof vs polynomial degree

degrees_list = [item[0] for item in chi2_dof_results]
chi2_dof_list = [item[1] for item in chi2_dof_results]

fig = plt.figure(figsize=(10, 6))
plt.plot(degrees_list, chi2_dof_list, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel(r'$\ chi^2$ per Degree of Freedom')
#line break between title and subtitle
plt.title(r'$\ chi^2$ per Degree of Freedom vs Polynomial Degree' + '\nWorld Population Data (1950-2023)')
plt.xticks(degrees_list)
plt.grid(True, linestyle=':')

plt.tight_layout()
fig.savefig(artifacts_dir / 'PLOT4_chi2_per_dof_vs_polynomial_degree.png', dpi=300)
plt.close(fig)

#PLOT 5: BIC vs polynomial degree
degrees_list_bic = [item[0] for item in chi2_bic_results]
bic_list = [item[2] for item in chi2_bic_results]

fig = plt.figure(figsize=(10, 6))
plt.plot(degrees_list_bic, bic_list, marker='o', color='orange')
plt.xlabel('Polynomial Degree')
plt.ylabel('BIC')
plt.title('BIC vs Polynomial Degree \nWorld Population Data (1950-2023)')
plt.xticks(degrees_list_bic)
plt.grid(True, linestyle=':')
plt.tight_layout()
fig.savefig(artifacts_dir / 'PLOT5_BIC_vs_polynomial_degree.png', dpi=300)
plt.close(fig)  
