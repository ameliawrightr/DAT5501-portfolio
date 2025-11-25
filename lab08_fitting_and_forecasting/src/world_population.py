import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

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

    #chi2 per degree of freedom
    chi2_per_dof = chi2 / dof

    #calculate bayesian information criterion (BIC)
    bic = N * np.log(chi2 / N) + p * np.log(N)

    chi2_dof_results.append((deg, chi2_per_dof))
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


#MODEL COMPARISON SUMMARY
#compare BIC vs chi2/dof -> pick best model

#best chi2/dof - lowest
best_deg_chi2, best_chi2_dof = min(chi2_dof_results, key=lambda x: x[1])

#best BIC - lowest
best_deg_bic, _, best_bic = min(chi2_bic_results, key=lambda x: x[2])

print("MODEL COMPARISON SUMMARY")
print(f"Minimum chi^2/dof is for degree {best_deg_chi2}: {best_chi2_dof:.3e}")
print(f"Minimum BIC is for degree {best_deg_bic}: {best_bic:.3e}")

if best_deg_chi2 == best_deg_bic:
    print(f"Both criteria agree on best model: degree {best_deg_bic}")
else:
    print(f"chi^2/dof prefers degree {best_deg_chi2}, "
          f"while BIC prefers degree {best_deg_bic} "
          f"(BIC penalises extra parameters more strongly).")
    
print(f"Using usual information criteria guidelines, the 'best' model is "
      f"the one with the lowest BIC: degree {best_deg_bic}.")


#PARAMETER VALUES, UNCERTAINTIES, ALTERNATIVE MODEL
#best polynomal model: parameters & covar

#refit best polynomial (by BIC) ask for covariance
#design matrix for poly of degree best_deg_bic
X = np.vander(x_train, best_deg_bic + 1, increasing=False) #shape (N, p)

#least squares fit: y ≈ X @ coeffs
best_poly_coeffs, residuals, rank, s = np.linalg.lstsq(X, y_train, rcond=None)

best_poly = np.poly1d(best_poly_coeffs)

N = len(x_train)
p = best_deg_bic + 1
dof_cov = N - p

#estimate of variance of residuals
if residuals.size > 0 and dof_cov > 0:
    sigma2 = residuals[0] / dof_cov
else:
    #fallback if residauls array is empty
    res = y_train - X @ best_poly_coeffs
    sigma2 = np.sum(res ** 2) / max(dof_cov, 1)

#covariance matrix of parameters
best_poly_cov = sigma2 * np.linalg.inv(X.T @ X)


print("Best polynomial model (by BIC)")
print(f"Degree: {best_deg_bic}")
print("Coefficients (highest power first):")
for i, c in enumerate(best_poly_coeffs):
    power = best_deg_bic - i
    print(f" a_{power} = {c:.6e}")

print(f"\nCovariance matrix of polynomial parameters:")
print(best_poly_cov)

#1sigma uncertainties from sqrt of diagonal of covariance matrix
poly_param_errors = np.sqrt(np.diag(best_poly_cov))

print("\n1-sigma uncertainties of polynomial parameters:")
for i, (c, err) in enumerate(zip(best_poly_coeffs, poly_param_errors)):
    power = best_deg_bic - i
    rel = abs(err / c) if c != 0 else np.nan
    print(f" a_{power}: ±{err:.3e} (relative: {rel:.3e})")


#alternative model: exponential growth y = A * exp(k * (Year - x0))
#shift year to keep exponent numerically stable
x0 = x_train.mean()

def exp_model(x, A, k):
    return A * np.exp(k * (x - x0))

#rough intial guess
p0 = [y_train[0], 0.02]

popt_exp, pcov_exp = curve_fit(
    exp_model, x_train, y_train, p0=p0, maxfev=10000
)
A_exp, k_exp = popt_exp
err_exp = np.sqrt(np.diag(pcov_exp))

print("\nEXPONENTIAL MODEL FIT")
print(f"Model: y = A * exp(k * (Year - {x0:.1f}))")
print(f"Fitted parameters:")
print(f" A = {A_exp:.6e} ± {err_exp[0]:.3e} (relative: {abs(err_exp[0] / A_exp):.3e})")
print(f" k = {k_exp:.6e} ± {err_exp[1]:.3e} (relative: {abs(err_exp[1] / k_exp):.3e})")
print("\nCovariance matrix for (A, k):")
print(pcov_exp)

#goodness of fit for exponential model
y_exp_model = exp_model(x_train, *popt_exp)
residuals_exp = y_train - y_exp_model
chi2_exp = np.sum(residuals_exp ** 2)
N = len(x_train)
p_exp = 2 #A and k
dof_exp = N - p_exp
chi2_dof_exp = chi2_exp / dof_exp
bic_exp = N * np.log(chi2_exp / N) + p_exp * np.log(N)

print(f"\nExponential model goodness of fit:")
print(f" chi^2 = {chi2_exp:.3e}, dof = {dof_exp}, chi^2/dof = {chi2_dof_exp:.3e}")
print(f" BIC = {bic_exp:.3e}")

#compare best polynomial vs exponential model
print("\nCOMPARISON OF BEST POLYNOMIAL VS EXPONENTIAL MODEL")
print(f"Best polynomial (degree {best_deg_bic}) BIC = {best_bic:.3e}")
print(f"Exponential model BIC = {bic_exp:.3e}")

if bic_exp < best_bic:
    print("Exponential model preferred by BIC over best polynomial model.")
else:
    print(f"Polynomial model still preferred by (lower) BIC over exponential model."
          f"Exponential model does not improve the fit.")
    
#best poly vs exp on historical data
fig = plt.figure(figsize=(10, 6))

#actual history
plt.scatter(df['Year'], df['Population'] / 1e9, 
            label = "Actual data", s=20, color='black')

#reuse years_smooth for plotting
plt.plot(years_smooth, best_poly(years_smooth) / 1e9,
         label=f'Best Polynomial (deg {best_deg_bic})', color='blue')
plt.plot(years_smooth, exp_model(years_smooth, *popt_exp) / 1e9,
         label='Exponential Model', color='red', linestyle='--')

plt.xlabel('Year')
plt.ylabel("World Population (Billions)")
plt.title("Best Polynomial vs Exponential Fit\nWorld Population Data (1950-2023)")
plt.xticks(np.arange(df['Year'].min(), last_year + 5, 5))
plt.grid(True)
plt.legend()
plt.tight_layout()

fig.savefig(artifacts_dir / 'PLOT6_best_polynomial_vs_exponential_fit.png', dpi=300)
plt.close(fig)