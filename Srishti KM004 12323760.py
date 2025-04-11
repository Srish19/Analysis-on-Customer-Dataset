import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\BSES\Downloads\customers-10000.csv")
print("Sample of 'Subscription Date' before conversion:")
print(df['Subscription Date'].head())
print("\nData type of 'Subscription Date':")
print(df['Subscription Date'].dtype)

# Convert 'Subscription Date' based on its format

if df['Subscription Date'].dtype == 'object':
    try:
        df['Subscription Date'] = df['Subscription Date'].astype(float)
        df['Subscription Date'] = pd.to_datetime(df['Subscription Date'] - 25569, origin='1970-01-01')
        print("\nConverted as Excel serial dates.")
    except ValueError:
        df['Subscription Date'] = pd.to_datetime(df['Subscription Date'])
        print("\nConverted as date strings.")
else:
    df['Subscription Date'] = pd.to_datetime(df['Subscription Date'] - 25569, origin='1970-01-01')
    print("\nConverted as numeric Excel serial dates.")

# Create numerical features
df['Subscription_Year'] = df['Subscription Date'].dt.year
df['Subscription_Month'] = df['Subscription Date'].dt.month

# 2. Find null values and show statistical data
print("\nNull Values in Each Column:")
print(df.isnull().sum())
print("\nPercentage of Null Values:")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\nStatistical Summary:")
print(df.describe(include='all'))

# 3. Handle categorical and numerical null values
categorical_cols = ['First Name', 'Last Name', 'Company', 'City', 'Country', 'Phone 1', 'Phone 2', 'Email', 'Website']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')  
df['Subscription_Year'] = df['Subscription_Year'].fillna(df['Subscription_Year'].median())  
df['Subscription_Month'] = df['Subscription_Month'].fillna(df['Subscription_Month'].median())
print("\nNull Values After Handling:")
print(df.isnull().sum())

# Scatterplot: Subscription_Year vs. Subscription_Month
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Subscription_Year', y='Subscription_Month', data=df, alpha=0.6)
plt.title('Scatterplot: Subscription Year vs. Month')
plt.xlabel('Subscription Year')
plt.ylabel('Subscription Month')
plt.tight_layout()
plt.show()

# Line Plot: Subscriptions per Month
monthly_counts = df.groupby('Subscription_Month').size().reset_index(name='Count')
plt.figure(figsize=(10, 5))
sns.lineplot(x='Subscription_Month', y='Count', data=monthly_counts, marker='o')
plt.title('Line Plot: Subscriptions by Month')
plt.xlabel('Month')
plt.ylabel('Number of Subscriptions')
plt.tight_layout()
plt.show()

# Box Plot: Subscription_Year
plt.figure(figsize=(8, 5))
sns.boxplot(x='Subscription_Year', data=df, color='lightblue')
plt.title('Box Plot of Subscription Years')
plt.xlabel('Subscription Year')
plt.tight_layout()
plt.show()

# Correlation Heatmap: Subscription_Year and Subscription_Month
corr_matrix = df[['Subscription_Year', 'Subscription_Month']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# IQR for Subscription_Year
Q1 = df['Subscription_Year'].quantile(0.25)
Q3 = df['Subscription_Year'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['Subscription_Year'] < lower_bound) | (df['Subscription_Year'] > upper_bound)]
print("\nIQR Outliers for Subscription Year:")
print("Number of outliers:", len(outliers_iqr))
if len(outliers_iqr) > 0:
    print(outliers_iqr[['Customer Id', 'First Name', 'Last Name', 'Subscription_Year']])

# Z-test for Subscription_Year
z_scores = stats.zscore(df['Subscription_Year'])
outliers_z = df[abs(z_scores) > 3]
print("\nZ-test Outliers for Subscription Year:")
print("Number of outliers:", len(outliers_z))
if len(outliers_z) > 0:
    print(outliers_z[['Customer Id', 'First Name', 'Last Name', 'Subscription_Year']])

# 6. Check skewness
skewness_year = df['Subscription_Year'].skew()
skewness_month = df['Subscription_Month'].skew()
print("\nSkewness Analysis:")
print("Subscription_Year Skewness:", round(skewness_year, 2))
print("Subscription_Month Skewness:", round(skewness_month, 2))
if abs(skewness_year) < 0.5:
    print("Subscription_Year is approximately symmetric.")
elif skewness_year > 0:
    print("Subscription_Year is right-skewed.")
else:
    print("Subscription_Year is left-skewed.")
if abs(skewness_month) < 0.5:
    print("Subscription_Month is approximately symmetric.")
elif skewness_month > 0:
    print("Subscription_Month is right-skewed.")
else:
    print("Subscription_Month is left-skewed.")

# 7. Perform t-test
top_countries = df['Country'].value_counts().head(2).index
country1 = top_countries[0]
country2 = top_countries[1]
group1 = df[df['Country'] == country1]['Subscription_Year']
group2 = df[df['Country'] == country2]['Subscription_Year']
t_stat, p_value = stats.ttest_ind(group1, group2)
print("\nT-test Between Countries:")
print("Comparing Subscription_Year for", country1, "and", country2)
print("T-statistic:", round(t_stat, 2))
print("P-value:", round(p_value, 4))
if p_value < 0.05:
    print("Significant difference in Subscription_Year between", country1, "and", country2, "(p < 0.05).")
else:
    print("No significant difference in Subscription_Year between", country1, "and", country2, "(p >= 0.05).")

#Perform chi- square test
contingency_table = pd.crosstab(df['Country'], df['Subscription_Year'])
print("\nContingency Table for Chi-Square Test:")
print(contingency_table)

chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-Square Test Results:")
print("Chi-Square Statistic:", round(chi2_stat, 2))
print("P-value:", round(p_val, 4))
print("Degrees of Freedom:", dof)

# Interpretation
if p_val < 0.05:
    print("Significant association between Country and Subscription_Year (p < 0.05).")
else:
    print("No significant association between Country and Subscription_Year (p >= 0.05).")

# 8. Variance Inflation Factor (VIF)
numerical_df = df[['Subscription_Year', 'Subscription_Month']].dropna()
X = sm.add_constant(numerical_df)  # Add constant for VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# 9. Shapiro-Wilk Test
shapiro_stat, shapiro_p = stats.shapiro(df['Subscription_Year'].dropna())
print("\nShapiro-Wilk Test for Subscription_Year:")
print("Statistic:", round(shapiro_stat, 2))
print("P-value:", round(shapiro_p, 4))
if shapiro_p > 0.05:
    print("Data appears normal (p > 0.05).")
else:
    print("Data does not appear normal (p <= 0.05).")


# Uniform Distribution
uniform_data = np.random.uniform(low=df['Subscription_Year'].min(), high=df['Subscription_Year'].max(), size=1000)
plt.figure(figsize=(8, 5))
sns.histplot(uniform_data, bins=20, kde=True, label='Uniform')
plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# Normal Distribution
normal_data = np.random.normal(loc=df['Subscription_Year'].mean(), scale=df['Subscription_Year'].std(), size=1000)
plt.figure(figsize=(8, 5))
sns.histplot(normal_data, bins=20, kde=True, label='Normal')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# Binomial Distribution
binomial_data = np.random.binomial(n=10, p=0.5, size=1000)
plt.figure(figsize=(8, 5))
sns.histplot(binomial_data, bins=10, kde=False, label='Binomial (n=10, p=0.5)')
plt.title('Binomial Distribution')
plt.xlabel('Number of Successes')
plt.legend()
plt.tight_layout()
plt.show()

# Poisson Distribution
poisson_data = np.random.poisson(lam=5, size=1000)
plt.figure(figsize=(8, 5))
sns.histplot(poisson_data, bins=15, kde=False, label='Poisson (λ=5)')
plt.title('Poisson Distribution')
plt.xlabel('Number of Events')
plt.legend()
plt.tight_layout()
plt.show()

# 11. Introduction to A/B Testing
print("\nIntroduction to A/B Testing:")
print("Simulating A/B test for subscription counts in top 2 countries:")
country1_count = len(group1)
country2_count = len(group2)
total = country1_count + country2_count
prop1 = country1_count / total
prop2 = country2_count / total
print(f"{top_countries[0]}: {country1_count} subscriptions ({prop1:.2f} proportion)")
print(f"{top_countries[1]}: {country2_count} subscriptions ({prop2:.2f} proportion)")
# Perform z-test for proportions
count = np.array([country1_count, country2_count])
nobs = np.array([total, total])
z_stat, ab_p_value = sm.stats.proportions_ztest(count, nobs)
print("A/B Test Z-statistic:", round(z_stat, 2))
print("P-value:", round(ab_p_value, 4))
if ab_p_value < 0.05:
    print("Significant difference in subscription proportions (p < 0.05).")
else:
    print("No significant difference in subscription proportions (p >= 0.05).")
