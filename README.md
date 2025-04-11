# Analysis-on-Customer-Dataset
This analysis explores a customer dataset (customer_data.csv) to uncover patterns and insights related to subscription behavior across different countries and time periods. The primary goal is to preprocess the data, perform statistical tests, visualize trends, and evaluate distribution properties, providing a foundation for understanding customer subscription dynamics. The analysis is implemented in Python using a suite of libraries (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels) and is designed to be both exploratory and educational, avoiding complex machine learning models.

Dataset
The dataset, customer_data.csv, contains customer records with fields such as Customer Id, First Name, Last Name, Company, Address, City, Country, Phone 1, Phone 2, Email, Subscription Date, and Website. A sample includes:

Nathaniel Stewart, Yeager Inc., Kathmandu, Nepal, Subscription Date: 43889 (2020-03-30).
Jane Doe, ABC Corp, Harare, Zimbabwe, Subscription Date: 43900 (2020-04-10). The Subscription Date is processed from Excel serial format or string dates into datetime objects, deriving Subscription_Year and Subscription_Month for numerical analysis.
Methodology
The analysis follows a structured Exploratory Data Analysis (EDA) pipeline:

Data Preprocessing:
Handles missing values: categorical fields (e.g., Country) filled with 'Unknown', numerical fields (e.g., Subscription_Year) with medians.
Converts Subscription Date to datetime for temporal analysis.
Statistical Summary:
Computes null counts and descriptive statistics (mean, std, min, max) to assess data quality and distribution.
Visualizations (Seaborn-styled):
Scatterplot: Plots Subscription_Year vs. Subscription_Month to explore temporal relationships.
Line Plot: Shows subscription counts by month to identify trends.
Box Plot: Visualizes Subscription_Year distribution and outliers.
Outlier Detection:
Uses Interquartile Range (IQR) and Z-test to identify anomalies in Subscription_Year.
Distribution Analysis:
Calculates skewness to evaluate symmetry of Subscription_Year.
Applies Shapiro-Wilk test to check normality.
Statistical Tests:
T-test: Compares mean Subscription_Year between the top two countries.
Chi-Square Test: Tests independence between Country and Subscription_Year using a contingency table.
Variance Inflation Factor (VIF): Assesses multicollinearity between Subscription_Year and Subscription_Month.
Probability Distributions:
Generates and visualizes theoretical distributions (Uniform, Normal, Binomial, Poisson) for comparison with empirical data.
A/B Testing:
Simulates an A/B test to compare subscription proportions between the top two countries using a Z-test.
Key Features
Libraries Used:
Pandas for data manipulation.
NumPy for numerical operations.
Matplotlib and Seaborn for visualizations.
SciPy for statistical tests (e.g., T-test, Chi-Square, Shapiro-Wilk).
Statsmodels for VIF and A/B testing (proportions Z-test).