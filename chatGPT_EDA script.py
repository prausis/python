import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\Users\rausispa\Post CH AG\LS75 Annahme, Transport & Sortierung - LS75.1-01 Business Intelligence und Analytics\04 Work\02 BI Pakete\03 Analysen\Leitnummer_HAE_Juni22.csv')

# Print the first few rows of the data to inspect the structure
print(data.head())

# Print the summary statistics of the dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Check the data types of each column
print(data.dtypes)

# Explore the distribution of a continuous variable with a histogram
plt.hist(data['column_name'], bins=20)
plt.title('Histogram of Column Name')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Explore the relationship between two continuous variables with a scatter plot
plt.scatter(data['column_name_1'], data['column_name_2'])
plt.title('Scatter Plot of Column Name 1 vs. Column Name 2')
plt.xlabel('Column Name 1')
plt.ylabel('Column Name 2')
plt.show()

# Explore the relationship between a categorical variable and a continuous variable with a box plot
plt.boxplot(data['column_name'], by=data['categorical_variable'])
plt.title('Box Plot of Column Name by Categorical Variable')
plt.xlabel('Categorical Variable')
plt.ylabel('Column Name')
plt.show()

# Explore the relationship between two categorical variables with a bar plot
pd.crosstab(data['column_name_1'], data['column_name_2']).plot(kind='bar')
plt.title('Bar Plot of Column Name 1 by Column Name 2')
plt.xlabel('Column Name 1')
plt.ylabel('Frequency')
plt.show()
