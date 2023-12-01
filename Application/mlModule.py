import pandas as pd


# Read the CSV file
data = pd.read_csv('DataSet.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Check for missing values
missing_values = data.isnull().sum()

print(missing_values)

# Feature Engineering: Extract month and year from 'Date'
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

data.drop(['Unnamed'])

# Now that we have added new features, let's look at the first few rows of the updated dataframe
print(data.head())
