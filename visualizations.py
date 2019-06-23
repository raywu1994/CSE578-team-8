import pandas as pd

#read in both datasets, combine into one
training_data = pd.read_csv('adult.csv')
headers = training_data.columns.values.tolist()
test_data = pd.read_csv('adult.test.csv', names=headers)
all_data = pd.concat([training_data, test_data])
