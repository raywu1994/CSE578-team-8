import matplotlib.pyplot as plt
import pandas as pd
import csv

# Create a key in a dictionary with the first value if that key does not exist
def addToKey(dictionary, key, value):
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value

    return dictionary

file = open('adult.csv', 'r')

# field names from data
fieldNames = ('age','workclass','fnlwgt','education', 'education-num',
              'marital-status','occupation','relationship','race','sex','capital-gain',
              'capital-loss','hours-per-week','native-country','income')

# For initial testing, cut list of fields to only categorical data
dimFieldNames = ('workclass', 'education', 'education-num',
              'marital-status','occupation','relationship','race','sex','native-country')

# make 2 dictionaries of dictionaries to divide each category into under 50K and over 50K
under50 = {
    'workclass': {},
    'education': {},
    'education-num': {},
    'marital-status': {},
    'occupation': {},
    'relationship': {},
    'race': {},
    'sex': {},
    'native-country': {}
    }
over50 = {
    'workclass': {},
    'education': {},
    'education-num': {},
    'marital-status': {},
    'occupation': {},
    'relationship': {},
    'race': {},
    'sex': {},
    'native-country': {}
    }


reader = csv.DictReader(file, fieldNames)

# For each field name, add up all people that fit in that each category for that field for each income level
for field in dimFieldNames:
    print('Procesing ' + field)
    for row in reader:
        if row['income'].strip() == '>50K':
            over50[field] = addToKey(over50[field], row[field].strip(), int(row['fnlwgt']))
        else:
            under50[field] = addToKey(under50[field], row[field].strip(), int(row['fnlwgt']))
    # Reset iterator
    file.seek(0)

# Display results
for field in dimFieldNames:
    print(field)
    print("Over 50: " + over50[field])
    print("Under 50: " + under50[field])

