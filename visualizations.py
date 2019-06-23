import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = load_data()

    generateVsEarningsStackedBarChart(df, 'Age')

def load_data():
    #read in both datasets, combine into one
    training_data = pd.read_csv('adult.csv')
    headers = training_data.columns.values.tolist()
    test_data = pd.read_csv('adult.test.csv', names=headers)
    all_data = pd.concat([training_data, test_data])
    all_data['Below50k'] = all_data['Salarys'].apply(lambda x: False if x == ' >50K' else True)
    return all_data

def generateVsEarningsStackedBarChart(df, columnName, figSizeWidth = 20, figSizeLength = 5, barWidth = 0.5):
    df = df[[columnName, 'Below50k']]

    above50kByAge = df.loc[df['Below50k'] == False].groupby(columnName).count().rename(columns={'Below50k': 'CountOfAbove50k'})
    below50kByAge = df.loc[df['Below50k'] == True].groupby(columnName).count().rename(columns={'Below50k': 'CountOfBelow50k'})
    result = pd.merge(above50kByAge, below50kByAge,how='outer', on=columnName).sort_values(by=[columnName]).reset_index()

    bars1 = result['CountOfBelow50k'].values
    bars2 = result['CountOfAbove50k'].values
    columnValues = result[columnName].values

    r = np.arange(len(result.index))

    plt.figure(figsize=(20,5))
    p1 = plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
    p2 = plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)

    plt.legend((p1[0], p2[0]), ('<=50k', '>50k'), loc='upper right')
    plt.title(f'{columnName} vs Earnings', weight='bold')
    plt.ylabel('Count', weight='bold')
    plt.xticks(r, columnValues)
    plt.xlabel(columnName, weight='bold')

    plt.savefig(f'vis/{columnName}_vs_earnings.png')

if __name__ == "__main__":
    main()
