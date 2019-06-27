import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = load_data()

    generateVsEarningsStackedBarChart(df, 'Age')
    generateVsEarningsStackedBarChart(df, 'Education')
    generateVsEarningsStackedBarChart(df, 'Martial_Status')
    generateVsEarningsStackedBarChart(df, 'Occupation', figSizeWidth = 25, figSizeLength = 10, xticksRotation = 25)
    generateVsEarningsStackedBarChart(df, 'Relationship')
    generateVsEarningsStackedBarChart(df, 'Race')
    generateVsEarningsStackedBarChart(df, 'Gender')
    generateVsEarningsStackedBarChart(df, 'From_USA')
    generateVsEarningsStackedBarChart(df, 'Age_Bin')
    generateVsEarningsStackedBarChart(df, 'Hours_Per_Week_Bin')

def load_data():
    #read in both datasets, combine into one
    training_data = pd.read_csv('adult.csv')
    headers = training_data.columns.values.tolist()
    test_data = pd.read_csv('adult.test.csv', names=headers)
    all_data = pd.concat([training_data, test_data])

    #set some calculated fields
    all_data['Below_50k'] = all_data['Salarys'].apply(lambda x: False if x == ' >50K' else True)
    all_data['From_USA'] = all_data['NTVCTRY'].apply(lambda x: True if x == ' United-States' else False)
    all_data['Age_Bin'] = pd.cut(all_data['Age'],bins=[0,19,29,39,49,59,69,120], labels=['17-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-90'])
    all_data['Hours_Per_Week_Bin'] = pd.cut(all_data['HRSPERWK'], bins=[0,29,39,40,100], labels=['Under 30', '30-39', '40', 'Over 40'])
    return all_data

def generateVsEarningsStackedBarChart(df, columnName, figSizeWidth = 20, figSizeLength = 5, barWidth = 0.5,  xticksRotation = None):
    df = df[[columnName, 'Below_50k']]

    above50kByAge = df.loc[df['Below_50k'] == False].groupby(columnName).count().rename(columns={'Below_50k': 'CountOfAbove50k'})
    below50kByAge = df.loc[df['Below_50k'] == True].groupby(columnName).count().rename(columns={'Below_50k': 'CountOfBelow50k'})
    result = pd.merge(above50kByAge, below50kByAge,how='outer', on=columnName)
    result = result.sort_values(by=[columnName]).reset_index()

    bars1 = result['CountOfBelow50k'].values
    bars2 = result['CountOfAbove50k'].values
    columnValues = result[columnName].values

    r = np.arange(len(result.index))

    plt.figure(figsize=(figSizeWidth, figSizeLength))
    p1 = plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
    p2 = plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)

    plt.legend((p1[0], p2[0]), ('<=50k', '>50k'), loc='upper right')
    plt.title(f'{columnName} vs Earnings', weight='bold')
    plt.ylabel('Count', weight='bold')
    plt.xticks(r, columnValues, rotation = xticksRotation)
    plt.xlabel(columnName, weight='bold')

    plt.savefig(f'vis/{columnName}_vs_earnings.png')

if __name__ == "__main__":
    main()
