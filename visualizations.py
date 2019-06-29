import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = load_data()
    
    generateVsEarningsStackedBarChart(df, 'Age')
    generateVsEarningsStackedBarChart(df, 'Education')
    generateVsEarningsStackedBarChart(df, 'Marital Status')
    generateVsEarningsStackedBarChart(df, 'Occupation', figSizeWidth = 25, figSizeLength = 10, xticksRotation = 25)
    generateVsEarningsStackedBarChart(df, 'Relationship')
    generateVsEarningsStackedBarChart(df, 'Race')
    generateVsEarningsStackedBarChart(df, 'Gender')
    generateVsEarningsStackedBarChart(df, 'From_USA')
    generateVsEarningsStackedBarChart(df, 'Hours Worked Per Week')
    plt.close('all') # to prevent memory warnings
    
    generateOverUnderPieCharts(df, 'Education')
    generateOverUnderPieCharts(df, 'Marital Status')
    generateOverUnderPieCharts(df, 'Occupation')
    generateOverUnderPieCharts(df, 'Relationship')
    generateOverUnderPieCharts(df, 'Race')
    generateOverUnderPieCharts(df, 'Gender')
    generateOverUnderPieCharts(df, 'Age')
    generateOverUnderPieCharts(df, 'Hours Worked Per Week')
    plt.close('all') # to prevent memory warnings
        
    generateGeneralPopulationPieCharts(df, 'Education')
    generateGeneralPopulationPieCharts(df, 'Martial_Status')
    generateGeneralPopulationPieCharts(df, 'Occupation')
    generateGeneralPopulationPieCharts(df, 'Relationship')
    generateGeneralPopulationPieCharts(df, 'Race')
    generateGeneralPopulationPieCharts(df, 'Gender')
    generateGeneralPopulationPieCharts(df, 'Age')
    generateGeneralPopulationPieCharts(df, 'Hours Worked Per Week')

def load_data():
    #read in both datasets, combine into one
    training_data = pd.read_csv('adult.csv')
    headers = training_data.columns.values.tolist()
    test_data = pd.read_csv('adult.test.csv', names=headers)
    all_data = pd.concat([training_data, test_data])

    #set some calculated fields and clean column names for presentation
    all_data['Below_50k'] = all_data['Salarys'].apply(lambda x: False if x == ' >50K' else True)
    all_data['From_USA'] = all_data['NTVCTRY'].apply(lambda x: True if x == ' United-States' else False)
    all_data['Age'] = pd.cut(all_data['Age'],bins=[0,19,29,39,49,59,69,120], labels=['17-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-90'])
    all_data['Hours Worked Per Week'] = pd.cut(all_data['HRSPERWK'], bins=[0,29,39,40,59,1000], labels=['Under 30', '30-39', '40', '41-59','Over 60'])
    all_data['Education'] = np.select([all_data['Education'].isin([' 9th',' 10th',' 11th',' 12th']),
                                            all_data['Education'] == ' HS-grad', all_data['Education'].isin([' Assoc-acdm',' Assoc-voc']),
                                            all_data['Education'] == ' Bachelors', all_data['Education'] == ' Some-college',
                                            all_data['Education'].isin([' Masters',' Doctorate'])],
                                          ['Some High School', 'HS Grad', 'Associates Degree', 'Bachelors Degree', 'Some College', 'Graduate Degree'], default='Other')

    # Fix this typo is the data file
    all_data['Marital Status'] = all_data['Martial_Status']
    
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

def generateOverUnderPieCharts(df, columnName, figSizeWidth = 10, figSizeLength = 5, barWidth = 0.5,  xticksRotation = None):
    df = df[[columnName, 'Below_50k']]
    above50k = df.loc[df['Below_50k'] == False].groupby(columnName).count().rename(columns={'Below_50k': 'CountOfAbove50k'})
    below50k = df.loc[df['Below_50k'] == True].groupby(columnName).count().rename(columns={'Below_50k': 'CountOfBelow50k'})

    values = list(above50k['CountOfAbove50k'])
    above50k.plot.pie(y='CountOfAbove50k', labels=values, figsize=(figSizeWidth, figSizeLength))
    plt.ylabel('')
    plt.title(columnName + ', Over 50K')
    plt.legend(labels=above50k.axes[0].values).set_bbox_to_anchor((1.2, .9, 0.1, 0.1))
    plt.savefig(f'vis/{columnName}_over50.png')
    
    values = list(below50k['CountOfBelow50k'])
    below50k.plot.pie(y='CountOfBelow50k', labels=values, figsize=(figSizeWidth, figSizeLength))
    plt.ylabel('')
    plt.title(columnName + ', Under 50K')
    plt.legend(labels=below50k.axes[0].values).set_bbox_to_anchor((1.2, .9, 0.1, 0.1))
    plt.savefig(f'vis/{columnName}_under50.png')

def generateGeneralPopulationPieCharts(df, columnName, figSizeWidth = 10, figSizeLength = 5, barWidth = 0.5,  xticksRotation = None):
    df = df[[columnName, 'Below_50k']]
    df = df.groupby(columnName).count().rename(columns={'Below_50k': 'Count'})
    values = list(df['Count'])
    df.plot.pie(y='Count', labels=values, figsize=(figSizeWidth, figSizeLength))
    plt.ylabel('')
    plt.title(columnName + ' Totals in Population')
    plt.legend(labels=df.axes[0].values).set_bbox_to_anchor((1.2, .9, 0.1, 0.1))
    plt.savefig(f'vis/{columnName}_total.png')
    
    
    
if __name__ == "__main__":
    main()
