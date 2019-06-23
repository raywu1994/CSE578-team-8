import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = load_data()
    df['Below50k'] = df['Salarys'].apply(lambda x: False if x == ' >50K' else True)

    columns_of_interest = ['Age','Below50k']
    coi_df = df[columns_of_interest]

    above50kByAge = coi_df.loc[coi_df['Below50k'] == False].groupby('Age').count().rename(columns={'Below50k': 'CountOfAbove50kByAge'})
    below50kByAge = coi_df.loc[coi_df['Below50k'] == True].groupby('Age').count().rename(columns={'Below50k': 'CountOfBelow50kByAge'})

    result = pd.merge(above50kByAge, below50kByAge,how='outer', on='Age').sort_values(by=['Age']).reset_index()
    print(result)

    bars1 = result['CountOfBelow50kByAge'].values
    bars2 = result['CountOfAbove50kByAge'].values
    ages = result['Age'].values

    r = np.arange(len(result.index))
    barWidth = 0.5

    plt.figure(figsize=(20,5))
    p1 = plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
    p2 = plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)

    plt.legend((p1[0], p2[0]), ('<=50k', '>50k'), loc='upper right')
    plt.title('Age vs Earnings')
    plt.ylabel('Count')
    plt.xticks(r, ages)
    plt.xlabel('Age')


    plt.savefig('vis/age_vs_earnings.png')
    # plt.show()

def load_data():
    #read in both datasets, combine into one
    training_data = pd.read_csv('adult.csv')
    headers = training_data.columns.values.tolist()
    test_data = pd.read_csv('adult.test.csv', names=headers)
    return pd.concat([training_data, test_data])

if __name__ == "__main__":
    main()
