import pandas as pd

from sklearn import tree
from sklearn import metrics
from sklearn import linear_model

tree_max_depth = None
# visualizing larger trees is expensive
tree_vis_max_depth = 6

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income',
    ]

def main():
    train_x, train_y = load_data('adult.data')
    test_x, test_y = load_data('adult.test')

    clf = train(train_x, train_y, max_depth=tree_max_depth)
    pred = clf.predict(test_x)

    report(pred, test_y)
    visualizations(clf, test_x)

    # # test set extracted from training set
    # from sklearn.model_selection import train_test_split
    # train_x, train_y = load_data('adult.data')
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,
    #     test_size=0.3)
    # clf = train(train_x, train_y)
    # predict(clf, test_x, test_y)

def train(train_x, train_y, max_depth=None):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    return clf.fit(train_x, train_y)

def visualizations(clf, test_x):
    # decision tree
    tree.export_graphviz(clf, out_file='tree.dot',
        max_depth=tree_vis_max_depth,
        feature_names=test_x.columns,
        class_names=['>50k', '<=50k'],
        filled=True, rounded=True)
    print('decision tree saved to: tree.dot')
    print()

def report(pred, test_y):
    print('confusion_matrix:')
    print(metrics.confusion_matrix(test_y, pred))
    print()

    print('classification_report:')
    print(metrics.classification_report(test_y, pred))

    print('accuracy_score:')
    print(metrics.accuracy_score(test_y, pred))
    print()

def load_data(path):
    df = pd.read_csv('adult.data', header=None, names=column_names,
        skipinitialspace=True)
    # infer types of columns
    df = df.infer_objects()
    # set income column to boolean values
    df['income'] = df['income'].map(lambda v: v=='>50K')

    # one-hot encoding of categorical values
    # get_dummies auto selects columns of type 'object' or 'category'
    df = pd.get_dummies(df)
    df.reset_index()

    # split features (data_x), samples (data_y)
    data_x = df.drop('income', axis=1)
    data_y = df['income']

    return data_x, data_y

if __name__ == "__main__":
    main()
