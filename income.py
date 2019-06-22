import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, cross_validation, svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score

### field Names for the Data provided 
fieldNames = ('age','workclass','fnlwgt','education', 'education-num',
              'marital-status','occupation','relationship','race','sex','capital-gain',
              'capital-loss','hours-per-week','native-country','income')
cat_coumns =('workclass','marital-status','occupation','relationship','race','sex','native-country','education')

### loading both training and test data in to two diferent dataframes
trn_data = pd.read_csv('adult.csv', names=fieldNames)
test_data = pd.read_csv('adult_test.csv', names=fieldNames, skiprows=1)
##trn_data.info()


## Classification of data into two categories , less than 50 and greater than 50

comb_df=pd.concat([trn_data,test_data],axis=0)

comb_df['income']=comb_df['income'].apply(lambda x: 1 if x==' >50K' else 0)

#### Removing NA and unknown nows from the train and Test data set
copy_df=comb_df
for column in comb_df.columns:
    if type(comb_df[column][0]) == str :
        comb_df[column] = comb_df[column].apply(lambda y: y.replace(" ",""))
comb_df.replace(' ?',np.nan,inplace=True) 

##Converting into columns values into columns with 0 if it exists and 1 if it doesnt exist
for column in cat_coumns:
    comb_df=pd.concat([comb_df,pd.get_dummies(comb_df[column],prefix=column,prefix_sep=':')],axis=1)
    comb_df.drop(column,axis=1,inplace=True)

comb_df.head()        

### Split data into Train and test 

data_x=np.array(comb_df.drop('income',1))
data_y=np.array(comb_df['income'])
data_x = preprocessing.scale(data_x)

train_x,test_x,train_y,test_y= cross_validation.train_test_split(data_x,data_y,test_size=0.3)

### Decision Tree to Predict  and acuracy score#########

cls_tr = DecisionTreeClassifier(max_depth=6)
cls_tr.fit(train_x,train_y)
tre_prdct = cls_tr.predict(test_x)
print(confusion_matrix(test_y,tre_prdct))
print(classification_report(test_y,tre_prdct))

A_score=accuracy_score(test_y,tre_prdct)
metrics.accuracy_score(test_y,tre_prdct)

print(A_score)

#############Visualizations to show the data#######################################
