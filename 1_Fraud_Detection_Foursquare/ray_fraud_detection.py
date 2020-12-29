# -*- coding: utf-8 -*-
"""
The code below is for building a classifier to detect fraud transactions based on the data provided.

@author: raysh
"""
"Import libraries"
# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
"import tensorflow as tf"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from datetime import datetime

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score


pd.set_option('display.max_columns', 100)
# read in the transaction dataset and the labels dataset
df_tran = pd.read_csv('transactions_obf.csv')
df_label=pd.read_csv('labels_obf.csv')
#left join this two dataset to create modeling dataset
df = pd.merge(df_tran,
                 df_label,
                 on='eventId',
                 how='left')
df.head()
df.describe()
#create the response column
df['fraud_ind']= np.where(pd.isnull(df['reportedTime']), 0, 1)

df.head

print("Credit Card Fraud Detection data -  rows:",df.shape[0]," columns:",df.shape[1])
#get statistics summary on numerical variables
df.describe()
#check missing value
df.isnull().sum()
#all missing value from merchantZip and reportedTime. Either level set the missing value as a group, or delete the rows. But the % is pretty high. will just group them into one group.
#fill missing value with 'NA'
df['merchantZip'].fillna('NA', inplace=True)
df['reportedTime'].fillna(0,inplace=True)
df['transactionTime']=pd.to_datetime(df["transactionTime"], format="%Y-%m-%d")
df['reportedTime']=pd.to_datetime(df["reportedTime"],format="%Y-%m-%d")

#create a new column indicating the utilization of the available credit line of the transaction
df['utilization_rt']=df.apply(lambda row: row.transactionAmount/row.availableCash, axis = 1) 

df['day_diff'] = (df['reportedTime'] - df['transactionTime']) / pd.Timedelta(days=1)
df.head
df.columns
#basic calculation of fraud rate
print('No Frauds', round(df['fraud_ind'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['fraud_ind'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

#imbalanced distribution for fraud_ind, need to do treatment for modeling

#bar chart of the fraud_ind 
colors = ["#0201AB", "#DF0101"]

sns.countplot('fraud_ind', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

#convert the transactin time into date

# check the distribution of some key attributes 
var = ['transactionAmount','availableCash', 'utilization_rt','day_diff']
i = 0
t0 = df.loc[df['fraud_ind'] == 0]
t1 = df.loc[df['fraud_ind'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(16,16))

for feature in var:
    i += 1
    plt.subplot(2,2,i)
    sns.kdeplot(t0[feature], bw=0.5,label="fraud_ind = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="fraud_ind = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


sns.kdeplot(t1['day_diff'], bw=0.5,label="fraud_ind = 1")
#deal with imbalance data on fraud 
#i will introduce SMOTE with cross validation setup to boost the performance of the classifier
# start with xgboost for prediction, to handle the categorical variables

df.columns
#initial setup of the data and EDA
#1. drop transactionTime as it is only sequencing, drop zipcode as it has missing value, drop report time as it is not relevant in prediction. drop the diff as reason stated before.
df = df.drop(['transactionTime','eventId','accountNumber','merchantId','merchantZip','reportedTime','day_diff'],axis=1)

#2. check correlation to see if there are obvious correlated varaibles
fig, ax = plt.subplots(figsize=(7,7))  
sns.heatmap(df.corr(),annot_kws={"size":4})
# will drop utilization_rt since it is perfectly correlated with transactionamount,also from previous charts there seems no big difference between fraud vs non fraud in terms of utilization rate.
df=df.drop(['utilization_rt'],axis=1)

df = df.sample(frac=1)
fraud_df = df.loc[df['fraud_ind'] == 1]
fraud_df.describe
non_fraud_df = df.loc[df['fraud_ind'] == 0][:875]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
new_df.describe()

#3. define X and Y
X=new_df.loc[:, new_df.columns != 'fraud_ind']
X['mcc']=X['mcc'].astype('category',copy=False)
X['merchantCountry']=X['merchantCountry'].astype('category',copy=False)
X['posEntryMode']=X['posEntryMode'].astype('category',copy=False)
X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

y= new_df.loc[:,new_df.columns == 'fraud_ind']

# 4. split into training and testing set
# Split into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#basic stats of traning and testing set
#Stats of training data 
print('---------Training data statistics-----------')
normal_trans_perc=sum( y_train['fraud_ind']==0)/(sum( y_train['fraud_ind']==0)+sum( y_train['fraud_ind']==1))
fraud_trans_perc=1-normal_trans_perc
print('Total number of records : {} '.format(len(y_train)))
print('Total number of normal transactions : {}'.format(sum(y_train['fraud_ind']==0)))
print('Total number of  fraudulent transactions : {}'.format(sum(y_train['fraud_ind']==1)))
print('Percent of normal transactions is : {:.4f}%,  fraudulent transactions is : {:.4f}%'.format(normal_trans_perc*100,fraud_trans_perc*100))

#Stats of testing data 
print('---------Testing data statistics-----------')
normal_trans_perc=sum( y_test['fraud_ind']==0)/(sum( y_test['fraud_ind']==0)+sum( y_test['fraud_ind']==1))
fraud_trans_perc=1-normal_trans_perc
print('Total number of records : {} '.format(len(y_test)))
print('Total number of normal transactions : {}'.format(sum(y_test['fraud_ind']==0)))
print('Total number of  fraudulent transactions : {}'.format(sum(y_test['fraud_ind']==1)))
print('Percent of normal transactions is : {:.4f}%,  fraudulent transactions is : {:.4f}%'.format(normal_trans_perc*100,fraud_trans_perc*100))

# for simplicity, not using Kfold CV for modeling training. 
# compare couple options

X_train.to_csv('x_train.csv')
y_train.to_csv('y_train.csv')
X_test.to_csv('x_test.csv')
y_test.to_csv('y_test.csv')

def train_predict(learner, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on       
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}

    learner.fit(X_train, y_train)
 
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    
    predictions_test_prob = learner.predict_proba(X_test)[:,1]
    predictions_train_prob = learner.predict_proba(X_train)[:,1]
    
    results['acc_train'] = accuracy_score(y_train, predictions_train)      
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['rec_train'] = recall_score(y_train, predictions_train)      
    results['rec_test'] = recall_score(y_test, predictions_test)
    
    results['prec_train'] = precision_score(y_train, predictions_train)      
    results['prec_test'] = precision_score(y_test, predictions_test)
    
    
    results['f1_train'] = f1_score(y_train, predictions_train)
    results['f1_test'] = f1_score(y_test, predictions_test)
    
    results['auc_train'] = average_precision_score(y_train, predictions_train_prob,average='weighted')
    results['auc_test'] = average_precision_score(y_test, predictions_test_prob,average='weighted')
    
    
       
    # Success
    print('success')
        
    # Return the results
    return results


# try the following options
# Initialize and train the models
clf_lr = LogisticRegression(random_state=0)
clf_rf = RandomForestClassifier(random_state=0)
clf_xg = XGBClassifier()

# Collect results on the learners
results = {}
for clf in [clf_lr,clf_rf,clf_xg]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    results[clf_name] = train_predict(clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())

lr_res=pd.DataFrame(results['LogisticRegression'],index=['LR'])
rf_res=pd.DataFrame(results['RandomForestClassifier'],index=['RF'])
xg_res=pd.DataFrame(results['XGBClassifier'],index=['XG'])
all_res= pd.concat([lr_res,rf_res,xg_res])

all_res[['acc_train','acc_test','rec_train','rec_test',\
         'prec_train','prec_test','f1_train','f1_test','auc_train','auc_test']]

# create summary chart to compare
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def evaluate(results):
  
	
	# Create figure
    fig, ax = pl.subplots(2, 5, figsize = (12,7))
    tit_label={0:'Training ',1:'Testing '}

	# Constants
    bar_width = 0.2
    colors = ['#5F9EA0','#6495ED','#90EE90','#9ACD32']

    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['acc_train','rec_train','prec_train','f1_train','auc_train']):                 
           # Creative plot code
           ax[0, j].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])
           ax[0, j].set_xlim((-0.1, .9))
           ax[0,j].set_facecolor('white')
           pl.setp(ax[0,j].get_xticklabels(),visible=False)
           
        for j, metric in enumerate(['acc_test','rec_test','prec_test','f1_test','auc_test']):                 
           # Creative plot code
           ax[1, j].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])
           ax[1, j].set_xlim((-0.1, .9))
           ax[1,j].set_facecolor('white')
      
    for r in range(2):
        # Add unique y-labels
        ax[r, 0].set_ylabel("Accuracy Score")
        ax[r, 1].set_ylabel("Recall Score")
        ax[r, 2].set_ylabel("Precision score")
        ax[r, 3].set_ylabel("F1 - Score")
        ax[r, 4].set_ylabel("AUC-score")
        # Add titles
        ax[r, 0].set_title(tit_label[r]+"Accuracy Score")
        ax[r, 1].set_title(tit_label[r]+"Recall Score")
        ax[r, 2].set_title(tit_label[r]+"Precision score")
        ax[r, 3].set_title(tit_label[r]+"F1 - Score")
        ax[r, 4].set_title(tit_label[r]+"AUC-score")
		
    
   

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
        pl.legend(handles = patches, bbox_to_anchor = (-2, 2.4), \
               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')
    

	# Aesthetics
    pl.suptitle("Performance Metrics for Four Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    

def comp_stats(results):
  
	
	# Create figure
    fig, ax = pl.subplots(1, 1, figsize = (4,4))
    tit_label={0:'Training ',1:'Testing '}

	# Constants
    bar_width = 0.2
    colors = ['c','g']
    start_l=-0.2

    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        if (k==0):
          bar_l=-0.6
        else:
          bar_l=-0.4
        for j, metric in enumerate(['acc_test','rec_test','prec_test','f1_test','auc_test']):                 
           bar_l=bar_l+.6 
           ax.bar(bar_l, results[learner][metric], width = bar_width, color = colors[k])
           

    ax.set_xlim((0, 3))
    ax.set_xticks([.2, .8, 1.4,2,2.6])
    ax.set_xticklabels(["Accuracy", "Recall", "Precision","F1","AUC"])
    ax.set_ylabel("Score")
    ax.set_facecolor('white')   
	# Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
        pl.legend(handles = patches, bbox_to_anchor = (0.4, 1.22), \
               loc = 'upper center', borderaxespad = 0., ncol = 2, fontsize = 10)
    
    rects = ax.patches
    labels=[]

 	 # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['acc_test','rec_test','prec_test','f1_test','auc_test']):                 
           labels.append("%.4f" % results[learner][metric])

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height*1.02, label, ha='center', va='bottom',rotation='vertical')



	# Aesthetics
    pl.suptitle("Metrics for Tuned Models", fontsize = 14, y = 1.20)
    pl.tight_layout()
    pl.show()

evaluate(results)
# looks like RF is the best model in this dataset!

# get variable importance 
# feature importance
plt.figure(figsize=(6, 6))
feature_importance = pd.Series(clf_rf.feature_importances_).sort_values(ascending=False)
feature_importance.plot(kind='bar', title='Feature Importance')
plt.ylabel('Feature Importance Score')

(pd.Series(clf_rf.feature_importances_, index=X_train.columns)
   .nlargest(10)
   .plot(kind='barh'))    

# create performance for 