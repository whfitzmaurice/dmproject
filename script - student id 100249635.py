## student id 100249635

import pandas as pd
import pandas.io.formats.style
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
import seaborn as sns

data_file_path = "https://whfitzmaurice.github.io/dmproject/DiabetesClassificationDataset2022.csv"
data = pd.read_csv(data_file_path)
dict_file_path = "https://whfitzmaurice.github.io/dmproject/DataDictionaryCoursework2022.csv"
dictionary = pd.read_csv(dict_file_path)

data.shape # 79159 records with 88 variables
# as df for export
data_shape = pd.DataFrame(index=["rows","columns"],columns=["qty"],data=data.shape) 
data_shape.style

data.info() # indicates non-null count and datatypes

## dtypes for export
dtypes = pd.DataFrame(data.dtypes)

## to csv then latex
data_shape.to_csv("data_shape.csv")
dtypes.to_csv("dtypes.csv")

#########################################################################################################

## PART 1

## split dataset into categoric and numeric based on data dictionary data types
## transfers bmi from categoric list to numeric
num = dictionary.loc[dictionary["Data Type"] == "numeric","Variable Name2"] # lists numeric
cat = dictionary.loc[dictionary["Data Type"] != "numeric","Variable Name2"] # lists categoric
numeric = num.values.tolist()
categoric = cat.values.tolist()
del num,cat
categoric.remove("bmi")
numeric.append("bmi")

data[numeric]
data[categoric]

## lists to csv then latex
numeric_table = pd.DataFrame(numeric, columns=['Numeric Feature'])
categoric_table = pd.DataFrame(categoric, columns=['Categoric Feature'])
numeric_table.to_csv("num_table.csv")
categoric_table.to_csv("cat_table.csv")

## most of the variables are numeric and many contain missing data
## 3 categoric variables contain minimal missing data

## number of missing data points
data.isnull().sum()
## for loop to show percentage missing values for each variable
for i in data.columns:
    pct_missing = np.mean(data[i].isna())
    print("{} - {}%".format(i, round(pct_missing*100)))
## many variables above 60% missing threshold


## DATA DICTIONARY OF DESCRIPTIVE STATS FOR NUMERIC FEATURES
numeric_stats = data[numeric].describe()
# adds median, range, missing, and constant values
for col in numeric:
    numeric_stats.loc['median',col]=data[col].median()
    numeric_stats.loc['missing',col]=data[col].isnull().sum()
    numeric_stats.loc['missing_pct',col]=data[col].isnull().sum()/len(data[col])
    numeric_stats.loc['unique_values',col]=data[col].nunique()
    numeric_stats.loc['unique_pct',col]=data[col].nunique()/len(data[col])
    numeric_stats.loc['range',col]=data[col].max()-data[col].min()
print(numeric_stats)
numeric_stats = numeric_stats.round(3)

## to csv then latex
numeric_stats.to_csv("numericstats.csv")


## features more than 60% missing
row = 10
value = 0.6
comp = [x for x in numeric_stats.columns if numeric_stats[x][row]>=value]
numeric_missing = numeric_stats[comp]

## to csv then latex
numeric_missing.to_csv("numericmissing.csv")

## data dictionary of categoric features
categoric_stats = data[categoric].describe()
# add missing and constant
for col in categoric:
    categoric_stats.loc['missing',col]=data[col].isnull().sum()
    categoric_stats.loc['missing_pct',col]=data[col].isnull().sum()/len(data[col])
    categoric_stats.loc['unique_values',col]=data[col].nunique()
    categoric_stats.loc['unique_pct',col]=data[col].nunique()/len(data[col])
print(categoric_stats)

## to csv then latex
categoric_stats.to_csv("categoricmissing.csv")

#######################################################################################################

## PART 2

## CLEANING AND PREPROCESSING
# REMOVE FEATURES

## remove features with all null values
data.drop(columns="readmission_status", inplace=True)

## remove irrelevant features
data.drop(columns=["encounter_id","hospital_id","icu_type"],inplace=True)

## remove features with missing percentage over 60%
limitPer = len(data) * .60
data = data.dropna(thresh=limitPer, axis=1)

for i in data.columns:
    pct_missing = np.mean(data[i].isna())
    print("{} - {}%".format(i, round(pct_missing*100)))

## update numeric and categoric lists
headers = (list(data.columns.values))
## gen new list of categoric features
categoric_new = [i for i in headers if i not in numeric]
## gen new list of numeric features
numeric_new = [i for i in headers if i not in categoric]

data[numeric_new]
data[categoric_new]

## check for categoric features with large imbalance
## KDE PLOTS FOR FEATURES

DM_pos = data[data['diabetes_mellitus']==1]
DM_neg = data[data['diabetes_mellitus']==0]

## plot for each numeric feature
for col in numeric_new:
    plt.figure()
    sns.kdeplot(data=DM_pos[col],label="DM_pos",shade=True,bw_method=1.5)
    sns.kdeplot(data=DM_neg[col],label="DM_neg",shade=True,bw_method=1.5)
    plt.legend()
    plt.title("Distribution of " +col)
    plt.show()
    
## subplot

sns.set()

#define plotting region (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(12,8))
fig.suptitle("KDE plots for first 4 numeric features")

#create boxplot in each subplot

sns.kdeplot(data=DM_pos['age'], label="DM_pos",shade=True, ax=axes[0,0])
sns.kdeplot(data=DM_neg['age'], label="DM_neg",shade=True, ax=axes[0,0])
sns.kdeplot(data=DM_pos['height'], label="DM_pos",shade=True, ax=axes[0,1])
sns.kdeplot(data=DM_neg['height'], label="DM_neg",shade=True, ax=axes[0,1])
sns.kdeplot(data=DM_pos['weight'], label="DM_pos",shade=True, ax=axes[1,0])
sns.kdeplot(data=DM_neg['weight'], label="DM_neg",shade=True, ax=axes[1,0])
sns.kdeplot(data=DM_pos['bmi'], label="DM_pos",shade=True, ax=axes[1,1])
sns.kdeplot(data=DM_neg['bmi'], label="DM_neg",shade=True, ax=axes[1,1])



## catplots for categoric features
## catplots require matplotlib 3.4 or newer
    
for col in categoric_new:
    plt.figure()
    ax = sns.countplot(y=data[col], order=data[col].value_counts(ascending=False).index)
    sns.countplot(y=col,data=data,order=data[col].value_counts(ascending=False).index)
    abs_values = data[col].value_counts(ascending=False)
    rel_values = data[col].value_counts(ascending=False, normalize=True).values * 100
    labels = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
    plt.bar_label(ax.containers[1], labels=labels)
    
## subplot of countplots
sns.set()

#define plotting region (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(12,8))
fig.suptitle("Countplot for 4 categorical features")

#create boxplot in each subplot

ax = sns.countplot(y=data['gender'], order=data['gender'].value_counts(ascending=False).index, ax=axes[0,0])
abs_values = data['gender'].value_counts(ascending=False)
rel_values = data['gender'].value_counts(ascending=False, normalize=True).values * 100
labels = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
plt.bar_label(ax.containers[0], labels=labels)
ax = sns.countplot(y=data['aids'], order=data['aids'].value_counts(ascending=False).index, ax=axes[0,1])
abs_values = data['aids'].value_counts(ascending=False)
rel_values = data['aids'].value_counts(ascending=False, normalize=True).values * 100
labels = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
plt.bar_label(ax.containers[0], labels=labels)
ax = sns.countplot(y=data['ethnicity'], order=data['ethnicity'].value_counts(ascending=False).index, ax=axes[1,0])
abs_values = data['ethnicity'].value_counts(ascending=False)
rel_values = data['ethnicity'].value_counts(ascending=False, normalize=True).values * 100
labels = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
plt.bar_label(ax.containers[0], labels=labels)
ax = sns.countplot(y=data['leukemia'], order=data['leukemia'].value_counts(ascending=False).index, ax=axes[1,1])
abs_values = data['leukemia'].value_counts(ascending=False)
rel_values = data['leukemia'].value_counts(ascending=False, normalize=True).values * 100
labels = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
plt.bar_label(ax.containers[0], labels=labels)

## check for constants in remaining features

#pip install fast_ml
from fast_ml.utilities import display_all
from fast_ml.feature_selection import get_constant_features

# get constant features
const = get_constant_features(data, threshold=1)
print(const)
# no constant features

# get quasi constant features (99%)
quasi_const = get_constant_features(data,threshold=0.99, dropna=False)
print(quasi_const)
## 3 quasi constant features

## remove quasi constant features
data.drop(columns=["aids","leukemia","lymphoma"],inplace=True)

## update numeric and categoric lists
headers = (list(data.columns.values))
## gen new list of categoric features
categoric_new = [i for i in headers if i not in numeric]
## gen new list of numeric features
numeric_new = [i for i in headers if i not in categoric]

data[numeric_new]
data[categoric_new]

###########################

## MISSING DATA HANDLING

## impute missing values
from sklearn.impute import SimpleImputer

## split into numeric and categoric
numeric_data = data[numeric_new].copy()
categoric_data = data[categoric_new].copy()

#gen imputers for numeric and categoric
num_imputer = SimpleImputer(missing_values=np.nan) ## imputes mean to numeric features
cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent") ## imputes most frequent for categoric features
## apply imputers
numeric_data = num_imputer.fit_transform(numeric_data)
categoric_data = cat_imputer.fit_transform(categoric_data)
## return to dataframe from array
numeric_data = pd.DataFrame(numeric_data, columns=list(numeric_new))
categoric_data = pd.DataFrame(categoric_data, columns=list(categoric_new)) 
## check nans
numeric_data.isna().sum()
categoric_data.isna().sum()
## rejoin dataframes
data = pd.concat([categoric_data, numeric_data], axis=1)
data.isnull().sum()

##########################

## CHECK FOR HIGHLY CORRELATED FEATURES

## correlation check
matrix = data.corr()
print(matrix)
## correlation matrix shows initial signs of correlation between some variables
## view correlation heatmap
sns.heatmap(matrix)
plt.figure(figsize=(30,30))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
_ = sns.heatmap(matrix, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
## mask to remove half of heatmap
plt.figure(figsize=(30,30))
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(matrix, mask=mask, center=0, annot=True,fmt='.2f', square=True, cmap=cmap)

## list variables with highest correlation
correlations = matrix.unstack()
corr_rank = correlations.sort_values(kind="quicksort", ascending=False)
print(corr_rank)

## remove highly correlated values
matrix_abs = data.corr().abs() # use absolute values
## create mask to subset variables in a reduced matrix
mask = np.triu(np.ones_like(matrix_abs, dtype=bool))
reduced_matrix = matrix_abs.mask(mask)
## identify features that has correlation above 0.9
to_drop = [x for x in reduced_matrix.columns if any(reduced_matrix[x] > 0.9)]
## drop identified features
data_reduced = data.drop(to_drop, axis=1)

## update numeric and categoric variables
headers = (list(data_reduced.columns.values))
categoric_new = [i for i in headers if i not in numeric]
numeric_new = [i for i in headers if i not in categoric]



############################

## OUTLIER DETECTION W/ ISOLATION FOREST

## outlier detection with IsolationForest
from sklearn.ensemble import IsolationForest 
## create predictor for outliers, outliers = -1
isofor = IsolationForest(max_samples=79159, random_state=1, contamination=0.01)
predict = isofor.fit_predict(data_reduced[numeric_new])
print(predict)
## calculate total outliers
isofor_outliers = 0
for i in predict:
    if i == -1:
        isofor_outliers = isofor_outliers+1
print(isofor_outliers)

## visualise outliers
outliers_array = data_reduced[numeric_new].to_numpy()
outliers = np.where(predict == -1)
values = outliers_array[outliers]
plt.figure(figsize=(20,12))
plt.scatter(outliers_array[:,0], outliers_array[:,1], label="Data points")
plt.scatter(values[:,0], values[:,1], color='y', label="Outliers")
plt.title("IsolationForest Outlier Detection")
plt.axis("tight")
plt.xlabel("Number of outliers: %d" % (isofor_outliers))
plt.legend(loc="best")
plt.show()
 
## keep outliers as data will be transformed at later stage

###########################

## ENCODING

## encode categoric features

## drop target variable from categoric list (do not want it encoded)
target = data_reduced.diabetes_mellitus
data_reduced.drop(columns="diabetes_mellitus", inplace=True)
categoric_new.remove("diabetes_mellitus")

from sklearn.preprocessing import OneHotEncoder

## generate encoder
hot_encoder = OneHotEncoder(handle_unknown="ignore")
## encode features
encoded = pd.DataFrame(hot_encoder.fit_transform(data_reduced[categoric_new]).toarray())
encoded.columns = hot_encoder.get_feature_names(categoric_new)
print(encoded)
## drop unencoded categoric features 
data_reduced = data_reduced.drop(columns=categoric_new)
## reset indexes for concat
data_reduced.reset_index(inplace=True)
encoded.reset_index(inplace=True)
data_reduced.isna().sum()
encoded.isna().sum()
## concat to dataset
data = pd.concat([data_reduced,encoded],axis=1)
data.drop(columns=["index"],inplace=True)
data.info()
data.isnull().sum()
target.isnull().sum()
data.info
target
data.to_csv("cleaned_data.csv")
########################################################################################################

## PART 3

## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

## create new instance of dataset containing feature columns
feature_cols = data.columns
## split features and target
X = data[feature_cols]
y = pd.to_numeric(target)

X.info()
y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=369)

###############################

## FEATURE SELECTION

## select best features using SelectKBest
from sklearn.feature_selection import SelectKBest, mutual_info_classif

## generate selector for top 10 best features
selector_10 = SelectKBest(score_func=mutual_info_classif, k=10)
## transform data
features = selector_10.fit_transform(X_train, y_train)
## display features and scores
feature_names = X_train.columns.values[selector_10.get_support()]
feature_scores = selector_10.scores_[selector_10.get_support()]
names_scores = list(zip(feature_names, feature_scores))
features_to_use = pd.DataFrame(data=names_scores, columns=["Feature name","Mutual info"])
features_to_use = features_to_use.sort_values(["Mutual info","Feature name"],ascending=[False,True])
features_to_use
## convert to list
feats_10 = list(features_to_use["Feature name"])
# top 10 best features
feats_10

features_to_use.to_csv("top10features.csv")

## generate selector for all features

selector_all = SelectKBest(score_func=mutual_info_classif, k="all")
## transform data
features = selector_all.fit_transform(X_train, y_train)
## display features and scores
feature_names = X_train.columns.values[selector_all.get_support()]
feature_scores = selector_all.scores_[selector_all.get_support()]
names_scores = list(zip(feature_names, feature_scores))
features_to_use = pd.DataFrame(data=names_scores, columns=["Feature name","Mutual info"])
features_to_use = features_to_use.sort_values(["Mutual info","Feature name"],ascending=[False,True])
features_to_use
## convert to list
feats_all = list(features_to_use["Feature name"])
# top 10 best features
feats_all

features_to_use.to_csv("allfeatures.csv")
############################

## BALANCING

## value counts on train_y
y_train.value_counts()
# imbalanced towards 0 (negative)

## upsample train data minority class to equal number of samples

from sklearn.utils import resample

## concat train data back together
resamples = X_train.copy()
resamples['target']=y_train.values
## split out minority and majority classes
majority = resamples[resamples['target']==0]
minority = resamples[resamples['target']==1]
## upsample minority
minority_upsample = resample(minority, replace=True, n_samples=40465, random_state=369)
## combine majority class with upsampled minority
upsampled = pd.concat([majority, minority_upsample])
## check upsample
sns.countplot(x="target", data=upsampled)
## compare raw data and balanced data
## raw
sns.stripplot(x=target,y=data['weight'], data=data, jitter=0.3)
sns.despine
## balanced
sns.stripplot(x=upsampled['target'],y=upsampled['weight'], data=upsampled,jitter=0.3)
sns.despine

## classification using upsampled data
## create secondary train data using only top 10 features
X_upsample_10 = upsampled[feats_10]
y_upsample_10 = upsampled.target
X_upsample = upsampled
y_upsample = upsampled.target
X_upsample.drop(columns=("target"),inplace=True)
X_upsample.head()
y_upsample.head()

##############################

## SUPERVISED ALGORITHMS FOR BINARY CLASSIFICATION SELECTED BY ENSEMBLE


## create instances of each classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class1 = RandomForestClassifier()
class2 = SVC()
class3 = GaussianNB()

classifier_list = [class1,class2,class3]

## create instance of transformer
from sklearn.preprocessing import StandardScaler

trans = StandardScaler()

## create multiple pipelines of transformers and classifiers
from sklearn.pipeline import Pipeline
from sklearn.utils import *

## gen pipeline of transformers and classifiers
pipelines = [Pipeline([("tf", trans),("clf",clf)])for clf in [class1,class2,class3]]


## fit each pipeline to train data containing top 10 features
[pipeline.fit(X_upsample_10,y_upsample_10) for pipeline in pipelines]

## accuracy scores for pipelines containing top 10 features
## gen test data for 10 selected features
X_test_10 = X_test[feats_10]

## accuracy scores for top 10 features
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

accs_10 = [(pipeline[1].__class__.__name__,accuracy_score(y_test, pipeline.predict(X_test_10)))for pipeline in pipelines]
baccs_10 = [(pipeline[1].__class__.__name__,balanced_accuracy_score(y_test, pipeline.predict(X_test_10)))for pipeline in pipelines]
f1s_10 = [(pipeline[1].__class__.__name__,f1_score(y_test, pipeline.predict(X_test_10)))for pipeline in pipelines]

## fit each pipeline to train data containing all features
[pipeline.fit(X_upsample,y_upsample) for pipeline in pipelines]

## accuracy scores for pipelines containing all features
accs = [(pipeline[1].__class__.__name__,accuracy_score(y_test, pipeline.predict(X_test)))for pipeline in pipelines]
baccs = [(pipeline[1].__class__.__name__,balanced_accuracy_score(y_test, pipeline.predict(X_test)))for pipeline in pipelines]
f1s = [(pipeline[1].__class__.__name__,f1_score(y_test, pipeline.predict(X_test)))for pipeline in pipelines]

## higher scores for using all features rather than top 10

## convert result lists to dataframe
accs_df = pd.DataFrame(accs, columns=["clf","acc_score"])
baccs_df = pd.DataFrame(baccs, columns=["clf","bacc_score"])
f1s_df = pd.DataFrame(f1s, columns=["clf","f1_score"])
accs_10_df = pd.DataFrame(accs_10, columns=["clf","acc_10_score"])
baccs_10_df = pd.DataFrame(baccs_10, columns=["clf","bacc_10_score"])
f1s_10_df = pd.DataFrame(f1s_10, columns=["clf","f1_10_score"])
from functools import reduce
dfs = [accs_df,baccs_df,f1s_df,accs_10_df,baccs_10_df,f1s_10_df]
all_results = reduce(lambda  left,right: pd.merge(left,right,on=['clf'],how='outer'), dfs)
all_results.to_csv("accuracyscores.csv")

## precision and recall on models
## refit pipelines to X_upsample and y_upsample
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve

## random forest
plt.figure()
plot_precision_recall_curve(pipelines[0], X_test, y_test)
plt.title("Random Forest Precision Recall Curve")
plt.show()

plt.figure()
plot_roc_curve(pipelines[0], X_test, y_test)
plt.title("Random Forest Precision ROC/AUC")
plt.show()


## SVC
plt.figure()
plot_precision_recall_curve(pipelines[1], X_test, y_test)
plt.title("SVC Precision Recall Curve")
plt.show()

plt.figure()
plot_roc_curve(pipelines[1], X_test, y_test)
plt.title("SVC Precision ROC/AUC")
plt.show()

## gaussian NB
plt.figure()
plot_precision_recall_curve(pipelines[2], X_test, y_test)
plt.title("Gaussian NB Precision Recall Curve")
plt.show()

plt.figure()
plot_roc_curve(pipelines[2], X_test, y_test)
plt.title("Gaussian NB Precision ROC/AUC")
plt.show()

## grid search on top model for tuning
## grid search to find optimal parameters 
from sklearn.model_selection import GridSearchCV
## define classifier parameters
RandomForestClassifier().get_params()
param_grid = {'clf__criterion':['gini','entropy'],'clf__n_estimators':[20,60,100]}

## find best sequence using gridsearch
grid_class_cv = GridSearchCV(pipelines[0], param_grid)
grid_class_cv.fit(X_upsample,y_upsample)
grid_score = grid_class_cv.score(X_test,y_test)
print(grid_score)
print(grid_class_cv.best_params_)

## scores and curves
plt.figure()
plot_precision_recall_curve(grid_class_cv, X_test, y_test)
plt.title("Random Forest Precision Recall Curve w/ Hyperparameter Optimisation")
plt.show()

plt.figure()
plot_roc_curve(grid_class_cv, X_test, y_test)
plt.title("Random Forest Precision ROC/AUC w/ Hyperparameter Optimisation")
plt.show()


#############################

## PART 4

## UNSUPERVISED CLUSTERING

## regenerate upsampled data
cluster_X = X_upsample_10.copy()
cluster_y = y_upsample_10.copy()

## use 30% of data for clustering
X_train,X_test,y_train,y_test = train_test_split(cluster_X,cluster_y, test_size=0.3, random_state=000)

## CLUSTERING USING SELECTED FEATURES
from sklearn.cluster import KMeans

## elbow plot
kmeans_models = [KMeans(n_clusters=k).fit(X_test) for k in range(1,10)]
inertia = [model.inertia_ for model in kmeans_models]

plt.plot(range(1,10),inertia)
plt.title("KMeans Clustering Elbow Plot")
plt.xlabel("Clusters")
plt.ylabel("Within Cluster Sum of Squares")
plt.show()

## elbow around 2 clusters

from sklearn.metrics import silhouette_score

## silhouette score for elbow plot
sil_score = [silhouette_score(X_test, model.labels_) for model in kmeans_models[1:5]]

plt.plot(range(2,6), sil_score, "bo-")
plt.xticks([2,3,4,5])
plt.title("Silhouette Score vs Clusters")
plt.xlabel("Clusters")
plt.ylabel("Silhouette Score")
plt.show()

## highest score 2

## train model for silhouette score 2
kmeans = KMeans(n_clusters=2, random_state=000)
kmeans.fit(X_test)

## identify clusters
X_test['cluster_id'] = kmeans.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(data=X_test, x="d1_glucose_min",y="weight", hue="cluster_id")
plt.legend(title='target',loc='best', labels=['with diabetes_mellitus-1', 'without diabetes_mellitus-0'])
plt.show()

## PCA to reduce dimensionality
from sklearn.decomposition import PCA

## standardise data with standard scaler
scale = StandardScaler()

pca_data = X_test.join(y_test)
pca_data_scaled = scale.fit_transform(pca_data)

## analyse with 2 princple componants
pca_2 = PCA(n_components=2)
pc = pca_2.fit_transform(pca_data_scaled)

## dataframe containing principle componants
pc_df = pd.DataFrame(pc, columns=["pc1","pc2"])
pc_target = pca_data.target
pc_target.reset_index(drop=True, inplace=True)
pc_df = pc_df.join(pc_target)

## visualise with scatter
plt.figure(figsize=(12,8))
sns.scatterplot(data=pc_df, x="pc1",y="pc2", hue="target")
plt.legend(title='target',loc='best', labels=['with diabetes_mellitus-1', 'without diabetes_mellitus-0'])
plt.show()

## scoring accuracy of clustering
from sklearn.metrics.cluster import adjusted_rand_score

## get kmeans predictions
X_test_pred = X_test.copy()
X_test_pred.drop(columns=('cluster_id'),inplace=True)
pred_labels = kmeans.predict(X_test_pred)

## rand score
rand_score = adjusted_rand_score(kmeans.labels_, pred_labels)
print(rand_score)
## = 1
## probable error

## adjusted mutual info score
from sklearn.metrics.cluster import adjusted_mutual_info_score
AMI_score = adjusted_mutual_info_score(kmeans.labels_,pred_labels)
print(AMI_score)
## = 1
## probable error 