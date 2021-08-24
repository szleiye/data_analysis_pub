[TOC]



# 一、sklearn

#### 模型保存 pickle

```PYTHON
# 1
from sklearn import svm
from sklearn import datasets
import pickle

clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])


# 2 输出pickle文件
output = open('./model/feature_select_lgb.pkl', 'wb')
pickle.dump(bst, output)
output.close()
```



## 数据处理

#### [不同列用不同预处理方法 ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer)

```PYTHON
# http://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    
# Author: Pedro Morales <part.morales@gmail.com>
#
# License: BSD 3 clause

from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# Read data from Titanic dataset.
titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)

# We will train our classifier with the following features:
# Numeric Features:
# - age: float.
# - fare: float.
# Categorical Features:
# - embarked: categories encoded as strings {'C', 'S', 'Q'}.
# - sex: categories encoded as strings {'female', 'male'}.
# - pclass: ordinal integers {1, 2, 3}.

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

# output:
# model score: 0.790
```

#### columntransformer 还原变量列名

```PYTHON
def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features

pd.DataFrame(transformed_data, 
             columns=get_ct_feature_names(combined_pipe))
```



 #### 取columntransformer 中 ordinal encoder 的属性

```PYTHON
preprocessor.transformers_[0][1].named_steps['ordinal'].categories_
```





#### [对齐数据框 pandas.DataFrame.align](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.align.html)



#### 转换数据类型以节省内存

```PYTHON
def convert_types(df, print_info = False):
    original_memory = df.memory_usage().sum()
    
    for c in df:  # Iterate through each column
        
        # Convert ids and booleans to integers
        if 'SK_ID' in c:
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # 转换 object to categories
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
    
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df
```



### 类别变量转换

#### [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.get_dummies.html)

```PYTHON
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
```



#### [preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)



#### [preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder )

将target label转换成0 至 n_class-1 之间的值。一般用在y上转换，不用再x上。

|方法|备注|
|-----|-----|
| [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.fit)(self, y) | Fit label encoder                           |
| [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.fit_transform)(self, y) | Fit label encoder and return encoded labels |
| [`get_params`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.get_params)(self[, deep]) | Get parameters for this estimator.          |
| [`inverse_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.inverse_transform)(self, y) | Transform labels back to original encoding. |
| [`set_params`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.set_params)(self, \*\*params) | Set the parameters of this estimator.       |
| [`transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.transform)(self, y) | Transform labels to normalized encoding.    |

```PYTHON
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()

>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']

>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)

>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
```



### 变量衍生

#### [多项式衍生 preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures )

返回输入变量所有可能的多项式组合

例如输入`[a, b]`，返回` [1, a, b, a^2, ab, b^2] `

| 参数             | 输入值                         | 备注                                                         |
| ---------------- | ------------------------------ | ------------------------------------------------------------ |
| degree           | integer, Default = 2           |                                                              |
| interaction_only | boolean, default = False       | If true, only interaction features are produced: features that are products of at most `degree` *distinct* input features (so not `x[1] ** 2`, `x[0] * x[2] ** 3`, etc.). |
| include_bias     | boolean, default = True        | 含不含一列全是1                                              |
| order            | str in {‘C’, ‘F’}, default ‘C’ | Order of output array in the dense case. ‘F’ order is faster to compute, but may slow down subsequent estimators. |





| 方法| 备注|
|---|---|
| [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.fit)(self, X[, y]) | Compute number of output features.       |
| [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.fit_transform)(self, X[, y]) | Fit to data, then transform it.          |
| [`get_feature_names`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.get_feature_names)(self[, input_features]) | Return feature names for output features |
| [`get_params`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.get_params)(self[, deep]) | Get parameters for this estimator.       |
| [`set_params`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.set_params)(self, \*\*params) | Set the parameters of this estimator.    |
| [`transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.transform)(self, X) | Transform data to polynomial features    |



```PYTHON
from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)

# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

# 输出变量名
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
```



## 样本处理

### 随机欠采样

```PYTHON
# ------ 方法1 ------ 
# Shuffle the Dataset.
shuffled_df = credit_df.sample(frac=1,random_state=4)

# Put all the fraud class in a separate dataset.
fraud_df = shuffled_df.loc[shuffled_df['Class'] == 1]

#Randomly select 492 observations from the non-fraud (majority class)
non_fraud_df = shuffled_df.loc[shuffled_df['Class'] == 0].sample(n=492,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([fraud_df, non_fraud_df])

#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))

sns.countplot('Class', data=normalized_df)
plt.title('Balanced Classes')
plt.show()

# ------ 方法2 ------
# RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X, y)

```



#### SMOTE算法(过采样)

```PYTHON
from imblearn.over_sampling import SMOTE

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='minority', random_state=7)

# Fit the model to generate the data.
oversampled_trainX, oversampled_trainY = sm.fit_sample(credit_df.drop('Class', axis=1), credit_df['Class'])
oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
oversampled_train.columns = normalized_df.columns

```

#### ADASYN

```PYTHON
from imblearn.over_sampling import ADASYN

X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X, y)
sorted(Counter(y_resampled_adasyn).items())
# [(0, 2522), (1, 2520), (2, 2532)]
```



#### 集成算法+采样

##### [BalancedBaggingClassifier](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedBaggingClassifier.html#imblearn.ensemble.BalancedBaggingClassifier)

```PYTHON
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto', 
                                replacement=False, 
                                random_state=0)
y_train = credit_df['Class']
X_train = credit_df.drop(['Class'], axis=1, inplace=False)

#Train the classifier.
bbc.fit(X_train, y_train)
preds = bbc.predict(X_train)

```



```PYTHON
# 生成不平衡分类数据集
from collections import Counter
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.1, 0.05, 0.85],
                           class_sep=0.8, random_state=2018)
Counter(y)
# Counter({2: 2532, 1: 163, 0: 305})

# 使用RandomOverSampler从少数类的样本中进行随机采样来增加新的样本使各个分类均衡
from imblearn.over_sampling import RandomOverSampler
 
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
sorted(Counter(y_resampled).items())
# [(0, 2532), (1, 2532), (2, 2532)]

# SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
from imblearn.over_sampling import SMOTE
 
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
 
sorted(Counter(y_resampled_smote).items())
# [(0, 2532), (1, 2532), (2, 2532)]

# ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本
from imblearn.over_sampling import ADASYN

X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X, y)
 
sorted(Counter(y_resampled_adasyn).items())
# [(0, 2522), (1, 2520), (2, 2532)]

# RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X, y)
 
sorted(Counter(y_resampled).items())
# [(0, 163), (1, 163), (2, 163)]

# 在之前的SMOTE方法中, 当由边界的样本与其他样本进行过采样差值时, 很容易生成一些噪音数据. 因此, 在过采样之后需要对样本进行清洗. 
# 这样TomekLink 与 EditedNearestNeighbours方法就能实现上述的要求.
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X, y)
 
sorted(Counter(y_resampled).items())
# [(0, 2111), (1, 2099), (2, 1893)]

from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
 
sorted(Counter(y_resampled).items())
# [(0, 2412), (1, 2414), (2, 2396)]

# 使用SVM的权重调节处理不均衡样本 权重为balanced 意味着权重为各分类数据量的反比
from sklearn.svm import SVC  
svm_model = SVC(class_weight='balanced')
svm_model.fit(X, y)

# # EasyEnsemble 通过对原始的数据集进行随机下采样实现对数据集进行集成.
# EasyEnsemble 有两个很重要的参数: (i) n_subsets 控制的是子集的个数 and (ii) replacement 决定是有放回还是无放回的随机采样.
from imblearn.ensemble import EasyEnsemble
ee = EasyEnsemble(random_state=0, n_subsets=10)
X_resampled, y_resampled = ee.fit_sample(X, y)
sorted(Counter(y_resampled[0]).items())
# [(0, 163), (1, 163), (2, 163)]

# BalanceCascade(级联平衡)的方法通过使用分类器(estimator参数)来确保那些被错分类的样本在下一次进行子集选取的时候也能被采样到. 同样, n_max_subset 参数控制子集的个数, 以及可以通过设置bootstrap=True来使用bootstraping(自助法).
from imblearn.ensemble import BalanceCascade
from sklearn.linear_model import LogisticRegression
bc = BalanceCascade(random_state=0,
                    estimator=LogisticRegression(random_state=0),
                    n_max_subset=4)
X_resampled, y_resampled = bc.fit_sample(X, y)
 
sorted(Counter(y_resampled[0]).items())
# [(0, 163), (1, 163), (2, 163)]

# BalancedBaggingClassifier 允许在训练每个基学习器之前对每个子集进行重抽样. 简而言之, 该方法结合了EasyEnsemble采样器与分类器(如BaggingClassifier)的结果.
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                ratio='auto',
                                replacement=False,
                                random_state=0)
bbc.fit(X, y) 


```



## Cross-validation



#### [训练测试集划分 train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html )

sklearn.model_selection.train_test_split(*arrays, **options)

| 参数         | 类型                                                       | 备注                                                         |
| ------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| arrays       |                                                            | sequence of indexables with same length / shape[0]           |
| test_size    | float, int or None, optional (default=None, 0.25)          | the proportion of the dataset to include in the test split   |
| train_size   | float, int or None, optional                               |                                                              |
| random_state | int, RandomState instance or None, optional (default=None) |                                                              |
| shuffle      | boolean, optional (default=True)                           | Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None. |
| stratify     | array-like or None (default=None)                          | If not None, data is split in a stratified fashion, using this as the class labels. |



| 返回      | 类型                           | 备注                                       |
| --------- | ------------------------------ | ------------------------------------------ |
| splitting | list, length=2  \* len(arrays) | List containing train-test split of inputs |



```PYTHON
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 50)
```





#### [K-Fold检验 model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold )

| 参数         | 类型                                                     | 备注                                                         |
| ------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| n_splits     | int, default=5                                           | Number of folds. Must be at least 2.<br />*Changed in version 0.22:* `n_splits` default value changed from 3 to 5. |
| shuffle      | boolean, optional                                        | Whether to shuffle the data before splitting into batches    |
| random_state | nt, RandomState instance or None, optional, default=None | If int, random_state is the seed used by the random number generator; <br /> Only used when `shuffle` is True. This should be left to None if `shuffle` is False. |



| 方法                                                         | 备注                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`get_n_splits`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold.get_n_splits)(self[, X, y, groups]) | Returns the number of splitting iterations in the cross-validator |
| [`split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold.split)(self, X[, y, groups]) | 返回样本的编号 Generate indices to split data into training and test set. |



```python
# 官方例子
>>> import numpy as np
>>> from sklearn.model_selection import KFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4])
>>> kf = KFold(n_splits=2)
>>> kf.get_n_splits(X)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
>>> for train_index, test_index in kf.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]
```

```PYTHON
for train_indices, valid_indices in k_fold.split(features):
    
    # Training data for the fold
    train_features, train_labels = features[train_indices], labels[train_indices]
    # Validation data for the fold
    valid_features, valid_labels = features[valid_indices], labels[valid_indices]
    
    # Create the model
    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                               class_weight = 'balanced', learning_rate = 0.05, 
                               reg_alpha = 0.1, reg_lambda = 0.1, 
                               subsample = 0.8, n_jobs = -1, random_state = 50)
    
    # Train the model
    model.fit(train_features, train_labels, eval_metric = 'auc',
              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
              eval_names = ['valid', 'train'], categorical_feature = cat_indices,
              early_stopping_rounds = 100, verbose = 200)
    
    # Record the best iteration
    best_iteration = model.best_iteration_
    
    # Record the feature importances
    feature_importance_values += model.feature_importances_ / k_fold.n_splits
    
    # 对 test 集的样本进行预测， 取每次预测加总后平均的概率
    test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
    
    # 记录下本次out of fold的预测
    out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
    
    # Record the best score
    valid_score = model.best_score_['valid']['auc']
    train_score = model.best_score_['train']['auc']
    
    valid_scores.append(valid_score)
    train_scores.append(train_score)
    
    # Clean up memory
    gc.enable()
    del model, train_features, valid_features
    gc.collect()
```



## 决策树

###  [分类树 DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) 

#### 参数汇总表

| 参数                         | 备注 |
| ---------------------------- | ---- |
| **criterion**                |      |
| **splitter**                 |      |
| **max_depth**                |      |
| **min_samples_split**        |      |
| **min_samples_leaf**         |      |
| **min_weight_fraction_leaf** |      |
| **max_features**             |      |
| **random_state**             |      |
| **max_leaf_nodes**           |      |
| **min_impurity_decrease**    |      |
| **min_impurity_split**       |      |
| **class_weight**             |      |
| **ccp_alpha**                |      |



#### 方法汇总表
| 方法 | 备注 |
| ---- | ---- |
| [`apply`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.apply)(self, X[, check_input]) | Return the index of the leaf that each sample is predicted as. |
| [`cost_complexity_pruning_path`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path)(self, X, y[, …]) | Compute the pruning path during Minimal Cost-Complexity Pruning. |
| [`decision_path`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.decision_path)(self, X[, check_input]) | Return the decision path in the tree.                        |
| [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit)(self, X, y[, sample_weight, …]) | Build a decision tree classifier from the training set (X, y). |
| [`get_depth`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.get_depth)(self) | Return the depth of the decision tree.                       |
| [`get_n_leaves`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.get_n_leaves)(self) | Return the number of leaves of the decision tree.            |
| [`get_params`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.get_params)(self[, deep]) | Get parameters for this estimator.                           |
| [`predict`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict)(self, X[, check_input]) | Predict class or regression value for X.                     |
| [`predict_log_proba`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict_log_proba)(self, X) | Predict class log-probabilities of the input samples X.      |
| [`predict_proba`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict_proba)(self, X[, check_input]) | Predict class probabilities of the input samples X.          |
| [`score`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.score)(self, X, y[, sample_weight]) | Return the mean accuracy on the given test data and labels.  |
| [`set_params`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.set_params)(self, \*\*params) | Set the parameters of this estimator.                        |



#### [模型剪枝](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)

剪枝方法： 最小代价复杂度剪枝 Minimal Cost-Complexity Pruning

设子树的损失函数如下：
$$
R_\alpha(T) = R(T) + \alpha|T|
$$
剪枝前：$$R_\alpha(T_t) = R(T_t) + \alpha|T_t|$$

剪枝后：$$R_\alpha(t) = R(t) + \alpha $$

临界值：$$R_\alpha(T_t)=R_\alpha(t)$$ 或者 $$\alpha_{eff}(t)=\frac{R(t)-R(T_t)}{|T|-1} $$

> * 剪枝前的复杂度肯定比剪枝后的复杂度小，$R(T_t) < R(t)$ 
> * $\alpha_{eff}(t)$ 从公式上看代表每新增一个复杂度，换取不纯度下降的值。所以该值越小，说明该节点分支太复杂且对分类没有帮助

剪枝策略：

1. 计算每个node的临界值$\alpha_{eff}(t)$

2. 从临界值最小的地方开始剪枝

3. 重复2直到最小的$\alpha_{eff}(t)$ 满足设定的$\alpha$阈值为止

    > 也可以剪枝至只剩一个节点，然后用cross validation决定最优子树





####  模型评价[score(X, y)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.score )

返回模型的准确率(mean accuracy)
$$
\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i) 
$$


调用的方法是： [`sklearn.metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)





### 回归树

pass



### 属性和方法

####  tree的属性sklearn.tree._tree.Tree 

| 属性                    | 返回类型                                                     | 备注                               | 样例 |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------- | ---- |
| node_count              | int                                                          | 结点数量(含内部节点)               |      |
| capacity                | int                                                          | 树的深度                           |      |
| children_left           | array of int, shape [node_count]                             | 返回各节点的左子节点id，没有返回-1 |      |
| children_right          | array of int, shape [node_count]                             | 返回各节点的右子节点id，没有返回-1 |      |
| feature                 | array of int, shape [node_count]                             | feature[i]记录了对节点i的分裂变量  |      |
| threshold               | array of double, shape [node_count]                          | threshold[i]记录了对节点i的分裂值  |      |
| value                   | array of double, shape [node_count, n_outputs, max_n_classes] | 各节点内各类别的数量               |      |
| impurity                | array of double, shape [node_count]                          | 各节点的impurity                   |      |
| n_node_samples          | array of int, shape [node_count]                             | 各节点的样本数                     |      |
| weighted_n_node_samples | array of int, shape [node_count]                             | 各节点的加权样本数                 |      |



```PYTHON
# 储存树的属性
n_nodes = estimator.tree_.node_count  
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
value = estimator.tree_.value 
impurity = estimator.tree_.impurity 
n_node_samples = estimator.tree_.n_node_samples 

tree_stat = pd.DataFrame()
tree_stat['id'] = [i for i in range(0, n_nodes)]
tree_stat['children_left'] = children_left
tree_stat['children_right'] = children_right
tree_stat['feature'] = feature
tree_stat['threshold'] = threshold
# tree_stat['value'] = value
tree_stat['impurity'] = impurity
tree_stat['n_node_samples'] = n_node_samples
```



#### [提取树的规则]( https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree )

```PYTHON
# 输出规则格式1
def tree_to_code(tree, feature_names):
    '''
    parameters:
    -----------
    tree:  输入决策树estimator
    feature_names: 输入特征名称列表
    
    return:
    -------
    决策树规则的函数
    '''
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))  # 输出特征名称

    def recurse(node, depth):
        """
        node: 起始节点
        depth: 树深度，控制缩进
        
        递归算法
        """
        indent = "    " * depth  # 设置进位符
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
    
# 输出规则格式2
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))  # 输出特征名称

    con=[]
    def recurse(node, depth, con):
        """
        node: 起始节点
        depth: 树深度，控制缩进
        
        递归算法
        """
        indent = "    " * depth  # 设置进位符
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            con.append("{} <= {}".format( name, threshold))          
            recurse(tree_.children_left[node], depth + 1, con)
            
            con.append("{} > {}".format( name, threshold))
            recurse(tree_.children_right[node], depth + 1, con)
        else:
            print("if", " and \n    ".join(con), "\n    return {}".format(tree_.value[node]))
            con.pop()
    recurse(0, 1, con)
```



#### 决策树可视化

```PYTHON
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
```

```
# 直接在linux中将dot文件转成图
dot -Tpng iris_tree.dot -o iris_tree.png
```



## 随机森林 

#### [分类森林 sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier)



参数

| 参数         | 类型                            | 备注                               |
| ------------ | ------------------------------- | ---------------------------------- |
| n_estimators | integer, optional (default=100) | The number of trees in the forest. |
|              |                                 |                                    |



| 方法 | 备注 |
| ---- | ---- |
| [`apply`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.apply)(self, X) | Apply trees in the forest to X, return leaf indices.        |
| [`decision_path`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.decision_path)(self, X) | Return the decision path in the forest.                     |
| [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit)(self, X, y[, sample_weight]) | Build a forest of trees from the training set (X, y).       |
| [`get_params`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.get_params)(self[, deep]) | Get parameters for this estimator.                          |
| [`predict`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict)(self, X) | Predict class for X.                                        |
| [`predict_log_proba`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_log_proba)(self, X) | Predict class log-probabilities for X.                      |
| [`predict_proba`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba)(self, X) | Predict class probabilities for X.                          |
| [`score`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.score)(self, X, y[, sample_weight]) | Return the mean accuracy on the given test data and labels. |
| [`set_params`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.set_params)(self, \*\*params) | Set the parameters of this estimator.                       |



```PYTHON
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.datasets import make_blobs
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.tree import DecisionTreeClassifier

>>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
...     random_state=0)

>>> clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
...     random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean()
0.98...

>>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=2, random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean()
0.999...

>>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=2, random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean() > 0.999
True
```



## [lightGBM](https://lightgbm.readthedocs.io/en/latest/)

### Data Interface

#### 读取numpy数据

```python
data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(data, label=label)
```



#### 将数据存成 LightGBM binary 格式, 可快速读取

```PYTHON
train_data = lgb.Dataset('train.svm.txt')
train_data.save_binary('train.bin')
```



#### 高效利用内存的方法

* Numpy/Array/Pandas 对象很耗内存，LightGBM 的 `Dataset` 对象因为只存 discrete bins，所以省内存

* 优化内存的方法如下：

    1. Set `free_raw_data=True` (default is `True`) when constructing the `Dataset`

    2. Explicitly set `raw_data=None` after the `Dataset` has been constructed

    3. Call `gc`

        



### 原生API

#### 训练模型 lgb.train

```PYTHON
bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

# 保存模型
bst.save_model('model.txt')
```



#### 模型保存和加载

```PYTHON
# 模型保存
gbm.save_model('model.txt')
 
# 模型加载
gbm = lgb.Booster(model_file='model.txt')

```





#### cv

返回：字典，最佳树棵树的长度，存了auc值mean和std

```PYTHON
train_set = lgb.Dataset(data=train_features, label=train_labels)
test_set = lgb.Dataset(data=test_features, label=test_labels)

model = lgb.LGBMClassifier()
default_params = model.get_params()

del default_params['n_estimators'] # Remove the number of estimators because we set this to 10000 in the cv call

# Cross validation with early stopping
cv_results = lgb.cv(default_params,
                    train_set, 
                    num_boost_round = 10000,
                    early_stopping_rounds = 100, 
                    metrics = 'auc', 
                    nfold = N_FOLDS, 
                    seed = 42)
```



### 类sklearnAPI

#### [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier)

**方法**

|方法|备注|
|---|---|
| [`__init__`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.__init__)([boosting_type, num_leaves, …]) | Construct a gradient boosting model.                         |
| [`fit`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit)(X, y[, sample_weight, init_score, …]) | Build a gradient boosting model from the training set (X, y). |
| [`get_params`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.get_params)([deep]) | Get parameters for this estimator.                           |
| [`predict`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.predict)(X[, raw_score, num_iteration, …]) | Return the predicted value for each sample.                  |
| [`predict_proba`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.predict_proba)(X[, raw_score, num_iteration, …]) | Return the predicted probability for each class for each sample. |
| [`set_params`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.set_params)(**params) | Set the parameters of this estimator.                        |



**属性**
|属性|备注|
|---|---|
| [`best_iteration_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.best_iteration_) | Get the best iteration of fitted model.                   |
| [`best_score_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.best_score_) | Get the best score of fitted model.                       |
| [`booster_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.booster_) | Get the underlying lightgbm Booster of this model.        |
| [`classes_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.classes_) | Get the class label array.                                |
| [`evals_result_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.evals_result_) | Get the evaluation results.                               |
| [`feature_importances_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.feature_importances_) | Get feature importances.                                  |
| [`feature_name_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.feature_name_) | Get feature name.                                         |
| [`n_classes_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.n_classes_) | Get the number of classes.                                |
| [`n_features_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.n_features_) | Get the number of features of fitted model.               |
| [`objective_`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.objective_) | Get the concrete objective used while fitting this model. |



#### 模型保存和加载

```PYTHON
from sklearn.externals import joblib

# 模型存储
joblib.dump(gbm, 'loan_model.pkl')
# 模型加载
gbm = joblib.load('loan_model.pkl')
```



#### 例子

```PYTHON
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
# 加载数据
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
 
# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
 
# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
 
# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
 
# feature importances
print('Feature importances:', list(gbm.feature_importances_))
 
# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
```



## 评价标准

#### [sklearn.metrics.roc_curve]( https://blog.csdn.net/u014264373/article/details/80487766)

* fpr: 误报率，所有的好客户有多少被预测为坏客户
* tpr: 灵敏度，所有的坏客户有多少被预测到

```PYTHON
import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2) # thresholds是遍历scores。返回以各个thresholds下的fpr, tpr
```

```PYTHON
from sklearn.metrics import roc_auc_score,roc_curve,auc

y_pred = lr_model.predict_proba(x)[:,1]
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)
train_ks = abs(fpr_lr_train - tpr_lr_train).max()
print('train_ks : ',train_ks)

y_pred = lr_model.predict_proba(val_x)[:,1]
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)
val_ks = abs(fpr_lr - tpr_lr).max()
print('val_ks : ',val_ks)

from matplotlib import pyplot as plt
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')
plt.show()
```



## 模型选择

### Computing cross-validated metrics

| 方法              | 备注                                            |
| ----------------- | ----------------------------------------------- |
| cross_val_score   | 返回用k-折模型之后的平均分                      |
| cross_validate    | 可以返回多个评价指标，和训练时间；返回estimator |
| cross_val_predict | 返回每个样本做测试集时候得到的分数              |



#### [cross_val_score](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection.cross_val_score)

Evaluate a score by cross-validation

```PYTHON
# 5-fold
>>> from sklearn import metrics
>>> scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
>>> scores

# 用别的cv 
>>> from sklearn.model_selection import ShuffleSplit
>>> n_samples = X.shape[0]
>>> cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
>>> cross_val_score(clf, X, y, cv=cv)

# 用自定义的cv
>>> def custom_cv_2folds(X):
...     n = X.shape[0]
...     i = 1
...     while i <= 2:
...         idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
...         yield idx, idx
...         i += 1
...
>>> custom_cv = custom_cv_2folds(X)
>>> cross_val_score(clf, X, y, cv=custom_cv)
array([1.        , 0.973...])
```



#### [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)

Evaluate metric(s) by cross-validation and also record fit/score times.

- It allows specifying multiple metrics for evaluation.
- It returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.
- It can return estimators.



#### [cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict)

返回每个样本做测试集时候得到的分数， `cross_val_score`则是返回average over cross-validation folds

**返回值：**

| 返回值 | 类型 | 备注                                                         |
| ------ | ---- | ------------------------------------------------------------ |
| scores | 字典 | 包含`test_score`, `train_score`, `fit_time`, `score_time`, `estimator` |



## [贝叶斯调参](https://github.com/fmfn/BayesianOptimization)

```PYTHON
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

# 
optimizer.maximize(
    init_points=2,
    n_iter=3,
)

# 
print(optimizer.max)
>>> {'target': -4.441293113411222, 'params': {'y': -0.005822117636089974, 'x': 2.104665051994087}}

# 
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

>>> Iteration 0:
>>>     {'target': -7.135455292718879, 'params': {'y': 1.3219469606529488, 'x': 2.8340440094051482}}
>>> Iteration 1:
>>>     {'target': -7.779531005607566, 'params': {'y': -1.1860045642089614, 'x': 2.0002287496346898}}
>>> Iteration 2:
>>>     {'target': -19.0, 'params': {'y': 3.0, 'x': 4.0}}
>>> Iteration 3:
>>>     {'target': -16.29839645063864, 'params': {'y': -2.412527795983739, 'x': 2.3776144540856503}}
>>> Iteration 4:
>>>     {'target': -4.441293113411222, 'params': {'y': -0.005822117636089974, 'x': 2.104665051994087}}
```



## Pipeline

#### pipeline 中使用自定义函数

```PYTHON
class ColumnExtractor(object):

    def transform(self, X):
        cols = X[:,2:4] # column 3 and 4 are "extracted"
        return cols

    def fit(self, X, y=None):
        return self
    
clf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('reduce_dim', ColumnExtractor()),           
    ('classification', GaussianNB())   
    ])

# 一般性写法
import numpy as np

class ColumnExtractor(object):

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self

    clf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('dim_red', ColumnExtractor(cols=(1,3))),   # selects the second and 4th column      
    ('classification', GaussianNB())   
    ])
```



# [二、statsmodel](https://www.statsmodels.org/stable/index.html)

#### [ridge regression](https://stackoverflow.com/questions/40072870/statistical-summary-table-in-sklearn-linear-model-ridge)

**方法1**

```py
y, X = data.endog, data.exog

model = sm.OLS(y, X)
results_fu = model.fit()

print results_fu.summary()  # 输出普通线性回归报告


# 调参
frames = []
for n in np.arange(0, 0.25, 0.05).tolist():
    results_fr = model.fit_regularized(L1_wt=0, alpha=n, start_params=results_fu.params)

    results_fr_fit = sm.regression.linear_model.OLSResults(model, 
                                                           results_fr.params, 
                                                           model.normalized_cov_params)
    frames.append(np.append(results_fr.params, results_fr_fit.ssr))

    df = pd.DataFrame(frames, columns=data.exog_name + ['ssr*'])
df.index=np.arange(0, 0.25, 0.05).tolist()
df.index.name = 'alpha*'
df.T
```



[<img src="assets/sklearn_doc/fSR9N.png" alt="enter image description here" style="zoom: 67%;" />](https://i.stack.imgur.com/fSR9N.png)

```py
%matplotlib inline

fig, ax = plt.subplots(1, 2, figsize=(14, 4))

ax[0] = df.iloc[:, :-1].plot(ax=ax[0])
ax[0].set_title('Coefficient')

ax[1] = df.iloc[:, -1].plot(ax=ax[1])
ax[1].set_title('SSR')
```

[![enter image description here](assets/sklearn_doc/irutn.png)](https://i.stack.imgur.com/irutn.png)

In.

```python
# 调用OLSResults得到岭回归结果
results_fr = model.fit_regularized(L1_wt=0, alpha=0.04, start_params=results_fu.params)
final = sm.regression.linear_model.OLSResults(model, 
                                              results_fr.params, 
                                              model.normalized_cov_params)

print final.summary()
```

Out.

```py
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.543
Model:                            OLS   Adj. R-squared:                  0.516
Method:                 Least Squares   F-statistic:                     20.17
Date:                Wed, 19 Oct 2016   Prob (F-statistic):           5.46e-11
Time:                        17:22:49   Log-Likelihood:                -507.28
No. Observations:                  72   AIC:                             1023.
Df Residuals:                      68   BIC:                             1032.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -5.6375      4.554     -1.238      0.220     -14.724       3.449
x2           159.1412     63.781      2.495      0.015      31.867     286.415
x3            -8.1360      6.034     -1.348      0.182     -20.176       3.904
x4            44.2597     80.093      0.553      0.582    -115.564     204.083
==============================================================================
Omnibus:                       76.819   Durbin-Watson:                   1.694
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              658.948
Skew:                           3.220   Prob(JB):                    8.15e-144
Kurtosis:                      16.348   Cond. No.                         87.5
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```



方法2

```python
from statsmodels.tools.tools import pinv_extended
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train)

# If True, the model is refit using only the variables that have non-zero 
# coefficients in the regularized fit. The refitted model is not regularized.
result = model.fit_regularized(
                    method = 'elastic_net',
                    alpha = alp,
                    L1_wt = l1,
                    start_params = None,
                    profile_scale = False,
                    #refit = True,
                    refit = False,
                    maxiter = 9000,
                    zero_tol = 1e-5,
                    )

pinv_wexog,_ = pinv_extended(model.wexog)
normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))


final = sm.regression.linear_model.OLSResults(model, 
                                  result.params, 
                                  normalized_cov_params)
#print(final.summary())
x = sm.add_constant(X_test)
#x = X_test
R2 = r2_score(y_test, final.predict(x))

p = final.pvalues
t = final.tvalues
```



#### 逐步回归

```PYTHON
# encoding: utf-8
"""
功能：多元逐步回归 
描述：基于python实现多元逐步回归的功能 
作者：CHEN_C_W （草木陈） 
时间：2019年4月12日 
参考：https://www.cnblogs.com/lantingg/p/9535010.html 
""" 
import numpy as np 
import pandas as pd from sklearn.preprocessing 
import StandardScaler from statsmodels.formula 
import api as smf 
class FeatureSelection(): 
	def stepwise(self, df, response, intercept=True, normalize=False, criterion='bic', f_pvalue_enter=.05, p_value_enter=.05, direction='backward', show_step=True, criterion_enter=None, criterion_remove=None, max_iter=200, **kw):
		"""
		逐步回归。

		参数
		----
		df : dataframe
			分析用数据框，response为第一列。
		response : str
			回归分析相应变量。
		intercept : bool, 默认是True
			模型是否有截距项。
		criterion : str, 默认是'bic'
			逐步回归优化规则。
		f_pvalue_enter : float, 默认是.05
			当选择criterion=’ssr‘时，模型加入或移除变量的f_pvalue阈值。
		p_value_enter : float, 默认是.05
			当选择derection=’both‘时，移除变量的pvalue阈值。
		direction : str, 默认是'backward'
			逐步回归方向。
		show_step : bool, 默认是True
			是否显示逐步回归过程。
		criterion_enter : float, 默认是None
			当选择derection=’both‘或'forward'时，模型加入变量的相应的criterion阈值。
		criterion_remove : float, 默认是None
			当选择derection='backward'时，模型移除变量的相应的criterion阈值。
		max_iter : int, 默认是200
			模型最大迭代次数。
		"""
		criterion_list = ['bic', 'aic', 'ssr', 'rsquared', 'rsquared_adj']
		if criterion not in criterion_list:
			raise IOError('请输入正确的criterion, 必须是以下内容之一：', '\n', criterion_list)

		direction_list = ['backward', 'forward', 'both']
		if direction not in direction_list:
			raise IOError('请输入正确的direction, 必须是以下内容之一：', '\n', direction_list)

		# 默认p_enter参数
		p_enter = {'bic': 0.0, 'aic': 0.0, 'ssr': 0.05, 'rsquared': 0.05, 'rsquared_adj': -0.05}
		if criterion_enter:  # 如果函数中对p_remove相应key传参，则变更该参数
			p_enter[criterion] = criterion_enter

		# 默认p_remove参数
		p_remove = {'bic': 0.01, 'aic': 0.01, 'ssr': 0.1, 'rsquared': 0.05, 'rsquared_adj': -0.05}
		if criterion_remove:  # 如果函数中对p_remove相应key传参，则变更该参数
			p_remove[criterion] = criterion_remove

		if normalize:  # 如果需要标准化数据
			intercept = False  # 截距强制设置为0
			df_std = StandardScaler().fit_transform(df)
			df = pd.DataFrame(df_std, columns=df.columns, index=df.index)

		""" forward """
		if direction == 'forward':
			remaining = list(df.columns)  # 自变量集合
			remaining.remove(response)
			selected = []  # 初始化选入模型的变量列表
			# 初始化当前评分,最优新评分
			if intercept:  # 是否有截距
				formula = "{} ~ {} + 1".format(response, remaining[0])
			else:
				formula = "{} ~ {} - 1".format(response, remaining[0])

			result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
			current_score = eval('result.' + criterion)
			best_new_score = eval('result.' + criterion)

			if show_step:
				print('\nstepwise starting:\n')
			iter_times = 0
			# 当变量未剔除完，并且当前评分更新时进行循环
			while remaining and (current_score == best_new_score) and (iter_times < max_iter):
				scores_with_candidates = []  # 初始化变量以及其评分列表
				for candidate in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
					if intercept:  # 是否有截距
						formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
					else:
						formula = "{} ~ {} - 1".format(response, ' + '.join(selected + [candidate]))

					result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
					fvalue = result.fvalue
					f_pvalue = result.f_pvalue
					score = eval('result.' + criterion)
					scores_with_candidates.append((score, candidate, fvalue, f_pvalue))  # 记录此次循环的变量、评分列表

				if criterion == 'ssr':  # 这几个指标取最小值进行优化
					scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
					best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
					if ((current_score - best_new_score) > p_enter[criterion]) and (
							best_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
						remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
						selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
						current_score = best_new_score  # 更新当前评分
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
								  (best_candidate, best_new_score, best_new_fvalue, best_new_f_pvalue))
					elif (current_score - best_new_score) >= 0 and (
							best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
						selected.append(remaining[0])
						remaining.remove(remaining[0])
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
				elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
					scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
					best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
					if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
						remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
						selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
						current_score = best_new_score  # 更新当前评分
						# print(iter_times)
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif (current_score - best_new_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
						selected.append(remaining[0])
						remaining.remove(remaining[0])
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
				else:
					scores_with_candidates.sort()
					best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()
					if (best_new_score - current_score) > p_enter[criterion]:
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						print(iter_times, flush=True)
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif (best_new_score - current_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
						selected.append(remaining[0])
						remaining.remove(remaining[0])
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
				iter_times += 1

			if intercept:  # 是否有截距
				formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
			else:
				formula = "{} ~ {} - 1".format(response, ' + '.join(selected))

			self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合

			if show_step:  # 是否显示逐步回归过程
				print('\nLinear regression model:', '\n  ', self.stepwise_model.model.formula)
				print('\n', self.stepwise_model.summary())

		""" backward """
		if direction == 'backward':
			remaining, selected = set(df.columns), set(df.columns)  # 自变量集合
			remaining.remove(response)
			selected.remove(response)  # 初始化选入模型的变量列表
			# 初始化当前评分,最优新评分
			if intercept:  # 是否有截距
				formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
			else:
				formula = "{} ~ {} - 1".format(response, ' + '.join(selected))

			result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
			current_score = eval('result.' + criterion)
			worst_new_score = eval('result.' + criterion)

			if show_step:
				print('\nstepwise starting:\n')
			iter_times = 0
			# 当变量未剔除完，并且当前评分更新时进行循环
			while remaining and (current_score == worst_new_score) and (iter_times < max_iter):
				scores_with_eliminations = []  # 初始化变量以及其评分列表
				for elimination in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
					if intercept:  # 是否有截距
						formula = "{} ~ {} + 1".format(response, ' + '.join(selected - set(elimination)))
					else:
						formula = "{} ~ {} - 1".format(response, ' + '.join(selected - set(elimination)))

					result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
					fvalue = result.fvalue
					f_pvalue = result.f_pvalue
					score = eval('result.' + criterion)
					scores_with_eliminations.append((score, elimination, fvalue, f_pvalue))  # 记录此次循环的变量、评分列表

				if criterion == 'ssr':  # 这几个指标取最小值进行优化
					scores_with_eliminations.sort(reverse=False)  # 对评分列表进行降序排序
					worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()  # 提取最小分数及其对应变量
					if ((worst_new_score - current_score) < p_remove[criterion]) and (
							worst_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
						remaining.remove(worst_elimination)  # 从剩余未评分变量中剔除最新最优分对应的变量
						selected.remove(worst_elimination)  # 从已选变量列表中剔除最新最优分对应的变量
						current_score = worst_new_score  # 更新当前评分
						if show_step:  # 是否显示逐步回归过程
							print('Removing %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
								  (worst_elimination, worst_new_score, worst_new_fvalue, worst_new_f_pvalue))
				elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
					scores_with_eliminations.sort(reverse=False)  # 对评分列表进行降序排序
					worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()  # 提取最小分数及其对应变量
					if (worst_new_score - current_score) < p_remove[criterion]:  # 如果评分变动不显著
						remaining.remove(worst_elimination)  # 从剩余未评分变量中剔除最新最优分对应的变量
						selected.remove(worst_elimination)  # 从已选变量列表中剔除最新最优分对应的变量
						current_score = worst_new_score  # 更新当前评分
						if show_step:  # 是否显示逐步回归过程
							print('Removing %s, %s = %.3f' % (worst_elimination, criterion, worst_new_score))
				else:
					scores_with_eliminations.sort(reverse=True)
					worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()
					if (current_score - worst_new_score) < p_remove[criterion]:
						remaining.remove(worst_elimination)
						selected.remove(worst_elimination)
						current_score = worst_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Removing %s, %s = %.3f' % (worst_elimination, criterion, worst_new_score))
				iter_times += 1

			if intercept:  # 是否有截距
				formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
			else:
				formula = "{} ~ {} - 1".format(response, ' + '.join(selected))

			self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合

			if show_step:  # 是否显示逐步回归过程
				print('\nLinear regression model:', '\n  ', self.stepwise_model.model.formula)
				print('\n', self.stepwise_model.summary())

		""" both """
		if direction == 'both':
			remaining = list(df.columns)  # 自变量集合
			remaining.remove(response)
			selected = []  # 初始化选入模型的变量列表
			# 初始化当前评分,最优新评分
			if intercept:  # 是否有截距
				formula = "{} ~ {} + 1".format(response, remaining[0])
			else:
				formula = "{} ~ {} - 1".format(response, remaining[0])

			result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
			current_score = eval('result.' + criterion)
			best_new_score = eval('result.' + criterion)

			if show_step:
				print('\nstepwise starting:\n')
			# 当变量未剔除完，并且当前评分更新时进行循环
			iter_times = 0
			while remaining and (current_score == best_new_score) and (iter_times < max_iter):
				scores_with_candidates = []  # 初始化变量以及其评分列表
				for candidate in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
					if intercept:  # 是否有截距
						formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
					else:
						formula = "{} ~ {} - 1".format(response, ' + '.join(selected + [candidate]))

					result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
					fvalue = result.fvalue
					f_pvalue = result.f_pvalue
					score = eval('result.' + criterion)
					scores_with_candidates.append((score, candidate, fvalue, f_pvalue))  # 记录此次循环的变量、评分列表

				if criterion == 'ssr':  # 这几个指标取最小值进行优化
					scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
					best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()    # 提取最小分数及其对应变量
					if ((current_score - best_new_score) > p_enter[criterion]) and (
							best_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
						remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
						selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
						current_score = best_new_score  # 更新当前评分
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
								  (best_candidate, best_new_score, best_new_fvalue, best_new_f_pvalue))
					elif (current_score - best_new_score) >= 0 and (
							best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
						selected.append(remaining[0])
						remaining.remove(remaining[0])
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
				elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
					scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
					best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
					if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
						remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
						selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
						current_score = best_new_score  # 更新当前评分
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif (current_score - best_new_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
						selected.append(remaining[0])
						remaining.remove(remaining[0])
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
				else:
					scores_with_candidates.sort()
					best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()
					if (best_new_score - current_score) > p_enter[criterion]:  # 当评分差大于p_enter
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif (best_new_score - current_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
						remaining.remove(best_candidate)
						selected.append(best_candidate)
						current_score = best_new_score
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
					elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
						selected.append(remaining[0])
						remaining.remove(remaining[0])
						if show_step:  # 是否显示逐步回归过程
							print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))

				if intercept:  # 是否有截距
					formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
				else:
					formula = "{} ~ {} - 1".format(response, ' + '.join(selected))

				result = smf.ols(formula, df).fit()  # 最优模型拟合
				if iter_times >= 1:  # 当第二次循环时判断变量的pvalue是否达标
					if result.pvalues.max() > p_value_enter:
						var_removed = result.pvalues[result.pvalues == result.pvalues.max()].index[0]
						p_value_removed = result.pvalues[result.pvalues == result.pvalues.max()].values[0]
						selected.remove(result.pvalues[result.pvalues == result.pvalues.max()].index[0])
						if show_step:  # 是否显示逐步回归过程
							print('Removing %s, Pvalue = %.3f' % (var_removed, p_value_removed))
				iter_times += 1

			if intercept:  # 是否有截距
				formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
			else:
				formula = "{} ~ {} - 1".format(response, ' + '.join(selected))

			self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合
			if show_step:  # 是否显示逐步回归过程
				print('\nLinear regression model:', '\n  ', self.stepwise_model.model.formula)
				print('\n', self.stepwise_model.summary())
				# 最终模型选择的变量
		if intercept:
			self.stepwise_feat_selected_ = list(self.stepwise_model.params.index[1:])
		else:
			self.stepwise_feat_selected_ = list(self.stepwise_model.params.index)
		return self


def main(): 
	print('main方法进入了......')
	df = pd.DataFrame({'A': data[:, 0], 'B': data[:, 1], 'C': data[:, 2], 'D': data[:, 3]})
	print('测试data大小', df.shape)
	print(df.head())
	FeatureSelection().stepwise(df=df, response='A', max_iter=200, criterion='ssr')  # criterion='ssr'是为了移除不合适特征

```

