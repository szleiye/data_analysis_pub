[TOC]

# sklearn

## 数据处理

#### [对齐数据框 pandas.DataFrame.align](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.align.html)





#### [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)



### 类别变量转换

#### [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.get_dummies.html)

```PYTHON
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
```





#### [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder )

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

#### [多项式衍生 sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures )

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





