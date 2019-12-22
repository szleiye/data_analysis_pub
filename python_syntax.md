[TOC]



## python有用的包

pandas-profiling: 一键生成EDA报告



## Python 模块的标准文件写法

* **第1行、第2行：**

  标准注释，第1行注释可以让这个`hello.py`文件直接在Unix/Linux/Mac上运行，第2行注释表示`.py`文件本身使用标准`UTF-8`编码；

* **第4行：**

  是一个字符串，表示模块的文档注释，任何模块代码的第一个字符串都被视为模块的文档注释；

* **第6行：**

  使用`__author__`变量把作者写进去，这样当你公开源代码后别人就可以瞻仰你的大名；

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' a test module '
__author__ = 'Michael Liao'
import sys
def test():
    args = sys.argv
    if len(args)==1:
    	print('Hello, world!')
    elif len(args)==2:
    	print('Hello, %s!' % args[1])
    else:
    	print('Too many arguments!')
if __name__=='__main__':
	test()
```



## [python风格规范](<https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/>)

#### 缩进

* 不要用tab，用4个空格缩进代码



```PYTHON
Yes:   # Aligned with opening delimiter
       # 用开放的分割符对齐
       foo = long_function_name(var_one, var_two,
                                var_three, var_four)

       # Aligned with opening delimiter in a dictionary
       # 字典中不允许用悬挂缩进，一定要用分隔符后换行的方式
       foo = {
           long_dictionary_key: value1 +
                                value2,
           ...
       }

       # 4-space hanging indent; nothing on first line
       # 第一行保留空白，用悬挂缩进
       foo = long_function_name(
           var_one, var_two, var_three,
           var_four)

       # 4-space hanging indent in a dictionary
       foo = {
           long_dictionary_key:
               long_dictionary_value,
           ...
       }
```

#### 空行

* 函数或者类定义之间要空两行
* 方法定义, 类定义与第一个方法之间, 都应该空一行
* 函数或方法中, 某些地方要是你觉得合适, 就空一行.

#### 空格

* 当’=’用于指示关键字参数或默认参数值时, 不要在其两侧使用空格.

```PYTHON
Yes: def complex(real, imag=0.0): return magic(r=real, i=imag)
```

* 不要用空格来垂直对齐多行间的标记, 因为这会成为维护的负担(适用于:, #, =等):

```PYTHON
#Yes:
     foo = 1000  # comment
     long_name = 2  # comment that should not be aligned

     dictionary = {
         "foo": 1,
         "long_name": 2,
         }
        
#No:
     foo       = 1000  # comment
     long_name = 2     # comment that should not be aligned

     dictionary = {
         "foo"      : 1,
         "long_name": 2,
         }
```

#### 字符串

* 字符串要不都用单引号要不都用双引号
* 多重字符串要用`"""`， 除非你文中统一用单引号
* 即使参数都是字符串, 使用%操作符或者格式化方法格式化字符串. 不过也不能一概而论, 你需要在+和%之间好好判定.

``` PYTHON
# Yes: 
     x = a + b
     x = '%s, %s!' % (imperative, expletive)
     x = '{}, {}!'.format(imperative, expletive)
     x = 'name: %s; score: %d' % (name, n)
     x = 'name: {}; score: {}'.format(name, n)
    
# No: 
    x = '%s%s' % (a, b)  # use + in this case
    x = '{}{}'.format(a, b)  # use + in this case
    x = imperative + ', ' + expletive + '!'
    x = 'name: ' + name + '; score: ' + str(n)
```

#### TODO注释

* TODO + (名字:) + 要做啥

```PYTHON
# TODO(kl@gmail.com): Use a "*" here for string repetition.
# TODO(Zeke) Change this to use relations.
```

#### 命名

1. 所谓”内部(Internal)”表示仅模块内可用, 或者, 在类内是保护或私有的.
2. 用单下划线(_)开头表示模块变量或函数是protected的(使用from module import *时不会包含).
3. 用双下划线(__)开头的实例变量或方法表示类内私有.
4. 将相关的类和顶级函数放在同一个模块里. 不像Java, 没必要限制一个类一个模块.
5. 对类名使用大写字母开头的单词(如CapWords, 即Pascal风格), 但是模块名应该用小写加下划线的方式(如lower_with_under.py). 尽管已经有很多现存的模块使用类似于CapWords.py这样的命名, 但现在已经不鼓励这样做, 因为如果模块名碰巧和类名一致, 这会让人困扰.

| Type                       | Public             | Internal                                                     |
| -------------------------- | ------------------ | ------------------------------------------------------------ |
| Modules                    | lower_with_under   | _lower_with_under                                            |
| Packages                   | lower_with_under   |                                                              |
| Classes                    | CapWords           | _CapWords                                                    |
| Exceptions                 | CapWords           |                                                              |
| Functions                  | lower_with_under() | _lower_with_under()                                          |
| Global/Class Constants     | CAPS_WITH_UNDER    | _CAPS_WITH_UNDER                                             |
| Global/Class Variables     | lower_with_under   | _lower_with_under                                            |
| Instance Variables         | lower_with_under   | _lower_with_under (protected) or __lower_with_under (private) |
| Method Names               | lower_with_under() | _lower_with_under() (protected) or __lower_with_under() (private) |
| Function/Method Parameters | lower_with_under   |                                                              |
| Local Variables            | lower_with_under   |                                                              |

#### Main函数

* 要有main函数，防止脚本被导入时执行了其主功能

```PYTHON
def main():
      ...

if __name__ == '__main__':
    main()
```



## 数据加载

### 获取当前路径

```PYTHON
import os
#当前文件夹的绝对路径
path1=os.path.abspath('.')
#当前文件夹的上级文件夹的绝对路径
path1=os.path.abspath('..')
```

### 绝对路径和相对路径写法


我们常用`/`来表示相对路径，`\`来表示绝对路径，上面的路径里\\\是转义的意思，不懂的自行百度。

```PYTHON
open('aaa.txt')
open('/data/bbb.txt')
open('D:\\user\\ccc.txt')
```



## 数据写入

#### 写入到文件 open()+print()

```PYTHON
  data=open("D:\data.txt",'w+') 
  print('这是个测试',file=data)
  data.close()
```



## python 内置函数

#### 对可迭代对象进行排序 sorted() 

`iterable`: 可迭代对象。
`cmp` -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
`key` -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
`reverse` -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

```python
sorted(iterable[, cmp[, key[, reverse]]])
```
```python
>>>a = [5,7,6,3,4,1,2]
>>> b = sorted(a)       # 保留原列表
>>> a 
[5, 7, 6, 3, 4, 1, 2]
>>> b
[1, 2, 3, 4, 5, 6, 7]
 
>>> L=[('b',2),('a',1),('c',3),('d',4)]
>>> sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))   # 利用cmp函数
[('a', 1), ('b', 2), ('c', 3), ('d', 4)]
>>> sorted(L, key=lambda x:x[1])               # 利用key
[('a', 1), ('b', 2), ('c', 3), ('d', 4)]
 
 
>>> students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
>>> sorted(students, key=lambda s: s[2])            # 按年龄排序
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
 
>>> sorted(students, key=lambda s: s[2], reverse=True)       # 按降序
[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
```

#### 枚举遍历元素 enumerate

```PYTHON
for i,col in enumerate(['Pclass','SibSp','Parch']):
    ax = plt.subplot(2,3,i+1)
    sns.catplot(x=col, y="Survived", kind='bar',data=train_data, ax=ax)
    plt.close()
```

#### 检查元素类型 isinstance()

```python
isinstance(a, int)

isinstance(a, (int, float))
```

#### 判断变量是否为None

```python
a is None # 返回一个布尔值
```

## if 语句

#### 三元表达式

```PYTHON
'Non-negative' if x >=0 else 'Negative'
```

## 



## 其他

#### 转MD5 hashlib.md5() 

```PYTHON
def md5value(s):
    md5 = hashlib.md5()
    md5.update(s.encode("utf8"))
    return md5.hexdigest()
```

#### 判断文件是否存在 os.path.isfile() 

```PYTHON
if os.path.isfile(dt_fname):
    pass
else:
    data=pd.read_csv()
```



## Python 字符串

#### [格式化函数 str.format() ](<http://www.runoob.com/python/att-string-format.html>)

```PYTHON
# 不设置指定位置，按默认顺序
>>>"{} {}".format("hello", "world")    
'hello world'
 
# 设置指定位置
>>> "{0} {1}".format("hello", "world")  
'hello world'
 
# 设置指定位置
>>> "{1} {0} {1}".format("hello", "world")  
'world hello world'

# 
print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))
 
# 通过字典设置参数
site = {"name": "菜鸟教程", "url": "www.runoob.com"}
print("网站名：{name}, 地址 {url}".format(**site))
 
# 通过列表索引设置参数
my_list = ['菜鸟教程', 'www.runoob.com']
print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是必须的

# 传入一个对象
#!/usr/bin/python
# -*- coding: UTF-8 -*-
class AssignValue(object):
    def __init__(self, value):
        self.value = value
my_value = AssignValue(6)
print('value 为: {0.value}'.format(my_value))  # "0" 是可选的

# 同时指定名字和格式
print('Around {safe_pct:.2%} are safe loan and {risky_pct:.2%} are risky loan'.format(safe_pct=safe/total, risky_pct=risky/total))
```

#### 删除字符串中不想要的字符 re.sub() 

```PYTHON
import re
re.sub('[-/]', "", pbccStats.str_end_date)
```



数字格式化

| 数字       | 格式                                                         | 输出                                         | 描述                         |
| ---------- | ------------------------------------------------------------ | -------------------------------------------- | ---------------------------- |
| 3.1415926  | {:.2f}                                                       | 3.14                                         | 保留小数点后两位             |
| 3.1415926  | {:+.2f}                                                      | +3.14                                        | 带符号保留小数点后两位       |
| -1         | {:+.2f}                                                      | -1.00                                        | 带符号保留小数点后两位       |
| 2.71828    | {:.0f}                                                       | 3                                            | 不带小数                     |
| 5          | {:0>2d}                                                      | 05                                           | 数字补零 (填充左边, 宽度为2) |
| 5          | {:x<4d}                                                      | 5xxx                                         | 数字补x (填充右边, 宽度为4)  |
| 10         | {:x<4d}                                                      | 10xx                                         | 数字补x (填充右边, 宽度为4)  |
| 1000000    | {:,}                                                         | 1,000,000                                    | 以逗号分隔的数字格式         |
| 0.25       | {:.2%}                                                       | 25.00%                                       | 百分比格式                   |
| 1000000000 | {:.2e}                                                       | 1.00e+09                                     | 指数记法                     |
| 13         | {:10d}                                                       | 13                                           | 右对齐 (默认, 宽度为10)      |
| 13         | {:<10d}                                                      | 13                                           | 左对齐 (宽度为10)            |
| 13         | {:^10d}                                                      | 13                                           | 中间对齐 (宽度为10)          |
| 11         | `{:b}`.format(11) <br/>`{:d}`.format(11) <br/>`{:o}`.format(11) <br/>`{:x}`.format(11) <br/>`{:#x}`.format(11) <br/>`{:#X}`.format(11) | 1011 <br/>11 <br/>13 <br/>b<br/>0xb <br/>0XB | 进制                         |

^, <, > 分别是居中、左对齐、右对齐，后面带宽度， : 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。

\+ 表示在正数前显示 +，负数前显示 -；  （空格）表示在正数前加空格

b、d、o、x 分别是二进制、十进制、八进制、十六进制。

此外我们可以使用大括号 {} 来转义大括号，如下实例：

## 面向对象方法

### [更新对象绑定的方法](<https://www.jb51.net/article/109409.htm>)

```PYTHON
import types

class Person(object):
    pass

def say(self):
    print 'hello, world'
    
p = Person()
p.say = types.MethodType(say, p, Person)
p.say()
```

```PYTHON
# 首先，被修改的类的所有实例中的方法都会被更新，所以更新后的方法不仅仅存在于新创建的对象中，之前创建的所有对象都会拥有更新之后的方法，除非只是新增而不是覆盖掉原来的方法。
# 第二，你修改或者新增的方法应当是与对象绑定的，所以方法的第一个参数应当是被调用的对象（在这里就是类的实例self）
def newbark(self):
  print 'Wrooof!'
  
def howl(self):
  print 'Howl!'
  
# Replace an existing method
Dog.bark = newbark
  
# Add a new method
Dog.howl = howl
```

## 处理压缩文件

### 读取zip文件

```PYTHON
# https://blog.csdn.net/weixin_39724191/article/details/91350953
with zipfile.ZipFile('./exampleproject/_9_home_credit/data/home-credit-default-risk.zip', 'r') as z:
        f = io.TextIOWrapper(z.open('installments_payments.csv'))
        installments_payments = pd.read_csv(f)
```

### 返回压缩包内文件加名字

```PYTHON
print(azip.namelist())
```



## 列表

#### 一次加入多个元素 .extend()

```python
base_names = ['A'] + range(2,11) + ['J', 'K', 'Q']
cards=[]
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)
```

#### 添加元素到列表末尾 append()

```PYTHON
b_list.append('dwarf')
```

#### 添加元素到指定位置 insert()

```PYTHON
b_list.insert(1, 'red')
```

#### 移除并返回指定索引处的元素 pop()

```PYTHON
b_list.pop(2)
```

#### 按值删除元素 remove()

```PYTHON
b_list.remove('foo')
```

#### 判断列表中是否含有某个值

这个操作比dict和set要慢得多

```PYTHON
'dwarf' in b_list
```

#### 合并列表

列表的合并是一种费资源的操作，因为要创建新列表将所有对象复制过去

用extend将元素附加上去相对好很多

```PYTHON
# 用加号 + 连接
[4, None, 'foo'] + [7, 8, (2, 3)]

# 用extend方法一次性添加多个元素
x.extend([7, 8, (2, 3)])
```

## 循环

#### for 循环中提前进入下一次迭代 continue

即跳过代码块的剩余部分，直接进入下一次循环

```PYTHON
sequence = [1, 2, None, 4, None, 5]
total=0
for value in sequence:
    if value is None:
        continue
    total += value
```

#### for 循环中提前终止循环 break

break关键字用于使for循环完全退出

```python
sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value
    

```

#### 迭代生成整数xrange

range生成整数会预先产生所有值并存在列表中，xrange返回逐个产生整数的迭代器

```PYTHON
sum=0
for i in xrange(10000):
    # %是求模运算符
    if x % 3 ==0 or x % 5 == 0:
        sum += i
```

## 



## 字典 dict

#### 取指定键的值，没有则返回默认值 dict.get()

```PYTHON
dict.get(key, default=None)

# 下面这个例子可以用来清洗数据
f = lambda x: occ_mapping.get(x, x)  
fec.contbr_occupatio = .map(f) # 从字典里面取相应的映射值，否则取原值
```

#### 插入元素

```PYTHON
d1[7] = 'an integer'  # 会返回d1{7: 'an integer'}
```

#### 判断字典中是否存在某个键

```PYTHON
'b' in d1
```

删除指定键的值 del/pop

```python
d1[5] = 'some value'
d1['dummy'] = 'another value'

# del 删除元素
del d1[5]

# pop 删除元素并返回
ret = d1.pop('dummy')
```

#### 两个字典合并 update

```python
d1.update({'b': 'foo', 'c': 12})
```

#### 带有默认值的返回值 get

如果key不存在，则返回指定的默认值

```PYTHON
value = some_dict.get(key, default_value)
```

设置dict值时候给定默认值 setdefault()

```python
words = ['apple', 'bat', 'bar']
by_letter = {}

for word in words:
    letter = word[0]
	by_letter.setdefault(letter, []).append(word)  # 如果没有找到键，第一次先生成该键并对应值是空列表
```

#### defaultdict类

```PYTHON
from collections import defaultdict
by_letter = defaultdict(list)

for word in words:
    by_letter[word[0]].append(word)
```



## 内置序列函数

#### enumerate 逐个返回序列的(i, value) 元组

```PYTHON
for i, value in enumerate(collection):
    pass

# 将一个序列映射到其所在位置的字典
some_list = ['foo', 'bar', 'baz']
mapping = dict((v, i) for i, v in enumerate(some_list))
```

#### sorted 返回一个新的有序序列

```PYTHON
sorted(set('this is just some string')) # 返回唯一元素组成的有序序列
```

#### zip 将多个序列元素配对，产生新的元组列表

```PYTHON
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zip(seq1, seq2)  # 返回[('foo', 'one'), ('bar', 'two'), ('baz', 'three')]

# 常见用法是迭代多个序列，还可以配合enumerate一起使用
for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('%d: %s, %s' % (i, a, b))
```

#### zip 将序列解压还原

```PYTHON
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens')]
first_names, last_names = zip(*pitchers)
```

## 异常处理

#### 异常处理 try/except/else/finally

当try中的语句发生错误时，执行except中的代码块

```PYTHON
def attempt_float(x):
    try:
        return float(x)
    except:
        return x
    
# 在except后面加入异常类型，可以指定需要处理的异常类型
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x
    
# 用元组来捕获多个一场
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x
```

如果希望有一段代码无论try块代码成功与否都能被执行，使用finally即可以达到这个目的

如果希望某些代码只在try块成功时执行，则可以使用else

```PYTHON
f = open(path, 'w')

try:
    write_to_file(f)
except:
    print 'Failed'
else:
    print 'Succeeded'
finally:
    f.close()
```



## 



# Pandas

#### [pd.set_option()数据展示](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)

```PTYHON
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 展示200
pd.set_option('display.width', 200)

#
pd.options.display.max_columns = None
# 
pd.reset_option("display.max_rows")
```

## 读取数据

#### pd.read_csv() 读取csv文件

`engine`：可以选C或者pyhon，如果文件名是中文，选C会报错

```python
ratings_1 = pd.read_csv("../data/AI派_数据分析项目/ml-latest-small/ratings_1.csv",engine='python')
# 读数据的时候直接给列命名
ex1data2=pd.read_csv(".\data\ex1data2.txt", header=None, names=['Size', 'Bedrooms', 'Price'])
```

#### 从mysql中读取数据

[Pandas从MySQL中读取和保存数据](https://cloud.tencent.com/developer/news/255810)

#### pd.read_sql() 读取mysql数据

```PYTHON
import pandas as pd
import mysql.connector

connection_mddb = mysql.connector.connect(
    user='leiwy',
    password='eLison_6800',
    host='192.168.9.133',
    database='mddb',
    port=3306)

data = pd.read_sql("select distinct app_num,dd_date "
                   "from mddb.cr_loan_repay_detail where etl_date='2019-03-18' and dd_date>='2018-12-01'"
                   , con=connection_mddb)

```

#### pd.to_sql() 写入到mysql数据

```PYTHON
import pandas as pd
from sqlalchemy import create_engine
# 创建MySQL数据库连接
connection = create_engine('mysql+mysqlconnector://root:123456@127.0.0.1:3306/test')

data.to_sql("data",con=connection,if_exists='append')
```

#### 创建一个Dataframe

```PYTHON
# 通过Dict来创建
test_dict = {'id':[1,2,3,4,5,6],
             'name':['Alice','Bob','Cindy','Eric','Helen','Grace '],
             'math':[90,89,99,78,97,93],
             'english':[89,94,80,94,94,90]}

test=pd.DataFrame(test_dict)
```



## 数据描述

#### 查看dataframe情况

```python
chipo.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4622 entries, 0 to 4621
Data columns (total 5 columns):
order_id              4622 non-null int64
quantity              4622 non-null int64
item_name             4622 non-null object
choice_description    3376 non-null object
item_price            4622 non-null object
dtypes: int64(2), object(3)
memory usage: 180.6+ KB
```

#### 查看数据格式 .dtypes 

```PYTHON
df.dtypes
```

#### 查看序列变量类型 Series.dype

```PYTHON
'b'       boolean
'i'       (signed) integer
'u'       unsigned integer
'f'       floating-point
'c'       complex-floating point
'O'       (Python) objects
'S', 'a'  (byte-)string
'U'       Unicode
'V'       raw data (void)
```

#### 查看数据基本情况 .info()

```PYTHON
user_info.info()
```

#### 获取数据形状 .shape()

```python
user_info.shape
```

#### [统计列中每个值的个数 .value_counts() ](<http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html>)

```python
# Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
user_info.sex.value_counts()
```

####  去重看唯一值 .unique()/ .drop_duplicates() 

```PYTHON
a.unique()
a.drop_duplicates()
```

#### 查看列名 .columns

```PYTHON
df.columns
```

#### [查看重复项 DataFrame.duplicated ](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html)

**subset** : column label or sequence of labels, optional

Only consider certain columns for identifying duplicates, by default use all of the columns

**keep** : {‘first’, ‘last’, False}, default ‘first’

- `first` : Mark duplicates as `True` except for the first occurrence.
- `last` : Mark duplicates as `True` except for the last occurrence.
- False : Mark all duplicates as `True`.

```PYTHON
dt_pbcc_wide_tb_basic_info[dt_pbcc_wide_tb_basic_info.duplicated(subset=['rid'], keep=False)].head()
```



## 索引和列属性

#### 设置索引名称 .index.name 

```PYTHON
dt_loan.index.name = "index_rid"
```

#### 修改列名 .columns /.rename 

```PYTHON
# 暴力方法
a.columns = ['a','b','c']
```

```PYTHON
a.rename(columns={'A':'a', 'B':'b', 'C':'c'}, inplace = True)
frame3.columns.name='state'
```

#### 修改多层索引名称

```PYTHON
# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in bureau_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in bureau_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_%s_%s' % (var, stat))
```

#### 多重索引降级

```PYTHON
# 只保留1层索引
gp2.columns = gp1.columns.droplevel(0)

# 合并多层索引
gp3.columns = ["_".join(x) for x in gp3.columns.ravel()]
```

#### 重新索引 .reindex() 

```PYTHON
# Series的reindex会根据新索引排序，如果索引不在，，传入缺失值
obj.reindex(['a','b','c','d','e'], fill_value=0)

# DataFrmame可以修改行、列索引
frame.reindex(['a','b','c','d'])
frame.reindex(columns=['Texas','Utah','California'])

```

#### 

## 数据类型

[在Pandas中更改列的数据类型【方法总结】](https://www.cnblogs.com/xitingxie/p/8426340.html)

https://zgljl2012.com/pandas-shu-ju-lei-xing/

#### 创建dataframe时指定数据类型

```PYTHON
df = pd.DataFrame(a, dtype='float')  #示例1
df = pd.DataFrame(data=d, dtype=np.int8) #示例2
df = pd.read_csv("somefile.csv", dtype = {'column_name' : str})
```

#### .to_numeric() Series的数据类型转化

```PYTHON
 # 如果有无效值报错
 pd.to_numeric(s) 
 pd.to_numeric(s, errors='raise')
 # 将无效值强行转化为NaN
 pd.to_numeric(s, errors='coerce') 
 # 遇到无效值不处理，返回原序列
 pd.to_numeric(s, errors='ignore')
```

#### 处理多列的数据类型转化

```PYTHON
# 用apply
df[['col2','col3']] = df[['col2','col3']].apply(pd.to_numeric)

# 不知道那些列可以成功转成数字的话
df.apply(pd.to_numeric, errors='ignore')

# 用列表
list_1 = ['personal_house_loan_num', 'house_loan_num', 'other_loan_num',
          'outstand_loan_legal_org_num', 'outstand_loan_org_num', 'outstand_loan_num',
          'outstand_contract_value', 'outstand_bal', 'outstand_last_6m_repay_avg']
dt_loan[list_1] = dt_loan[list_1].astype(float, copy=True)
```

#### .infer_objects() 类型自动推断

```PYTHON
 df = df.infer_objects()
```

#### .astype() 变量类型强制转换

```PYTHON
df[['two', 'three']] = df[['two', 'three']].astype(float)
```

#### [pandas.to_datetime](http://pandas.pydata.org/pandas-docs/version/0.15/generated/pandas.to_datetime.html)

```PYTHON
# 批量转换成日期格式
dt_overdue_record_raw[list_3] = dt_overdue_record_raw[list_3].apply(pd.to_datetime, infer_datetime_format=True)
```

[pandas.Series.dt.strftime](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.strftime.html)

根据日期返回指定格式的字符串

```python
# ex1
df1['month'] = df1.ListingInfo.dt.strftime('%Y-%m')

# ex2
 rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"), periods=3, freq='s')
 rng.strftime('%B %d, %Y, %r')
```

#### pandas.read_csv(`parse_dates`=,  `infer_dateime`=True)读取数据格式时推断日期

```PYTHON
pd.read_csv(parse_dates=['report_time'],  infer_dateime=True, index_col=0)
```

#### [pandas.DataFrame.select_dtypes() 选择数据类型](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)

DataFrame.select_dtypes(*self*, *include=None*, *exclude=None*)

- To select all *numeric* types, use `np.number` or `'number'`
- To select strings you must use the `object` dtype, but note that this will return *all* object dtype columns
- See the [numpy dtype hierarchy](http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html)
- To select datetimes, use `np.datetime64`, `'datetime'` or `'datetime64'`
- To select timedeltas, use `np.timedelta64`, `'timedelta'` or `'timedelta64'`
- To select Pandas categorical dtypes, use `'category'`
- To select Pandas datetimetz dtypes, use `'datetimetz'` (new in 0.20.0) or `'datetime64[ns, tz]'`

## 数据操作

#### 取值映射 .map()

```PYTHON
dec.cand_nm[123456:123461].map(parties)  # parties是一个映射字典
```

## 数据操作-切片

### [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-query>)

.loc: selection by label

.iloc: selection by integer position

#### [用callable进行切片](<https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-callable>)

```PYTHON
df1.loc[lambda df: df.A > 0, :]
df1.loc[:, lambda df: ['A', 'B']]

# 可以进行链式选择，不用创建temp var
(bb.groupby(['year', 'team']).sum()
   ....:    .loc[lambda df: df.r > 100])
```

#### 切片

```PYTHON
# series的索引切片
obj[2:4]
obj[[1,3]]
obj[obj<2]
obj['b':'c']
obj[['b','a','d']]

```

#### [pandas.DataFrame.loc](<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html>) 用索引名切片

```PYTHON
df.loc['cobra':'viper', 'max_speed']
```

#### [pandas.DataFrame.query](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html#pandas-dataframe-query)

* 逻辑运算中要用`&` 和`|` 来表示`and`和`or`
* 给index命名后，也可以将index当作一个普通列来用，或者直接用index
* index和列名重名的话，优先用列名
* 用`not`或者`~`表示否定
* 大的daraframe情况下，回快一些

```PYTHON
# 取出dataframe中 “a<b<c” 的列
# 1. 传统方法
df[(df.a < df.b) & (df.b < df.c)]
# 2. 用query方法
df.query('(a < b) & (b < c)')

# 涉及到index的，可以用index的名字或者就用`index`来运算
df.query('index < b < c')

# 多重索引的时候，ilevel_0表示 the 0th level of the index
df.query('ilevel_0 == "red"')

# in 和 not in
df.query('a in b')  
df[df.a.isin(df.b)]
df.query('a not in b')
df[~df.a.isin(df.b)]

# 和一个列表比较 跟in/not in用法一样
df.query('b == ["a", "b", "c"]')
df[df.b.isin(["a", "b", "c"])]
```

## 数据操作-聚合和分组运算

#### 经过优化的聚合函数

```PYTHON
# count
# sum
# mean
# meadian
# std、var
# min、max
# prod
# first、last
```

#### 调用自己得聚合函数 DataFrame.GroupBy.agg  

```PYTHON
# 对一个变量分组统计多个数值
df.groupby('sex').agg({'tip': np.max, 'total_bill': np.sum})

# count(distinct **)
df.groupby('tip').agg({'sex': pd.Series.nunique})

train_data[['Title','Survived']].groupby('Title', as_index=False).agg({np.mean, np.sum, 'size','count'})
```

```PYTHON
# 用groupby统计好坏客户数
print(dt_modelSample.query("rid == rid").groupby(['enter_month']).apply(
    lambda grp:pd.Series({"Good":grp.query("ifBad == False").shape[0],
                          "Bad" : grp.query("ifBad == True").shape[0],
                          "BadRate": grp.query("ifBad == True").shape[0]/grp.shape[0]})
))
```

#### 传入多个聚合函数并指定列名

传入一个由(name, function)元组组成得列表，则各元组第一个元素会用作DataFrame的列名

```PYTHON
grouped_pct.agg([('foo', 'mean'),('bar', np.std)])
```

#### 对不同的列应用不同的函数

```python
grouped = tips.groupby(['sex', 'smoker'])

#1
grouped.agg({'tip': np.max, 'size': 'sum'})

#2
grouped.agg({'tip_pct':['min', 'max', 'mean', 'std'], 
            'size':'sum'})
```

#### 无索引方式返回聚合函数 `as_index=False`

```PYTHON
# 第一种是用as_index选项
tips.groupby(['sex', 'smoker'], as_index=False).mean()

# 第二种是调用reset_index
```

#### 禁止分组键 group_keys=False

分组键会跟原dataframe索引共同构成结果中的层次化索引，将group_keys=False传入groupby即可禁止该效果

```python
tips.groupby('smoker', group_keys=False).apply(top)
```

#### 变量分段分析 cut/qcut + groupby

```python
frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})

factor = pd.cut(frame.data1, 4)

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

grouped = frame.data2.groupby(factor)

grouped.apply(get_stats).unstack()
```

#### 分组填充缺失值 lambda + groupby

```python
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)
```



#### transform 将聚合结果应用到dataframe上

transform中传入的函数只能是：

1. 产生可以广播的标量值，如np.mean
2. 或者是一个相同大小的结果数组

```PYTHON
# 简便的方法
key = ['one', 'two', 'one', 'two', 'one']
people.groupby(key).mean()  # 分组求平均值的结果
people.groupby(key).transform(np.mean)  # 效果是将聚合后平均值的结果合并到原来的dataframe中

# 笨拙的方法，先groupby然后用merge合并

```

#### apply 对groupby调用自定义函数

apply将待处理的对象拆分成多个片段，然后对各片段调用传入的函数，最后将各片段组合。

```PYTHON
# 定义一个函数
def top(df, n=5, column='tip_pct'):
    return df.sort_index(by=column)[-n:]

# apply的函数能够接受其他参数或关键字，将关键字放在函数名后面
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
```



#### 分组保留前n行

```PYTHON
var_values.groupby('var_name').head(20)
```

#### 分组求均值

```python
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

#### lambda + groupby 隐函数实现特殊分组

```PYTHON
get_suit = laambda card: card[-1] # 取card字段的最后一个字母，用来等下分组
    
#1 
deck.groupby(get_suit).apply(draw, n=2)
#2 不把分组键作为索引
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)
```

#### 分组执行线性回归

```python
import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

by_year.apply(regress,'AAPL', ['SPX'])
```

#### 透视表 pivot_table

pivot table可以实现groupby的功能，并且添加分项合计

`values`：待聚合的列的名称，默认聚合所有数值列

`aggfun`: 传入自定义的聚合函数

`margin`:是否统计分项合计

`fill_value`: NA值填充

```PYTHON
# 默认是计算分组平均数
tips.pivot_table(['tip_pct', 'size'], rows=['sex', 'day'], cols='smoker', margins=True)

#如果要使用其他函数，可以传入给aggfunc选项
tips.pivot_table('tip_pct', rows=['sex', 'smoker'], cols='day', aggfunc=len, margins=True)
```

#### 交叉表 crosstab

计算分组频率

```PYTHON
pd.crosstab(data.Gender, data.Handedness, margins=True)

pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)
```

#### 分组算占比 

```PYTHON
bins = np.array([0, 1, 10, 100, 1000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins) # 设置分组标签

grouped = fec_mrbo.groupby(['cand_nm', labels]) # 设置分组

grouped.size().unstack(0) # 统计分组数量

bucker_sums = grouped.contb_receipt_amt.sum().unstack(0) # 统计分组金额
normed_sums = bucker_sums.div(bucket_sums.sum(axis=1), axis=0)
```



## 数据操作-其他

#### .drop 删除列或行

```PYTHON
# 删除行
obj.drop(['d','c'])
# 删除列
data.drop(['two','four'],axis=1)
```

#### 纵向合并数据集

[pandas.DataFrame.append](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html)

[pandas.concat](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html#pandas.concat) 

#### 检查去重后的id数量

```python
print(len(movies.movieId.unique()))
```

#### 返回两列id的交集

`np.intersect1d`

```python
np.intersect1d(movies.movieId.unique(), links.movieId.unique())
```

#### 连接表

 `merge`

[pandas.DataFrame.merge](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)

```PYTHON
DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
```

> **right** : Object to merge with.
> **how** : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
> **on** : label or list
> Column or index level names to join on. These must be found in both DataFrames. If on is None and not merging on indexes then this defaults to the intersection of the columns in both DataFrames.
> **left_on** : label or list, or array-like
> **right_on** : label or list, or array-like
> **left_index** : bool, default False
> **right_index** : bool, default False
> **sort** : bool, default False
> **suffixes** : tuple of (str, str), default (‘_x’, ‘_y’)
> **copy** : bool, default True
> **indicator** : bool or str, default False
> **validate** : str, optional
>
> - “one_to_one” or “1:1”: check if merge keys are unique in both left and right datasets.
> - “one_to_many” or “1:m”: check if merge keys are unique in left dataset.
> - “many_to_one” or “m:1”: check if merge keys are unique in right dataset.
> - “many_to_many” or “m:m”: allowed, but does not result in checks.

```PYTHON
# 多个键合并 传入一个列表
pd.merge(left, right, on=['key1,'key2], how='outer')

# 一边为多层级索引时，列表形式制定合并键的多列
pd.merge(left,right,left_on=['key1','key2'], right_index=True)
```



##### `join` 默认按索引合并

[pandas.DataFrame.join](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html)

`DataFrame.join`(*other*, *on=None*, *how='left'*, *lsuffix=''*, *rsuffix=''*, *sort=False*)

join的用法是默认用索引去连接，两个表至少有一个表是要用索引

```python
# 两个表都用索引
caller.set_index('key').join(other.set_index('key'))
# 一个表用索引
caller.join(other.set_index('key'), on='key')
```

##### `pd.concat` 轴向连接

[pandas.concat](http://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.concat.html)

```python
# 纵向连接3个series
pd.concat([s1,s2,s3])
pd.concat([df1,df2],ignore_index=True)
# 横向连接
pd.concat([s1,s2,s3],axis=1)

# 合并3个数据集，打上标记
pd.concat([s1,s2,s3], keys=['one','two','three'])

# 横向合并两个数据集，打上标记
pd.concat([df1,df2],axis=1,keys=['level1','level2'])
```



#### .replace() 替换数据

```PYTHON
dt_loan.first_loan_month.replace({'--': np.nan}, inplace=True)

# 结合字典完成映射
data = (
  pd.read_excel(path).fillna("")
    .sort_values(["code"])
    .assign(loc = lambda df: df["loc"].replace(dict_loc))
       )
```



#### .map()映射数据

```PYTHON
animal={'aa':'pig','bb':'dog','cc':'horse','dd':'dog','ee':'pig','ff':'dog','gg':'horse'}
#data['animal']=data['food'].map(animal)
data['animal']=data['food'].map(lambda x:animal[x])
--------------------- 
原文：https://blog.csdn.net/castinga3t/article/details/79015054 
```

#### [DataFrame.sort_values 排序](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)

```PYTHON
DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
```

**by** : str or list of str
**axis** : {0 or ‘index’, 1 or ‘columns’}, default 0
**ascending** : bool or list of bool, default True
**inplace** : bool, default False
**kind** : {‘quicksort’, ‘mergesort’, ‘heapsort’}, default ‘quicksort’
**na_position** : {‘first’, ‘last’}, default ‘last’

#### sort_index()按索引排序

```PYTHON
dt_query_info.sort_index(inplace=True, axis=1)
```



#### [pandas.DataFrame.replace](<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html>) 全局修改dataframe数据

```PYTHON
# overall replace
df.replace(to_replace='Female', value='Sansa', inplace=True)

# dict replace
df.replace({'sex': {'Female': 'Sansa', 'Male': 'Leone'}}, inplace=True)

# replace on where condition 
df.loc[df.sex == 'Male', 'sex'] = 'Leone'
```

#### 类似sql的case when 操作

```python
casewhen=lambda x: 1 if x>=0 and x<=3 else 2 if x>3 and x<=6 else 3
print(frame['Texas'].apply(casewhen))

```

#### 用B表数据更新A表的值

```PYTHON
def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return combined

combined = process_age()
```



##  字符操作

#### .split() 分割字符

> expand=True: 分割后生成多列数据

```python
movies.genres.str.split("|",expand=True).head()
```

#### .str.contains() 字符串是否包含特定文字

```python
temp_df1 = dt_loan_info[(dt_loan_info.guar_type == "组合（不含保证）担保")
             & (dt_loan_info.loan_useage.str.contains("个人经营性贷款|个人消费贷款"))
             & (dt_loan_info.status != "结清")].groupby(["rid", "issue_agency", "loan_useage", "guar_type"], as_index=False)["issue_limit"].agg({"com_amount_1":np.sum})

```

#### 字符串中是否含有列表中的元素

```PYTHON
def is_local(address, org, dict):
    # 要去除列表中空值，否则报错
    mark = any(city in address for city in list(dict[org][dict[org].notnull()]))
    return mark
```



## 日期计算

#### 计算时间间隔 relativedelta()

```PYTHON
# 月对月
LoanLDB = lambda df: (df["create_time"].dt.year - df["latestloanissuedate"].dt.year)*12 +
                             (df["create_time"].dt.month - df["latestloanissuedate"].dt.month)
# 日对日
Age = lambda df : df.apply(lambda r: relativedelta(r["create_time"],  r["birthday"]).years, axis =1)
```

 #### 计算时间间隔

```PYTHON
#方法一：
#先利用to_datetime转换为时间格式，tm列的数据形式为'yyyy-MM-dd HH:mm:ss'
df['tm_1'] = pd.to_datetime(df['tm_1'])
df['tm_2'] = pd.to_datetime(df['tm_2'])
#利用".dt.seconds"转换为秒，除以相对于的间隔数得到分钟、小时等
df['diff_time'] = (df['tm_1'] - df['tm_2']).dt.seconds/60
#利用round函数可进行四舍五入
df['diff_time'] = round(df['diff_time'])

#方法二，日期相减变为小时；变为天的话将h替换为D即可：
df['diff_time'] = (df['tm_1'] - df['tm_2']).values/np.timedelta64(1, 'h')

```



## 数据输出

#### dataframe输出为表格

```PYTHON
from IPython.display import display, HTML

# Assuming that dataframes df1 and df2 are already defined:
print "Dataframe 1:"
display(df1)
print "Dataframe 2:"
HTML(df2.to_html())
```

#### .to_csv 输出到文本格式

````python
# 按指定顺序排列
data.to_csv(sys.stdout,index=False,cols=['a','b','c'])
````

## 运算

#### [df.add() 相加时填充值](<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.add.html>)

`add`(*other*, *axis='columns'*, *level=None*, *fill_value=None*)

```PYTHON
# dataframe相加时：
# 1. 索引自动对齐
# 2. 任一边不存在的索引，传入NA值

df1.add(df2,fill_value=0)
```

#### df.sub() 减法

#### df.div()除法

#### df.mul() 乘法



## 排序

#### .sort_index() 根据索引排序

```python
obj.sort_index()
frame.sort_index()
frame.sort_index(axis=1, ascending=False)
```

#### Series.order() 对索引值排序

#### [DataFrame.sort_values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)

```python
DataFrame.sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
```



### [**用pandas实现sql功能**](https://blog.csdn.net/zeng_xiangt/article/details/81519535)

more at <https://codeburst.io/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e>

| **SQL**                                        | **Pandas**                              |
| ---------------------------------------------- | --------------------------------------- |
| select   * from airports                       | airports                                |
| select   * from airports limit 3               | airports.head(3)                        |
| select   id from airports where ident = ‘KLAX’ | airports[airports.ident   == ‘KLAX’].id |
| select   distinct type from airport            | airports.type.unique()                  |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   * from airports where iso_region = ‘US-CA’ and type = ‘seaplane_base’ | airports[(airports.iso_region   == ‘US-CA’) & (airports.type == ‘seaplane_base’)] |
| select   ident, name, municipality from airports where iso_region = ‘US-CA’ and type =   ‘large_airport’ | airports[(airports.iso_region   == ‘US-CA’) & (airports.type == ‘large_airport’)][[‘ident’, ‘name’,   ‘municipality’]] |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   * from airport_freq where airport_ident = ‘KLAX’ order by type | airport_freq[airport_freq.airport_ident == ‘KLAX’].sort_values(‘type’) |
| select   * from airport_freq where airport_ident = ‘KLAX’ order by type desc | airport_freq[airport_freq.airport_ident   == ‘KLAX’].sort_values(‘type’, ascending=False) |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   * from airports where type in (‘heliport’, ‘balloonport’) | airports[airports.type.isin([‘heliport’,   ‘balloonport’])]  |
| select   * from airports where type not in (‘heliport’, ‘balloonport’) | airports[~airports.type.isin([‘heliport’,   ‘balloonport’])] |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, type | airports.groupby([‘iso_country’,   ‘type’]).size()           |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, count(*) desc | airports.groupby([‘iso_country’,   ‘type’]).size().to_frame(‘size’).reset_index().sort_values([‘iso_country’,   ‘size’], ascending=[True, False]) |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, type | airports.groupby([‘iso_country’,   ‘type’]).size()           |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, count(*) desc | airports.groupby([‘iso_country’,   ‘type’]).size().to_frame(‘size’).reset_index().sort_values([‘iso_country’,   ‘size’], ascending=[True, False]) |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   type, count(*) from airports where iso_country = ‘US’ group by type having   count(*) > 1000 order by count(*) desc | airports[airports.iso_country   == ‘US’].groupby(‘type’).filter(lambda g: len(g) >   1000).groupby(‘type’).size().sort_values(ascending=False) |



| **SQL**                                                      | **Pandas**                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| select   iso_country from by_country order by size desc limit 10 | by_country.nlargest(10,   columns=’airport_count’)          |
| select   iso_country from by_country order by size desc limit 10 offset 10 | by_country.nlargest(20,   columns=’airport_count’).tail(10) |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   max(length_ft), min(length_ft), mean(length_ft), median(length_ft) from   runways | runways.agg({‘length_ft’:   [‘min’, ‘max’, ‘mean’, ‘median’]}) |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   airport_ident, type, description, frequency_mhz from airport_freq join   airports on airport_freq.airport_ref = airports.id where airports.ident =   ‘KLAX’ | airport_freq.merge(airports[airports.ident   == ‘KLAX’][[‘id’]], left_on=’airport_ref’, right_on=’id’,   how=’inner’)[[‘airport_ident’, ‘type’, ‘description’, ‘frequency_mhz’]] |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| select   name, municipality from airports where ident = ‘KLAX’ union all select name,   municipality from airports where ident = ‘KLGB’ | pd.concat([airports[airports.ident   == ‘KLAX’][[‘name’, ‘municipality’]], airports[airports.ident ==   ‘KLGB’][[‘name’, ‘municipality’]]]) |



| **SQL**                                              | **Pandas**                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| create   table heroes (id integer, name text);       | df1   = pd.DataFrame({‘id’: [1, 2], ‘name’: [‘Harry Potter’, ‘Ron Weasley’]}) |
| insert   into heroes values (1, ‘Harry Potter’);     | df2   = pd.DataFrame({‘id’: [3], ‘name’: [‘Hermione Granger’]}) |
| insert   into heroes values (2, ‘Ron Weasley’);      |                                                              |
| insert   into heroes values (3, ‘Hermione Granger’); | pd.concat([df1,   df2]).reset_index(drop=True)               |



| **SQL**                                                      | **Pandas**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| update airports set home_link = ‘<http://www.lawa.org/welcomelax.aspx>’ where ident == ‘KLAX’ | airports.loc[airports[‘ident’] == ‘KLAX’,   ‘home_link’] = ‘<http://www.lawa.org/welcomelax.aspx>’ |

| **SQL**                                    | **Pandas**                                               |
| ------------------------------------------ | -------------------------------------------------------- |
| delete   from lax_freq where type = ‘MISC’ | lax_freq   = lax_freq[lax_freq.type != ‘MISC’]           |
|                                            | lax_freq.drop(lax_freq[lax_freq.type   == ‘MISC’].index) |

# Numpy

### [np.insert(*arr***,** *obj***,** *values***,** *axis=None*) 矩阵插入新数据](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.insert.html)

**arr** : array_like

> Input array.

**obj** : int, slice or sequence of ints

> Object that defines the index or indices before which *values* is inserted.
>
> Support for multiple insertions when *obj* is a single scalar or a sequence with one element (similar to calling insert multiple times).

**values** : array_like

> Values to insert into *arr*. If the type of *values* is different from that of *arr*, *values* is converted to the type of *arr*.*values* should be shaped so that `arr[...,obj,...] = values` is legal.

**axis** : int, optional

> Axis along which to insert *values*. If *axis* is None then *arr* is flattened first.

```PYTHON
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)#增加全部为1的一列
```

### [多维矩阵降为一维  np.ravel(*a*, *order='C'*) ](https://github.com/numpy/numpy/blob/v1.16.1/numpy/core/fromnumeric.py#L1583-L1687)

返回原始矩阵降维后的视图，修改值影响原始矩阵



### np.argmax() 返回最大值的索引

### np.random.choice*(*a, size=None, replace=True, p=None) 

返回：从一维array a 或 int 数字a 中，以概率p随机选取大小为size的数据，replace表示是否重用元素，即抽取出来的数据是否放回原数组中，默认为true（抽取出来的数据有重复）

```PYTHON
#在（0，5）区间内生成含5个数据的一维数组
>>a = np.random.randint(0, 5, (5,))
>>print('a = ', a)
    a =  [2 1 2 1 3]
#在a数组中抽取6个数，replace为true
>>b = np.random.choice(a, 6)
>>print('b = ', b)
    b =  [1 1 2 2 1]
#replace为False时，size需小于等于len(a)
>>c = np.random.choice(a, 5, replace=False, p=None)
>>print('c = ', c)
    c =  [3 2 1 1 2]
#p是数组a中所有数出现的概率，总和为1
>>d = np.random.choice(a, 5, replace=False, p=[0.2, 0.3, 0.1, 0.3, 0.1])
>>print('d = ', d)
    d =  [1 3 2 1 2]
```

### 生成序列  np.arange

```PYTHON
np.arange(0.0, 5.0, 0.01)  #0-5，每隔0.01
```

### 增加多一维

```PYTHON
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
```





### 随机数模块random

#### 生成随机数 np.random.rand

随机数种子生成器

```PYTHON
rng=np.random.RandState(42)  # 其中42没什么含义,随机数种子生成器

rng.rand(50)  # 产生50个随机数，0-1之间

rng.rand(2, 3)  # 产生2*3的矩阵随机数  

np.random.uniform(1, 2, (5, 5))  # 随机数1-2之间，5行5列
```

生成正态分布随机数

```PYTHON
numpy.random.randn(d0, d1, …, dn)  # 是从标准正态分布中返回一个或多个样本值。 ----------有负数
```

# sklearn

## 决策树





## Python-实战问题

### [SettingWithCopyWarning警告](https://blog.csdn.net/dta0502/article/details/82288837)

对pandas赋值的时候有时候会出现SettingWithCopyWarning警告

```PYTHON
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame 
See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
```

**SettingWithCopyWarning出现的原因：**  

*链式赋值/Chained Assignment* 下，程序不确定返回的是引用还是拷贝

**链式：**对一个pandas表格，选取两次及以上其中结果

```PYTHON
# 先选取左右A<0.3的行，其次再从中选取B列，就是个链式操作
df1[df1.A < 0.3].B

# 再对以上赋值，就会报出warning，查看df1里面的值，会发现完全没有改变
df1[df1.A < 0.3].B = 1

# 根据提示使用loc函数
df1.loc[df1.A<0.3, 'B'] = 1

# 隐蔽的链式结构
df2 = df1.loc[df1.A<0.3]  # 这里实际上分成两步取值，就是链式的
df2.loc[1,'B'] = 2 
df2 = df1.loc[df1.A<0.3].copy() # 上面运行完后，df2值变了，df1值没变，即df2是copy不是引用，不过最好还是明确下
```

**总结：**

- 避免任何形式的链式赋值，有可能会报warning也有可能不会报。而且即使报了，可能有问题，也可能没问题。
- 如果需要用到多级选取，则用loc
- 如果需要用到拷贝，则直接加copy()函数

### [numpy陷阱](https://www.jianshu.com/p/a75e522d5839)

#### array取一列后，会退化成数组，需要reshape

```PYTHON
# array
In [98]: a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

In [99]: a
    
Out[99]:
        array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
        
In [101]: a[:, 0]
Out[101]: array([1, 4, 7])

In [102]: a[:, 0].shape
Out[102]: (3,) # a[:, 0] 过滤完应该是一个 3 x 1 的列向量，可是它退化为一个数组了
In [111]: a[:, 0].reshape(3, 1) # 需要进行reshape
```

#### matrix 用mask方式取值后向量行列方向会更改

```PYTHON
In [101]: A
Out[101]:
        matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

In [112]: y
Out[112]:
        matrix([[1],
                [0],
                [1]])

In [113]: A[:,0]
Out[113]:
        matrix([[1],
                [4],
                [7]])

In [102]: A[:, 0].shape
Out[102]: (3,1)

In [114]: A[:,0][y == 1]
Out[114]: matrix([[1, 7]])

In [114]: A[:,0][y == 1].shape  # 预期的输入结果应该是一个 2 x 1 的列向量，可是这里变成了 1 x 2 的行向量！
Out[114]: (1,2)
```


### 画多个子图

```PYTHON

```

# matplotlib画图

### 图存成PDF文件

```PYTHON
with PdfPages('multipage_pdf.pdf') as pdf:
    # f, axes = plt.subplots(6, 12, figsize=[4, 4])
    for i in range(0, 1):
        for j in range(0, 12):
            # plt.figure(figsize=(3, 3))
            var = '{}_queryNumsLast{}Month'.format(qry_reason[i], str(j+1))
            sns.distplot(dt_query_info[var],  color="b")
            plt.title(var)
            pdf.attach_note(var)
            pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
```

### 画各变量箱形图

```python
def f_EDA_boxplot(df, vartype='num',
                  separate=False, sep_tar='', plot_col=6,
                  figsize=[40, 40]):
    '''
    画出df中各个变量的分布图，用3sigma方法去除极值

    df: 输入的dataframe
    vartype: 数值型还是object
    separate: 是否根据二元变量分开展示分布
    sep_tar: 需要分开展示的二元变量，值需为1和0
    plot_col: 每行显示几个图
    figsize: 画布大小
    '''

    df_num = df.select_dtypes(include='number')
    obj_var = df.describe(exclude=['number', 'datetime']).T[lambda x: x.unique < 200].index
    df_obj = df[obj_var]

    # 确定subplot的大小
    fig = plt.figure(figsize=figsize)

    if vartype == 'num':
        var_num = df_num.shape[1]
        plot_row = math.floor(var_num/plot_col) + 1

        if separate is False:
            for i in range(0, var_num):
                ax1 = plt.subplot(plot_row, plot_col, i+1)
                var_name = df_num.columns[i]
                s1 = df_num[var_name]
                sns.boxplot(df_num[var_name], data=df_num, ax=ax1)
                
        else:
            for i in range(0, var_num):
                ax1 = plt.subplot(plot_row, plot_col, i+1)
                var_name = df_num.columns[i]
                s1 = df_num[var_name]
                sns.boxplot(y=var_name, x=sep_tar, data=df, ax=ax1)
        plt.close()
    return fig
```

### 画各变量分布图

```PYTHON
def f_EDA_displot(df, vartype='num',
                  separate=False, sep_tar='', plot_col=6,
                  figsize=[40, 40]):
    '''
    数值变量: 画出df中各个变量的分布图，用3sigma方法去除极值
    类别变量: 画出各个class的数量统计

    df: 输入的dataframe
    vartype: 数值型还是object
    # del_outlier: 是否去除极值
    separate: 是否根据二元变量分开展示分布
    sep_tar: 需要分开展示的二元变量，值需为1和0
    plot_col: 每行显示几个图
    figsize: 画布大小
    '''
    def three_sigma(Ser1):
        '''
        Ser1：表示传入DataFrame的某一列。
        '''
        rule = (Ser1.mean()-3*Ser1.std()<Ser1) & (Ser1.mean()+3*Ser1.std()> Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        outrange = Ser1.iloc[index]
        return outrange

    df_num = df.select_dtypes(include='number')
    obj_var = df.describe(exclude=['number', 'datetime']).T[lambda x: x.unique < 200].index
    df_obj = df[obj_var]
    df_date = df.select_dtypes(include='datetime64')

    # 确定subplot的大小
    fig = plt.figure(figsize=figsize)

    if vartype == 'num':
        var_num = df_num.shape[1]
        plot_row = math.floor(var_num/plot_col) + 1

        if separate is False:
            for i in range(0, var_num):
                ax1 = plt.subplot(plot_row, plot_col, i+1)
                var_name = df_num.columns[i]
                s1 = three_sigma(df_num[var_name])
                sns.distplot(s1,  color="b", ax=ax1, kde=True)
                plt.title(var_name)
                
        else:
            for i in range(0, var_num):
                ax1 = plt.subplot(plot_row, plot_col, i+1)
                var_name = df_num.columns[i]
                s1 = three_sigma(df_num[var_name])
                sns.distplot(s1[df[sep_tar] == 1], label='positive',  color="b", ax=ax1, kde=True)
                sns.distplot(s1[df[sep_tar] == 0], label='negative', color="r", ax=ax1, kde=True)
                # plt.title(var_name)
                ax1.legend()
        plt.close()

    if vartype == 'obj':
        var_num = df_obj.shape[1]
        plot_row = math.floor(var_num/plot_col) + 1
        df_obj['target'] = df_obj[sep_tar].astype('float')
        
        
        for i in range(0, var_num):
        
            ax1 = fig.add_subplot(plot_row, plot_col, i+1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-90)
            ax2 = ax1.twinx()
            ax1.grid(False)
            var_name = df_obj.columns[i]
            plot_ord = df_obj.groupby(var_name)['target'].mean().sort_values().index

            sns.catplot(x=var_name, kind="count", data=df, ax=ax1)
            sns.catplot(x=var_name, y=sep_tar, kind='point', data=df, ax=ax2, order=plot_ord,lw=1)

        plt.close()

    if vartype == 'date':
        var_num = df_date.shape[1]
        plot_row = math.floor(var_num/plot_col) + 1
        # df_date['target'] = df_date[sep_tar].astype('float')
            
        for i in range(0, var_num):
            ax1 = fig.add_subplot(plot_row, plot_col, i+1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-90)
            ax1.grid(False)
            var_name = df_date.columns[i]
            sns.catplot(x=sep_tar, y=var_name, kind="swarm", data=df)

        plt.close()

    return fig
```

### seaborn 画多个子图

```PYTHON
f, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.catplot(x="Sex", y="Survived", kind='bar',data=train_data, ax=axes[0])
sns.catplot(x="Ticket", y="Survived",kind='bar',  data=train_data, ax=axes[1]);
```

### 调整x轴坐标

```python
# 设置x轴标签大小
plt.tick_params(axis='x', labelsize=8)    

# 旋转x轴坐标
plt.xticks(rotation=-15)    
```

### 子图重叠

```PYTHON
plt.tight_layout()
```

### 双坐标轴

```PYTHON
ax2 = ax1.twinx()
```



