[TOC]



## python有用的包

pandas-profiling: 一键生成EDA报告





## [python风格规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)

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

6. ##### 用复数形式命名list

7. ##### 用显式名称命名字dict

| Type                       | Public             | Internal                                 |
| -------------------------- | ------------------ | ---------------------------------------- |
| Modules                    | lower_with_under   | _lower_with_under                        |
| Packages                   | lower_with_under   |                                          |
| Classes                    | CapWords           | _CapWords                                |
| Exceptions                 | CapWords           |                                          |
| Functions                  | lower_with_under() | _lower_with_under()                      |
| Global/Class Constants     | CAPS_WITH_UNDER    | _CAPS_WITH_UNDER                         |
| Global/Class Variables     | lower_with_under   | _lower_with_under                        |
| Instance Variables         | lower_with_under   | _lower_with_under (protected) or __lower_with_under (private) |
| Method Names               | lower_with_under() | _lower_with_under() (protected) or __lower_with_under() (private) |
| Function/Method Parameters | lower_with_under   |                                          |
| Local Variables            | lower_with_under   |                                          |

#### 函数

1. 函数内等号两边不要用空格

   ```PYTHON
   def function_name(parameter_0, parameter_1='default value')
   function_name(value_0, parameter_1='value')
   ```




#### Main函数

* 要有main函数，防止脚本被导入时执行了其主功能

```PYTHON
def main():
      ...

if __name__ == '__main__':
    main()
```



#### 模块的标准文件写法

- **第1行、第2行：**

  标准注释，第1行注释可以让这个`hello.py`文件直接在Unix/Linux/Mac上运行，第2行注释表示`.py`文件本身使用标准`UTF-8`编码；

- **第4行：**

  是一个字符串，表示模块的文档注释，任何模块代码的第一个字符串都被视为模块的文档注释；

- **第6行：**

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



#### 函数说明写法

```PYTHON
# 写法1
def _validate_names(names):
    """
    Check if the `names` parameter contains duplicates.
    If duplicates are found, we issue a warning before returning.
    
    Parameters
    ----------
    names : array-like or None
        An array containing a list of the names used for the output DataFrame.
        
    Returns
    -------
    names : array-like or None
        The original `names` parameter.
    """

    if names is not None:
        if len(names) != len(set(names)):
            raise ValueError("Duplicate names are not allowed.")
    return names

# 写法2
def some_func(myParam1, myParam2):
    """
    This is where you add what your function does
	@param myParam1: (int) An integer for input
    @param myParam2: (int) An integer for input
    @return: (str) A string meesage
    """
```



## python 基础

#### [python中带下划线的变量和函数的意义](https://www.cnblogs.com/elie/p/5902995.html)

**变量**

| 类型                          | 含义                                       |
| --------------------------- | ---------------------------------------- |
| `_xxx` 单下划线开头               | 警告说明这是一个私有变量，原则上外部类不要去访问它                |
| `__xxx ` 双下划线开头             | 表示的是私有类型(private)的变量。只能是允许这个类本身进行访问了, 用于命名一个类属性（类变量）。<br />调用时名字被改变（在类Student内部，`__name`变成`_Student__name`,如 `self._Student__name`) |
| `__xxx__`，以双下划线开头，并且以双下划线结尾 | 是内置变量，内置变量是可以直接访问的，不是 private 变量，如`__init__`，`__import__`或是`__file__`。所以，不要自己定义这类变量。 |
| `xxx_`，单下划线结尾的              | 一般只是为了避免与 Python 关键字的命名冲突。               |
| `USER_CONSTANT`，大写加下划线      | 对于不会发生改变的全局变量，使用大写加下划线。                  |

**函数和方法**

| 类型                                    | 含义                                       |
| ------------------------------------- | ---------------------------------------- |
| `def _secrete(self):`<br />一个前导下划线    | 私有方法，并不是真正的私有访问权限。<br />同时也应该注意一般函数不要使用两个前导下划线(当遇到两个前导下划线时，Python 的名称改编特性将发挥作用) |
| `def __add__(self, other):`<br />特殊方法 | 这种风格只应用于特殊函数，比如操作符重载等。                   |
| 函数参数                                  | 小写和下划线，缺省值等号两边无空格<br />def connect(self, user=None):    <br />    self._user = user<br /> |



#### 导入模块和特定函数

```PYTHON
import module_name
import module_name as mn
from module_name import function_name
from module_name import function_name as fn

```











## 基础设置

#### 忽略warning

```PYTHON
warnings.simplefilter('ignoree')
```



## 数据加载

#### 获取当前路径

```PYTHON
import os
#当前文件夹的绝对路径
path1=os.path.abspath('.')
#当前文件夹的上级文件夹的绝对路径
path1=os.path.abspath('..')
```



#### 绝对路径和相对路径写法


我们常用`/`来表示相对路径，`\`来表示绝对路径，上面的路径里\\\是转义的意思，不懂的自行百度。

```PYTHON
open('aaa.txt')
open('/data/bbb.txt')
open('D:\\user\\ccc.txt')
```



#### 判断文件是否存在 os.path.isfile() 

```PYTHON
if os.path.isfile(dt_fname):
    pass
else:
    data=pd.read_csv()
```



#### 看当前文件夹下有什么文件 os.listdir()

```PYTHON
# 看文件夹内有什么数据
os.listdir('./data/home-credit-default-risk/')
```





#### 逐行读取

```PYTHON
filename = 'pd_digits.txt'

with open(filename) as file_object:
    for line in file_object:
        print(line.rstrip())
```



#### 将文件内容存成列表稍后处理 readlines()

```python
filename = 'pd_digits.txt'

with open(filename) as file_object:
    lines = file_object.readlines()  # 存成列表，后面调用
    
for line in lines:  # 调用之前储存的文件内容
    print(line.rstrip())
```

####  

#### 读取整个文件 with open()

open(): 表示打开文件

with: 在不需要访问文件后将其关闭，替代close() 

```PYTHON
with open('pi_digits.txt') as file_object:
    contents = file_object.read()
    print(contents)
```



## 数据写入

#### 写入空文件

`r`: 只读模型，默认

`r+`: 可以读取和写入

`w`: 写入模式，只能写入字符串

`a`: 附加模式

```PYTHON
filename = 'programming.txt'

with open(filename, 'w') as file_object:
    file_object.write("I love programming.")
```





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



#### [zip函数](https://www.runoob.com/python/python-func-zip.html )

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

```PYTHON
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```



#### [生成参数所有笛卡尔积itertools.product()](https://docs.python.org/zh-cn/3/library/itertools.html#itertools.product)

```PYTHON
    # https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
    keys, values = zip(*param_grid.items())  # 解压，将字段元素返回成 key 和 item 两个元组
    
    i = 0
    
    # Iterate through every possible combination of hyperparameters
    for v in itertools.product(*values):  # 生成所有超参的笛卡尔积
        
        # Create a hyperparameter dictionary
        hyperparameters = dict(zip(keys, v))  # 对每一组超参，重新打包成一个字典
        
        # Set the subsample ratio accounting for boosting type
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        
        # Evalute the hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
        
        i += 1
        
        # Normally would not limit iterations
        if i > MAX_EVALS:
            break
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

#### 静态方法  staticmethod

 staticmethod用于修饰类中的方法,使其可以在不创建类实例的情况下调用方法。当然，也可以像一般的方法一样用实例调用该方法。

静态方法不可以引用类中的属性或方法，其参数列表也不需要约定的默认参数self。

静态方法就是类对外部函数的封装，有助于优化代码结构和提高程序的可读性。

```PYTHON
class Time():
    def __init__(self,sec):
        self.sec = sec
    #声明一个静态方法
    @staticmethod
    def sec_minutes(s1,s2):
        #返回两个时间差
        return abs(s1-s2)

t = Time(10)
#分别使用类名调用和使用实例调用静态方法
print(Time.sec_minutes(10,5),t.sec_minutes(t.sec,5))
#结果为5 5
```

#### [将函数应用于列表中 map()](https://blog.csdn.net/qq_41800366/article/details/87881144)

 Python中的map() 会根据提供的函数对指定序列做映射。 返回一个迭代器

```PYTHON
A = [1, -1, 2, -3]
B = list(map(abs, A))
print(B)  # 结果 [1, 1, 2, 3]


B = map(abs, A)
C = [item for item in B]
print(B)  # 结果 <map object at 0x0000024B202476D8>
print(C)  # 结果 [1, 1, 2, 3]

```



#### 查看变量大小 sys.getsizeof()

```python
import sys

def return_size(df):
    '''Return size of dataframe in gigabytes'''
    return round(sys.getsizeof(df) / 1e9, 2)
```



#### 随[机取数 random.sample()](https://docs.python.org/3/library/random.html )

```PYTHON
random.seed(50)

# Randomly sample from dictionary
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}  # 从参数列表随机选1个random.sample(v, 1)[0]，[0]是要去除列表形式
# Deal with subsample ratio
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

random_params
```





## if 语句

#### 三元表达式

```PYTHON
'Non-negative' if x >=0 else 'Negative'
```

## 其他

#### 转MD5 hashlib.md5() 

```PYTHON
def md5value(s):
    md5 = hashlib.md5()
    md5.update(s.encode("utf8"))
    return md5.hexdigest()
```



#### [StringIO]( https://blog.csdn.net/lucyxu107/article/details/82728266 )

 数据读写不一定是文件，也可以在内存中读写。StringIO就是在内存中读写str 

```python
>>> from io import StringIO
>>> f = StringIO()
>>> f.write('hello')
5
>>> f.write(' ')
1
>>> f.write('world!')
6
>>> print(f.getvalue())

```



#### [动态生成变量 locals()](https://blog.csdn.net/u013061183/article/details/78015673 )

locals()是一个字典，键是变量名，值是对应的变量的值

可以通过新增键的形式来动态创建变量

```PYTHON
createVar = locals()
listTemp = range(1,10)
for i,s in enumerate(listTemp):
    createVar['a'+i] = s
print a1,a2,a3
```



#### [在class中动态生成变量  setattr( )](https://blog.csdn.net/u013061183/article/details/78015673 )

setattr( object, name, value)

>  This is the counterpart of `getattr()`. The arguments are an object, a string and an arbitrary value. The string may name an existing
> attribute or a new attribute. The function assigns the value to the attribute, provided the object allows it. 
>
> For example : `setattr(x,'foobar', 123)` is equivalent to `x.foobar = 123`. 

```PYTHON
class test(object) :
    def __init__(self):
        dic={'a':'aa','b':'bb'}
        for i in dic.keys() :
            setattr(self,i,dic[i]) #第一个参数是对象，这里的self其实就是test.第二个参数是变量名，第三个是变量值
        print(self.a)
        print(self.b)
t=test()

# 打印结果是：aa, bb
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

# 同时指定名字和格式, {:.2%}
print('Around {safe_pct:.2%} are safe loan and {risky_pct:.2%} are risky loan'.format(safe_pct=safe/total, risky_pct=risky/total))
```



#### 用f-Strings 让字符串更可读

````PYTHON
first = "Sam"
last = "Miller"
middle "Douglas"
f"You are a fantastic programmer, {first} {middle} {last}"
````



#### 删除字符串中不想要的字符 re.sub() 

```PYTHON
import re
re.sub('[-/]', "", pbccStats.str_end_date)
```

#### [小写字母转大写字母 upper()](https://www.runoob.com/python/att-string-upper.html )

```PYTHON
str = "this is string example....wow!!!";

print "str.upper() : ", str.upper()
```



#### 数字格式化

| 数字         | 格式                                       | 输出                                       | 描述                |
| ---------- | ---------------------------------------- | ---------------------------------------- | ----------------- |
| 3.1415926  | {:.2f}                                   | 3.14                                     | 保留小数点后两位          |
| 3.1415926  | {:+.2f}                                  | +3.14                                    | 带符号保留小数点后两位       |
| -1         | {:+.2f}                                  | -1.00                                    | 带符号保留小数点后两位       |
| 2.71828    | {:.0f}                                   | 3                                        | 不带小数              |
| 5          | {:0>2d}                                  | 05                                       | 数字补零 (填充左边, 宽度为2) |
| 5          | {:x<4d}                                  | 5xxx                                     | 数字补x (填充右边, 宽度为4) |
| 10         | {:x<4d}                                  | 10xx                                     | 数字补x (填充右边, 宽度为4) |
| 1000000    | {:,}                                     | 1,000,000                                | 以逗号分隔的数字格式        |
| 0.25       | {:.2%}                                   | 25.00%                                   | 百分比格式             |
| 1000000000 | {:.2e}                                   | 1.00e+09                                 | 指数记法              |
| 13         | {:10d}                                   | 13                                       | 右对齐 (默认, 宽度为10)   |
| 13         | {:<10d}                                  | 13                                       | 左对齐 (宽度为10)       |
| 13         | {:^10d}                                  | 13                                       | 中间对齐 (宽度为10)      |
| 11         | `{:b}`.format(11) <br/>`{:d}`.format(11) <br/>`{:o}`.format(11) <br/>`{:x}`.format(11) <br/>`{:#x}`.format(11) <br/>`{:#X}`.format(11) | 1011 <br/>11 <br/>13 <br/>b<br/>0xb <br/>0XB | 进制                |

^, <, > 分别是居中、左对齐、右对齐，后面带宽度， : 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。

\+ 表示在正数前显示 +，负数前显示 -；  （空格）表示在正数前加空格

b、d、o、x 分别是二进制、十进制、八进制、十六进制。

此外我们可以使用大括号 {} 来转义大括号，如下实例：



#### 换行符和制表符  /n  /t



#### 接受用户输入 input()

如果需要接受数值型输入，用 int() 对接受到的变量进行转换

```PYTHON
name = input(prompt)  # prompt 中储存提示语句，输入变量存在name中
```



## 面向对象方法

#### [更新对象绑定的方法](<https://www.jb51.net/article/109409.htm>)

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

#### 读取zip文件

```PYTHON
# https://blog.csdn.net/weixin_39724191/article/details/91350953
with zipfile.ZipFile('./exampleproject/_9_home_credit/data/home-credit-default-risk.zip', 'r') as z:
        f = io.TextIOWrapper(z.open('installments_payments.csv'))
        installments_payments = pd.read_csv(f)
```

#### 返回压缩包内文件加名字

```PYTHON
print(azip.namelist())
```



## 列表 list



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



#### 列表切片

```PYTHON
# 正常取
players[0:3]

# 从列表开头开始取
players[:4]

# 取到列表末尾
players[2:]

# 取最后三个元素
players[-3:]
```



#### 列表永久排序 sort()

```PYTHON
cars.sort(reverser=True)
```



#### 列表临时排序 sorted()

```PYTHON
sorted(cars)

sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
```





#### 列表临时排序 sorted()



#### 复制列表

```PYTHON
my_foods = ['pizza'. 'falafel', 'carrot cake']
friend_foods = my_foods[:]  # 复制列表
```



#### 生成数字列表 list + range

```python
list(range(1, 11))
```



#### 判断列表中是否含有某个值

这个操作比dict和set要慢得多

```PYTHON
'dwarf' in b_list
```



#### 合并列表 extend/+

列表的合并是一种费资源的操作，因为要创建新列表将所有对象复制过去

用extend将元素附加上去相对好很多

```PYTHON
# 用加号 + 连接
[4, None, 'foo'] + [7, 8, (2, 3)]

# 用extend方法一次性添加多个元素
x.extend([7, 8, (2, 3)])
```



#### 列表、集合以及字典的推导式

```python
[x.upper() for x in strings if len(x) > 2]

# 列表推导式结合函数
def process(item):
   item = item * 2
   item = item / 5
   item = item + 12
   return item# Done with the map function

new_list = map(process, items)# With a list comprehension, better readability!
new_list = [process(item) for item in items]

# 列表推导式加条件筛选
filtered = [item for item in items if item > 5]
```







## 字典 dict

#### 遍历字典的键值对 dict.items()

遍历输出结果不一定按原顺序，字典只关心键值对，不关心排序

```PYTHON
for key, value in user_0.items():
    print("\nKey: ", key)
    print("Value: ", value)    
```



#### 遍历字典中的键 dict.keys()

dict.keys() 返回字典所有键值的一个列表

```python
for name in favorite_languages.keys():
    print(name.title())
```



#### 遍历字典中的值 dict.values()

返回一个值列表

```PYTHON
for language in set(favorite_languages.values()):  # 用set去重
    print(language.title())
```



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

#### 列表中嵌套字典

例子：列表储存一群外星人，嵌套字典代表每个外星人的信息

```PYTHON

```



#### 字典中嵌套列表

```PYTHON
favorite_languages = {
    'jen': ['python', 'ruby'],
    'sarah': ['c'],
    'edward': ['ruby', 'go'],
    'phil': ['python', 'haskell']
}

# 双重循环输出列表内容
for name, languages in favorite_languages.items():
    pass
	for language in languages:
        pass
```



####  字典嵌套字典





## 集合 set

#### 创建一个set

set是由唯一元素组成的无序集

```PYTHON
# 创建方式1
set([2, 2, 2, 1, 3, 3])

# 创建方式2
{2, 2, 2, 1, 3, 3}
```

#### set的并、交、差及对称差

```PYTHON
a | b  # 并
a & b  # 交
a - b  # 差
a ^ b  # 堆成差(异或)  取出两遍不相交的部分
```

#### 判断一个集合是否另一个集合的子集

```PYTHON
a_set = {1, 2, 3, 4, 5, }

{1, 2, 3}.issubset(a_set)  # 新集合是否包含于原集合
```

#### 判断一个集合是否包含另一个集合

```python
a_set.issuperset({1, 2, 3})
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

 #### 循环中添加进度提示信息  tqdm(iterator) 

```PYTHON
# tqdm(list)方法可以传入任意一种list,比如数组
for i in tqdm(range(1000)):  
     #do something
     pass  

for char in tqdm(["a", "b", "c", "d"]):
    #do something
    pass

# 在for循环外部初始化tqdm,可以打印其他信息
bar = tqdm(["a", "b", "c", "d"])
for char in bar:
    pbar.set_description("Processing %s" % char)
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

## 函数

#### 形参、实参

形参：函数定义中的变量名

实参：传递给参数的参数值



#### [传递任意数量位置实参，关键字实参 *args、 **kwargs](https://www.liaoxuefeng.com/wiki/1016959663602400/1017261630425888)

位置参数(`*args`)：被打包成元组

```PYTHON
def calc(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum

>>> calc([1, 2, 3])
14
>>> calc((1, 3, 5, 7))
84
```



关键字参数(`**kwargs`)：通常用于指定默认值或可选参数，**必须位于位置参数之后**，被打包成字典

```PYTHON
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)

# 用法1
>>> person('Adam', 45, gender='M', job='Engineer')
name: Adam age: 45 other: {'gender': 'M', 'job': 'Engineer'}  
            
# 用法2： 将字典传入关键字参数
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, **extra)
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
```



#### 命名关键字参数，限制接受关键字参数的范围

和关键字参数`**kw`不同，命名关键字参数需要一个特殊分隔符`*`，`*`后面的参数被视为命名关键字参数。 

 如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符`*`了。

```PYTHON
# 1
def person(name, age, *, city, job):
    print(name, age, city, job)
    
>>> person('Jack', 24, city='Beijing', job='Engineer')  # 调用的时候一定要写参数名
Jack 24 Beijing Engineer

# 2
def person(name, age, *args, city, job):
    print(name, age, args, city, job)
```





#### 传递列表给函数

* 传递列表给参数时，函数直接对原列表进行修改
* 如果禁止函数修改列表，传入列表副本`list[:]`即可，但不建议这么做，耗费资源




#### 定义函数

```PYTHON
def g(x, y, z=1):
    return (x+y)/z
```

#### 函数返回多个值

```PYTHON
# 以元组形式返回
def f():
    a = 5
    b = 6
    c = 7
    return a, b, c

# 以字典形式返回
def f():
    a = 5
    b = 6
    c = 7
    return {'a':a, 'b':b, 'c':c}
```

#### 函数也可以作为对象

```PYTHON
def remove_punctuation(value):
    return re.sub('[!#?]', '', value)  # 对字符做替换处理

clean_ops = [str.strip, remove_punctuation, str.title]  # 要使用的函数列表

# 将函数作为对象形式调用，提高clean_strings复用性
def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:  # 调用op函数列表里面的函数
            value = function(value)
        result.append(value)
    return result
```



### 类

#### 创建类 

* `__init__()`初始化方法，实例化时候自动运行
* 带有前缀 `self` 的变量可通过实例访问，被称为`属性`
* 

```PYTHON
class Dog():
    """
    类的说明
    """
    
    # init初始化方法，实例化时候自动运行
    def __init__(self, name, age):  
        """
        初始化属性name和age
        """
        self.name = name  # 类的属性
        self.age = age
        
    # 类的方法1
    def sit(self):
        """
        模拟小狗被命令时蹲下
        """
        print(self.name.title() + "is now sitting.")
    
    # 类的方法2
    def roll_over(self):
        """
        模拟小狗被命令时打滚
        """
        print(self.name.title() + "rolled over!")
```



#### 继承

```PYTHON

# 父类要在子类之前
class Car():
    pass

class ElectricCar(Car):  # 定义子类，括号内指定父类名称
    """
    电动汽车的独特之处
    """
    
    def __init__(self, make, modelm year):  # init方法要提供创建父类实例所需信息
        """
        初始化父类的属性
        """
        super().__init__(make, model, year)  # 让实例包含父类所有属性
        
my_tesla = ElectriCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())
```

#### 重写父类的方法

父类中如果一些方法不合适，可以在子类中直接重写。子类中写一个名称一样的方法即可



#### 将实例用作属性

如果类属性和方法太复杂，可将类的一部分作为一个实例的类提取出来，拆分成协同工作的小类。

例如，可将ElectriCar类中关于电池的部分提取出来，放到Battery类中

```PYTHON
class Battery():
    -snip-

class ElectricCar(Car):
    """
    """
    
    def __init__(self)
    	self.battery = Battery
```





## 异常处理

#### 异常处理 try/except/else/finally

 <img  src="assets/try_except_else_finally.png" alt="img" align="center"  style="zoom: 25%; " />

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

#### raise 触发异常

 <img src="assets/raise.png" alt="img" style="zoom:33%;" /> 

```PYTHON
# 如果 x 大于 5 就触发异常
x = 10
if x > 5:
    raise Exception('x 不能大于 5。x 的值为: {}'.format(x))
```

#### assert 触发异常

 Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。 

<img src="assets/assert.png" alt="img" style="zoom: 33%;" align='left'/>

 

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

`dtype`: 输入一个字典制定列的数据类型，如 {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}. 配合`na_values `设定空值

`na_values` : 输入scalar, str, list-like, or dict 。指定特定的值为NA值，默认以下值会认定为NA： ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’.

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

#### 去重保留最后一条 dataframe.drop_duplicates(take_last=True)

`dataframe.drop_duplicates(take_last=True)`

```PYTHON
data.drop_duplicates(['k1', 'k2'], take_last=True)
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

#### 返回两列id的交集 np.intersect1d()

`np.intersect1d`

```python
np.intersect1d(movies.movieId.unique(), links.movieId.unique())
```

#### 检查去重后的id数量

```python
print(len(movies.movieId.unique()))
```

#### 查看变量相关性 

[返回dataframe中两两的相关性 pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)

[返回两个dataframe之间两两变量的相关性 pandas.DataFrame.corrwith](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corrwith.html#pandas.DataFrame.corrwith)

[返回一个series和其他的相关性]( https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.corr.html#pandas.Series.corr )

```PYTHON
# 1
correlations = app_train.corr()['TARGET'].sort_values()

# 2
app_train[['DAYS_BIRTH', 'FLAG_DOCUMENT_3']].corrwith(app_train['TARGET'])

# 3
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
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

#### 重新索引 .reindex() 

```PYTHON
# Series的reindex会根据新索引排序，如果索引不在，，传入缺失值
obj.reindex(['a','b','c','d','e'], fill_value=0)

# DataFrmame可以修改行、列索引
frame.reindex(['a','b','c','d'])
frame.reindex(columns=['Texas','Utah','California'])

```

#### 重命名索引 dataframe.index.map()  dataframe.index.rename()

```PYTHON
# 也可以用map方法
data.index.map(str.upper)

# rename则是复制一份数据集，不在原df上做修改
data.rename(index=str.title, columns=str.upper)

# rename可以结合紫癜性对象实现对部分轴标签的更新
data.rename(index={'OHIO': 'INDIANA'}, columns={'three', 'peekaboo'})
```

#### 笛卡尔积生成多重索引 pd.MultiIndex.from_product([[0, 1], ['a', 'b', 'c']])

```PYTHON
s_mi = pd.Series(np.arange(6), index=pd.MultiIndex.from_product([[0, 1], ['a', 'b', 'c']]))
```

### 多重索引

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

#### 多重索引交换层级 df.swaplevel()

```PYTHON
df[:5].swaplevel(0, 1, axis=0)
```

#### 多重索引层级重新排序 df.reorder_levels()

```PYTHON
df[:5].reorder_levels([1, 0], axis=0)
```

#### 多重索引排序 df.sort_index()

```PYTHON
s.sort_index(level=0)

# 用名字排序
s.index.set_names(['L1', 'L2'], inplace=True)
s.sort_index(level='L1')
```

#### 多重索引调整列顺序

```PYTHON
dfmi.reindex(['two','one'], level = 0, axis=1)
dfmi.reindex(['second','first'], level = 1, axis=1)
```



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



### 日期格式

#### [转换成日期格式 pandas.to_datetime](http://pandas.pydata.org/pandas-docs/version/0.15/generated/pandas.to_datetime.html)

* 可以处理`None`, 空字符串

```PYTHON
# 批量转换成日期格式
dt_overdue_record_raw[list_3] = dt_overdue_record_raw[list_3].apply(pd.to_datetime, infer_datetime_format=True)
```



#### 字符转日期 datetime.strptime

```python
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')
```



#### [根据日期返回指定格式的字符串 pandas.Series.dt.strftime](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.strftime.html)

```python
# ex1
df1['month'] = df1.ListingInfo.dt.strftime('%Y-%m')

# ex2
 rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"), periods=3, freq='s')
 rng.strftime('%B %d, %Y, %r')

# ex3 
stamp = datetime(2011, 1, 3)
str(stamp)
stamp.strftime('%Y-%m-%d')
```



#### 用dateutil中的parser解析日期字符

```PYTHON
from dateutil.parser import parse

parse('2011-01-03')
```





#### pandas.read_csv(parse_dates=, infer_dateime=True)读取数据格式时推断日期

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



####  [pandas.DataFrame.to_json()]( https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html )

`path_or_buf` : string or file handle, optional

   File path or object. If not specified, the result is returned as a string.

`orient` : string;  Indication of expected JSON string format.

-    Series

     - default is ‘index’
     - allowed values are: {‘split’,’records’,’index’,’table’}

-    DataFrame

     - default is ‘columns’
     - allowed values are: {‘split’,’records’,’index’,’columns’,’values’,’table’}

-    The format of the JSON string

     - ‘split’ : dict like {‘index’ -> [index], ‘columns’ -> [columns], ‘data’ -> [values]}

     - ‘records’ : list like [{column -> value}, … , {column -> value}]

     - ‘index’ : dict like {index -> {column -> value}}

     - ‘columns’ : dict like {column -> {index -> value}}

     - ‘values’ : just the values array

     - ‘table’ : dict like {‘schema’: {schema}, ‘data’: {data}} describing the data, and the data component is like `orient='records'`.

         *Changed in version 0.20.0.*

`index` : bool, default True

   Whether to include the index values in the JSON string. Not including the index (`index=False`) is only supported when orient is ‘split’ or ‘table’.

```python
df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                   index=['row 1', 'row 2'],
                   columns=['col 1', 'col 2'])

df.to_json(orient='split')

```



#### [转多层json]( https://blog.csdn.net/LeeGe666/article/details/89156691 )

```PYTHON
import json

article_info = {}
data = json.loads(json.dumps(article_info))

data['article1'] = 'NONE'

article2 = {'title': 'python基础', 'publish_time': '2019-4-1', 'writer': {}}
data['article2'] = article2

writer = {'name': '李先生', 'sex': '男', 'email': 'xxx@gmail.com'}
data['article2']['writer'] = writer

article = json.dumps(data, ensure_ascii=False)
print(article)
```



## 数据操作

#### 新增多列

```PYTHON
colList = list(dataset.columns)
colList.extend(['weekday_col5','week_col5','year_col5','month_col5','day_col5','hour_col5'])
dataset = dataset.reindex(columns = colList)

```



#### 删除列或行 .drop

```PYTHON
# 删除行
obj.drop(['d','c'])
# 删除列
data.drop(['two','four'],axis=1)
```

#### 取值映射 .map()

```PYTHON
dec.cand_nm[123456:123461].map(parties)  # parties是一个映射字典

animal={'aa':'pig','bb':'dog','cc':'horse','dd':'dog','ee':'pig','ff':'dog','gg':'horse'}
#data['animal']=data['food'].map(animal)
data['animal']=data['food'].map(lambda x:animal[x])
```

#### 数值范围映射

```PYTHON

# 方法1： 用自定义函数
def after_salary(s):                # s 为 salary 列对应的每一个元素
    if s <= 3000:
        return s
    else:
        return s - (s-3000)*0.5     # 对每一个元素进行操作并返回
   
df['after_salary'] = df['salary'].map(after_salary)

# 方法2：用pd.cut
bins = [0, 2, 18, 35, 65, np.inf]
names = ['<2', '2-18', '18-35', '35-65', '65+']

df['AgeRange'] = pd.cut(df['Age'], bins, labels=names)

# 方法3：用np.digitize
df = pd.DataFrame({'Age': [99, 53, 71, 84, 84],
                   'Age_units': ['Y', 'Y', 'Y', 'Y', 'Y']})

bins = [0, 2, 18, 35, 65]
names = ['<2', '2-18', '18-35', '35-65', '65+']

d = dict(enumerate(names, 1))

df['AgeRange'] = np.vectorize(d.get)(np.digitize(df['Age'], bins))
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

#### 全局修改dataframe数据 [pandas.DataFrame.replace](<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html>) 

```PYTHON
# overall replace
df.replace(to_replace='Female', value='Sansa', inplace=True)

# dict replace
df.replace({'sex': {'Female': 'Sansa', 'Male': 'Leone'}}, inplace=True)

# replace on where condition 
df.loc[df.sex == 'Male', 'sex'] = 'Leone'
```

#### 设定指定的值

```PYTHON
data[np.abs(data) > 3] = np.sign(data)*3
```

#### interval 属性

| Attributes                               |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`closed`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.closed.html#pandas.Interval.closed) | Whether the interval is closed on the left-side, right-side, both or neither |
| [`closed_left`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.closed_left.html#pandas.Interval.closed_left) | Check if the interval is closed on the left side. |
| [`closed_right`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.closed_right.html#pandas.Interval.closed_right) | Check if the interval is closed on the right side. |
| [`is_empty`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.is_empty.html#pandas.Interval.is_empty) | Indicates if an interval is empty, meaning it contains no points. |
| [`left`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.left.html#pandas.Interval.left) | Left bound for the interval              |
| [`length`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.length.html#pandas.Interval.length) | Return the length of the Interval        |
| [`mid`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.mid.html#pandas.Interval.mid) | Return the midpoint of the Interval      |
| [`open_left`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.open_left.html#pandas.Interval.open_left) | Check if the interval is open on the left side. |
| [`open_right`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.open_right.html#pandas.Interval.open_right) | Check if the interval is open on the right side. |
| [`right`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.right.html#pandas.Interval.right) | Right bound for the interval             |

#### 连续变量分段 pd.cut()

| 参数        | 说明   |
| --------- | ---- |
| right     |      |
| labels    |      |
| precision |      |
| rebins    | 返回分组 |

```PYTHON
# 指定分段，默认左开右闭
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)

# 指定分段标签
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)

# 查看分段的categories属性
cats.categories
```

#### 按分位数分段 pd.qcut()

```python
# 按4分位数进行切割
cats = pd.qcut(data, 4)

# 设置自定义的分位数
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1])
```

#### 取分段后的左边区间/右边区间

```PYTHON
ages = [20,22,25,27,21,23,37,23,23,54,26,29]
bins = [18,25,35,60,100]
cats = pd.cut(ages,10)
temp = pd.DataFrame(cats)
temp.columns = ['a']
temp['b'] = temp.a.apply(lambda x:x.left)
```



## 数据操作-切片

### [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-query>)

主要有三种方法

* .loc: selection by label

* .iloc: selection by integer position

* []

#### 通过[]方式取切片，赋值

```PYTHON
# 交换A,B两列的值
df[['B', 'A']] = df[['A', 'B']]

# 以下方式不能起到交换作用，因为loc在指定值之前先对齐列
df.loc[:, ['B', 'A']] = df[['A', 'B']]
df.loc[:, ['B', 'A']] = df[['A', 'B']].to_numpy()  # 正确的方法
```

#### 同时给多列赋值

```PYTHON
a=[['js',100,'cz',200],['zj',120,'xs',300],['zj',150,'xs',200],['js',200,'cz',200],['js',110,'wx',180],['js',300,'sz',250],['zj',210,'hz',280]]
df=pd.DataFrame(a)
df1.columns=['a','b','c','d']
df1['e']=111
df1['f']=222
df1.loc[2,'f']=223

con=(df1['a']=='zj') & (df1['c']=='xs')
df1.loc[con,['b','d']]=df1.loc[con,['e','f']].to_numpy()
```



#### 给某一行赋值

```PYTHON
 x.iloc[1] = {'x': 9, 'y': 99}
```

####  **if-then和if-then-else逻辑赋值** 

```PYTHON
df = pd.DataFrame(
        {'AAA': [4,5,6,7], 'BBB': [10,20,30,40], 'CCC': [100,50,-30,-50] })
# loc函数
# if-then 的逻辑在某列上，【选取满足的条件的行数据，然后对指定的列赋值】
df.loc[df.AAA >= 5, 'BBB'] = -1
df.loc[df.AAA >= 5,['BBB','CCC']] = 555
df.loc[df.AAA < 5,['BBB','CCC']] = 2000
df.loc[df.AAA >= 5, 'BBB'] = df.AAA

# where函数
# 根据每个位置元素的true或false，去赋值，true原值不变，false改变
df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True,False] * 2})
# df_mask的'AAA'列全是true，'BBB'列全是false，'CCC'是true,false,true,false
df.where(df_mask, -1000)
Out[6]: 
   AAA   BBB   CCC
0    4 -1000  2000
1    5 -1000 -1000
2    6 -1000   555
3    7 -1000 -1000
# 利用numpy中的where函数,实现if-then-else逻辑
df = pd.DataFrame(
     {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
# 'AAA'列如果大于5，赋值某列(新列)为high，否则赋值low
df['logic'] = np.where(df['AAA']>5, 'high', 'low')

```



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

# 多重索引的情况下
df.loc[('bar', 'two'), 'A']
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

#### 选取任意超过x的值 df.any(1)

```python
data[(np.abs(col)>3).any(1)]
```

####  dataframe中不同的列是否包含不同的值 df.isin()

```PYTHON
values = {'ids': ['a', 'b'], 'vals': [1, 3]}
df.isin(values)

# 结合any() 或者 all() 实现不同的筛选
row_mask = df.isin(values).all(1)
```

#### 复杂的切片操作

```PYTHON
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                    'b': ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                    'c': np.random.randn(7)})

# 建一个筛选标准
criterion = df2['a'].map(lambda x: x.startswith('t'))
df2[criterion]

# 同时用多个标准
df2[criterion & (df2['b'] == 'x')]
```



#### [随机抽取样本 df.sample()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html#pandas.DataFrame.sample)

> **n :** int, optional
> Number of items from axis to return. Cannot be used with frac. Default = 1 if frac = None.
>
> **frac :** float, optional
> Fraction of axis items to return. Cannot be used with n.
>
> **replace :** bool, default False
> Sample with or without replacement.
>
> **weights :** str or ndarray-like, optional
> Default ‘None’ results in equal probability weighting. If passed a Series, will align with target object on index. Index values in weights not found in sampled object will be ignored and index values in sampled object not in weights will be assigned weights of zero. If called on a DataFrame, will accept the name of a column when axis = 0. Unless weights are a Series, weights must be same length as axis being sampled. If weights do not sum to 1, they will be normalized to sum to 1. Missing values in the weights column will be treated as zero. Infinite values not allowed.
>
> **random_state :** int or numpy.random.RandomState, optional
> Seed for the random number generator (if int), or numpy RandomState object.
>
> **axis :** int or string, optional
> Axis to sample. Accepts axis number or name. Default is stat axis for given data type (0 for Series and DataFrames).



####  df.where()

* 返回筛选结果并维持原dataframe大小
* 可以替换dataframe中不满足条件的值

| 参数      | 作用                                 |
| ------- | ---------------------------------- |
| inplace | True, 不返回一个copy, 直接在原dataframe上改动  |
| axis    | 选择要对齐的轴，当需要指定dataframe中某列或某行作为替换值时 |
| level   | alignment level                    |

```PYTHON
df.where(df < 0, -df)

# 按行索引对齐，不符合条件的用A列的值替换
df2.where(df2 > 0, df2['A'], axis='index')

# 下面这个和上面等价，但会慢一些
df.apply(lambda x, y: x.where(x>0, y), y=df['A'])
```

#### df.mask()

df.where()的反向操作，满足条件的值要被替换掉

#### query() 方法

```PYTHON
# 例子1: 取符合 a<b<c 的行 
df[(df.a < df.b) & (df.b < df.c)]  # 传统方法


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

#### aggregation function

常用的`agg`函数如下:

* 这些函数会忽略`NA`值
* 任何将Series降维到scalar的函数都可以看作agg函数，比如` df.groupby('A').agg(lambda ser: 1) `
* `agg`和`apply`的区别在于，`agg`是传入一列数据返回一个标量，而`apply`是将一组数据全部传入，可以返回多维结果

| Function     | Description                              |
| :----------- | :--------------------------------------- |
| `mean()`     | Compute mean of groups                   |
| `sum()`      | Compute sum of group values              |
| `size()`     | Compute group sizes                      |
| `count()`    | Compute count of group                   |
| `std()`      | Standard deviation of groups             |
| `var()`      | Compute variance of groups               |
| `sem()`      | Standard error of the mean of groups     |
| `describe()` | Generates descriptive statistics         |
| `first()`    | Compute first of group values            |
| `last()`     | Compute last of group values             |
| `nth()`      | Take nth value, or a subset if n is a list |
| `min()`      | Compute min of group values              |
| `max()`      | Compute max of group values              |

#### 调用自己得聚合函数 DataFrame.GroupBy.agg  

```PYTHON
# count(distinct **)
df.groupby('tip').agg({'sex': pd.Series.nunique})

# 一次传入多个agg function
train_data[['Title','Survived']].groupby('Title', as_index=False).agg({np.mean, np.sum, 'size','count'})
grouped['C'].agg([np.sum, np.mean, np.std])
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

(grouped['C'].agg([np.sum, np.mean, np.std])
             .rename(columns={'sum': 'foo',
                              'mean': 'bar',
                              'std': 'baz'}))

# 新版本特性1
animals = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'],
                        'height': [9.1, 6.0, 9.5, 34.0],
                        'weight': [7.9, 7.5, 9.9, 198.0]})

animals.groupby("kind").agg(
    min_height=pd.NamedAgg(column='height', aggfunc='min'),
    max_height=pd.NamedAgg(column='height', aggfunc='max'),
    average_weight=pd.NamedAgg(column='weight', aggfunc=np.mean),
)

# 新版本特性2 或者直接传入元组 一样的
animals.groupby("kind").agg(
    min_height=('height', 'min'),
    max_height=('height', 'max'),
    average_weight=('weight', np.mean),
)

# 新版本特性3 传入非法的列名
animals.groupby("kind").agg(**{
    'total weight': pd.NamedAgg(column='weight', aggfunc=sum),
})
```

#### 对不同的列应用不同的函数

```python
grouped = tips.groupby(['sex', 'smoker'])

#1
grouped.agg({'tip': np.max, 'size': 'sum'})
grouped.agg({'C': np.sum,
             'D': lambda x: np.std(x, ddof=1)})

#2
grouped.agg({'tip_pct':['min', 'max', 'mean', 'std'], 
            'size':'sum'})
```

#### 传入多个lambda函数

```PYTHON
 grouped['C'].agg([lambda x: x.max() - x.min(),
                   lambda x: x.median() - x.mean()])
```

#### 无索引方式返回聚合函数 as_index=False

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



####  将聚合结果应用到dataframe上  grouped.transform()

transform中传入的函数只能是：

1. 产生可以广播的标量值，如np.mean
2. 或者是一个相同大小的结果数组

```PYTHON
# 简便的方法
key = ['one', 'two', 'one', 'two', 'one']
people.groupby(key).mean()  # 分组求平均值的结果
people.groupby(key).transform(np.mean)  # 效果是将聚合后平均值的结果合并到原来的dataframe中

# 填充缺失值
df['B'] = df.groupby(['A']).transform(lambda x : x.fillna(x.mean))

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

#### filter

```PYTHON
sf = pd.Series([1, 1, 2, 3, 3, 3])
sf.groupby(sf).filter(lambda x: x.sum() > 2)
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

#### [调整pivot_table显示层级](https://blog.csdn.net/weixin_40161254/article/details/91450574 )

需要将pivot_table结果转换如下：

<center>
    <img src="assets/20190611172741248.png" alt="原始" style="zoom: 27%;" />
    <img src="assets/20190611172843612.png" alt="目标" style="zoom: 29%;" />
</center>

```PYTHON
df1 = df.pivot_table(index=["user_id"], columns=["question_id"], values=["score","duration"]).fillna(0)
print(df1)

# 通过设置多重索引来调整显示的层级
c1=df['question_id'].drop_duplicates()
c2=['score','duration']
index=pd.MultiIndex.from_product([c1,c2])
df1.columns=index
print(df1)
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

## 数据操作-连接

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

#### 默认按索引合并`dataframe.join()` 

[pandas.DataFrame.join](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html)

`DataFrame.join`(*other*, *on=None*, *how='left'*, *lsuffix=''*, *rsuffix=''*, *sort=False*)

join的用法是默认用索引去连接，两个表至少有一个表是要用索引

```python
# 两个表都用索引
caller.set_index('key').join(other.set_index('key'))
# 一个表用索引
caller.join(other.set_index('key'), on='key')
```

#### 轴向连接 `pd.concat()`

[pandas.concat](http://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.concat.html)

直接将两个dataframe或者series进行纵向连接或者横向连接

| 参数               | 说明                                       |
| ---------------- | ---------------------------------------- |
| objs             | 参与连接的series, dict, dataframe             |
| axis             | 连接方向 {0 index, 1 columns}                |
| join             | {‘inner’, ‘outer’}, default ‘outer’      |
| join_axes        | list of Index objects  <br> Specific indexes to use for the other n - 1 axes instead of performing inner/outer set logic <br> 指定其他轴上面参与连接的索引 |
| ignore_index     | 不保留原连接轴上的索引，用一组新索引代替                     |
| keys             | 用于形成层次化索引                                |
| levels           | 设置了kets的情况下，指定层次化索引的level                |
| names            | 设置层次化索引的名称                               |
| verify_integrity | 检查结果对象新轴上的重复情况                           |
| sort             | Sort non-concatenation axis if it is not already aligned when join is ‘outer’. The current default of sorting is deprecated and will change to not-sorting in a future version of pandas. |

```PYTHON
# 在连接轴上创建层次化索引来区分连接的对象
result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])

# 通过传入字典也能实现层次化索引
pd.concat({'level1': df1, 'level2': df2}, axis=1)

# 通过ignore_index实现忽略索引直接合并
pd.concat([df1, df2], ignore_index=True)

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

#### 纵向合并数据集

[pandas.DataFrame.append](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html)

[pandas.concat](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html#pandas.concat) 



## 数据操作-排序

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

#### 拆分字符串并转换成list

```PYTHON
def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]
```

#### 字符先转小写再映射

```PYTHON
# 方法1
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)  # meat_to_animal是一个映射字典

# 方法2
data['food'].map(lambda x: meat_to_animal[x.lower()])
```



## 日期计算

Python标准库主要会用到以下模块：

* datetime
* time
* calendar

#### datetime模块中数据类型

| 类型      | 说明                 |
| --------- | -------------------- |
| date      | (年， 月， 日)       |
| time      | (时，分，秒，毫秒)   |
| datetime  | 储存日期和时间       |
| timedelta | 两个datetime之间差值 |



#### 取当前时间 datetime.now()

```python
from datetime import datetime
now = datetime.now()  # 取当前时间

now.year, now.month, now.day
```



#### 计算两个datetime对象时间差

```PYTHON
delta = datetime(2011, 1, 7) - datetime(2008, 6 24, 8, 15)  # 返回一个(day, second)的timedelta对象

delta.days  # 取间隔日数
delta.seconds  # 取剩余的秒数
```



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



#### [生成日期范围 pd.date_range()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html )

函数参数：
> 
> **start** : str or datetime-like, optional
> 
> **end** : str or datetime-like, optional
> 
> **periods** : integer, optional; 要生成多少期的日期范围
> 
> **freq** : str or DateOffset, default ‘D’
> 
> | Alias    | Description                                      |
> | :------- | :----------------------------------------------- |
> | C        | custom business day frequency                    |
> | D        | calendar day frequency                           |
> | W        | weekly frequency                                 |
> | M        | month end frequency                              |
> | SM       | semi-month end frequency (15th and end of month) |
> | CBM      | custom business month end frequency              |
> | MS       | month start frequency                            |
> | SMS      | semi-month start frequency (1st and 15th)        |
> | CBMS     | custom business month start frequency            |
> | Q        | quarter end frequency                            |
> | BQ       | business quarter end frequency                   |
> | QS       | quarter start frequency                          |
> | BQS      | business quarter start frequency                 |
> | A, Y     | year end frequency                               |
> | BA, BY   | business year end frequency                      |
> | AS, YS   | year start frequency                             |
> | BAS, BYS | business year start frequency                    |
> | BH       | business hour frequency                          |
> | H        | hourly frequency                                 |
> 
> **tz** : str or tzinfo, optional;  时区设置l，例如 `Asia/Hong_Kong` 
> 
> **normalize** : bool, default False
> 
> Normalize start/end dates to midnight before generating date range.
> 
> **name** : str, default None;  Name of the resulting DatetimeIndex.
> 
> **closed** : {None, ‘left’, ‘right’}, optional
> 
> Make the interval closed with respect to the given frequency to the ‘left’, ‘right’, or both sides (None, the default).
> 


```PYTHON
# 生成按天计算的时间范围
index = pd.date_range('4/1/2012', '6/1/2012')
```

#### 生成每个月月末日期

```PYTHON
index = pd.date_range('4/1/2012', '6/1/2012', freq='M')
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

#### np.cumsum()累加

```PYTHON
df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
```



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

| **SQL**                                  | **Pandas**                              |
| ---------------------------------------- | --------------------------------------- |
| select   * from airports                 | airports                                |
| select   * from airports limit 3         | airports.head(3)                        |
| select   id from airports where ident = ‘KLAX’ | airports[airports.ident   == ‘KLAX’].id |
| select   distinct type from airport      | airports.type.unique()                  |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   * from airports where iso_region = ‘US-CA’ and type = ‘seaplane_base’ | airports[(airports.iso_region   == ‘US-CA’) & (airports.type == ‘seaplane_base’)] |
| select   ident, name, municipality from airports where iso_region = ‘US-CA’ and type =   ‘large_airport’ | airports[(airports.iso_region   == ‘US-CA’) & (airports.type == ‘large_airport’)][[‘ident’, ‘name’,   ‘municipality’]] |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   * from airport_freq where airport_ident = ‘KLAX’ order by type | airport_freq[airport_freq.airport_ident == ‘KLAX’].sort_values(‘type’) |
| select   * from airport_freq where airport_ident = ‘KLAX’ order by type desc | airport_freq[airport_freq.airport_ident   == ‘KLAX’].sort_values(‘type’, ascending=False) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   * from airports where type in (‘heliport’, ‘balloonport’) | airports[airports.type.isin([‘heliport’,   ‘balloonport’])] |
| select   * from airports where type not in (‘heliport’, ‘balloonport’) | airports[~airports.type.isin([‘heliport’,   ‘balloonport’])] |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, type | airports.groupby([‘iso_country’,   ‘type’]).size() |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, count(*) desc | airports.groupby([‘iso_country’,   ‘type’]).size().to_frame(‘size’).reset_index().sort_values([‘iso_country’,   ‘size’], ascending=[True, False]) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, type | airports.groupby([‘iso_country’,   ‘type’]).size() |
| select   iso_country, type, count(*) from airports group by iso_country, type order by   iso_country, count(*) desc | airports.groupby([‘iso_country’,   ‘type’]).size().to_frame(‘size’).reset_index().sort_values([‘iso_country’,   ‘size’], ascending=[True, False]) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   type, count(*) from airports where iso_country = ‘US’ group by type having   count(*) > 1000 order by count(*) desc | airports[airports.iso_country   == ‘US’].groupby(‘type’).filter(lambda g: len(g) >   1000).groupby(‘type’).size().sort_values(ascending=False) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   iso_country from by_country order by size desc limit 10 | by_country.nlargest(10,   columns=’airport_count’) |
| select   iso_country from by_country order by size desc limit 10 offset 10 | by_country.nlargest(20,   columns=’airport_count’).tail(10) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   max(length_ft), min(length_ft), mean(length_ft), median(length_ft) from   runways | runways.agg({‘length_ft’:   [‘min’, ‘max’, ‘mean’, ‘median’]}) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   airport_ident, type, description, frequency_mhz from airport_freq join   airports on airport_freq.airport_ref = airports.id where airports.ident =   ‘KLAX’ | airport_freq.merge(airports[airports.ident   == ‘KLAX’][[‘id’]], left_on=’airport_ref’, right_on=’id’,   how=’inner’)[[‘airport_ident’, ‘type’, ‘description’, ‘frequency_mhz’]] |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| select   name, municipality from airports where ident = ‘KLAX’ union all select name,   municipality from airports where ident = ‘KLGB’ | pd.concat([airports[airports.ident   == ‘KLAX’][[‘name’, ‘municipality’]], airports[airports.ident ==   ‘KLGB’][[‘name’, ‘municipality’]]]) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| create   table heroes (id integer, name text); | df1   = pd.DataFrame({‘id’: [1, 2], ‘name’: [‘Harry Potter’, ‘Ron Weasley’]}) |
| insert   into heroes values (1, ‘Harry Potter’); | df2   = pd.DataFrame({‘id’: [3], ‘name’: [‘Hermione Granger’]}) |
| insert   into heroes values (2, ‘Ron Weasley’); |                                          |
| insert   into heroes values (3, ‘Hermione Granger’); | pd.concat([df1,   df2]).reset_index(drop=True) |



| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| update airports set home_link = ‘<http://www.lawa.org/welcomelax.aspx>’ where ident == ‘KLAX’ | airports.loc[airports[‘ident’] == ‘KLAX’,   ‘home_link’] = ‘<http://www.lawa.org/welcomelax.aspx>’ |

| **SQL**                                  | **Pandas**                               |
| ---------------------------------------- | ---------------------------------------- |
| delete   from lax_freq where type = ‘MISC’ | lax_freq   = lax_freq[lax_freq.type != ‘MISC’] |
|                                          | lax_freq.drop(lax_freq[lax_freq.type   == ‘MISC’].index) |

# Numpy

### 矩阵

#### [矩阵插入新数据 np.insert(*arr***,** *obj***,** *values***,** *axis=None*) ](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.insert.html)

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

#### [多维矩阵降为一维  np.ravel(*a*, *order='C'*) ](https://github.com/numpy/numpy/blob/v1.16.1/numpy/core/fromnumeric.py#L1583-L1687)

返回原始矩阵降维后的视图，修改值影响原始矩阵



#### 返回最大值的索引 np.argmax() 

#### 生成序列  np.arange

```PYTHON
np.arange(0.0, 5.0, 0.01)  #0-5，每隔0.01
```

#### 增加多一维

```PYTHON
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
```



#### [上三角矩阵 numpy.triu](https://docs.scipy.org/doc/numpy/reference/generated/numpy.triu.html)

 `numpy.triu(m, k=0)`

```PYTHON
# 1
np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)

array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 0,  8,  9],
       [ 0,  0, 12]])

# 2
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
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

### np.random.choice(a, size=None, replace=True, p=None) 

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



### 其他

#### [生成一列均分的数 numpy.linspace](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linspace.html#numpy-linspace)

```PYTHON
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([ 2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
```



#### [等比数列 numpy.logspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html )

numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)

* In linear space, the sequence starts at $base^{start}$ (base to the power of start) and ends with $base^{stop}$ "

```PYTHON
>>> a = np.logspace(0,9,10,base=2)
>>> a
array([   1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.,  512.])
```



# Python-实战问题

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

## 基础设置

#### matplotlib 配置 plt.rc()

```PYTHON
# 例子1
plt.rc('figure', figsize=(10, 10))

# 例子2
font_options = {'family':'monospace',
                'weight':'bold',
                'size':'small'}  # 将选项写成字典
plt.rc('font', **font_options)  # 然后将字典传入
```

#### 画图默认配置的基础代码

```PYTHON
# 导入 matplotlib 的所有内容（nympy 可以用 np 这个名字来使用）
from pylab import *

# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8,6), dpi=80)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(1,1,1)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plot(X, S, color="green", linewidth=1.0, linestyle="-")

# 设置横轴的上下限
xlim(-4.0,4.0)

# 设置横轴记号
xticks(np.linspace(-4,4,9,endpoint=True))

# 设置纵轴的上下限
ylim(-1.0,1.0)

# 设置纵轴记号
yticks(np.linspace(-1,1,5,endpoint=True))

# 以分辨率 72 来保存图片
# savefig("exercice_2.png",dpi=72)

# 在屏幕上显示
show()
```

#### 调整subplot周围的间距 plt.subplots_adjust()

```python
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)
```

#### 设置标题 ax.set_title()

```python
ax.set_title('aaa')
```

### 坐标轴调整 

#### 设置刻度、标签和图例

大部分的图标装饰项可以通过以下两种方式实现：

* 过程型的pyplot接口 (如plt.xxx，只对最近的fig起作用)
* [面向对象的原生matplotlib API (推荐，可以指定对哪个fig使用)](https://matplotlib.org/api/axes_api.html) 

#### [调整x轴坐标标签](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html#matplotlib.axes.Axes.set_xticklabels)

```python
# 设置x轴标签大小
plt.tick_params(axis='x', labelsize=8)    

# 旋转x轴坐标
plt.xticks(rotation=-15)    

# 旋转x轴坐标
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontdict={'horizontalalignment': 'right'})
```

#### 双坐标轴 ax.twinx()

```PYTHON
ax2 = ax1.twinx()
```

#### [三坐标轴](https://matplotlib.org/gallery/ticks_and_spines/multiple_yaxis_with_spines.html#sphx-glr-gallery-ticks-and-spines-multiple-yaxis-with-spines-py )

```PYTHON
par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))  # 设置右边的轴到
```

#### 设置坐标轴位置 ax.spines.set_position()

Spine position is specified by a 2 tuple of (position type, amount). The position types are:

- 'outward' : place the spine out from the data area by the specified number of points. (Negative values specify placing the spine inward.)
- 'axes' : place the spine at the specified Axes coordinate (from 0.0-1.0).
- 'data' : place the spine at the specified data coordinate. 移动到指定的坐标位置

Additionally, shorthand notations define a special positions:

- 'center' -> ('axes',0.5)
- 'zero' -> ('data', 0.0)

```PYTHON
par2.spines["right"].set_position(("axes", 1.2))  # 设置右边的轴到
```



#### 设置坐标轴范围 ax.set_xlim()

```PYTHON
# 通过pyplot接口
plt.xlim()  # 没有给参数，则返回当前图像的坐标轴范围
plt.xlim([0,10])  # 传入参数则设置坐标轴范围，仅对最近的AxesSubplot起作用

# 通过subplot实例方法
ax.get_xlim()
ax.set_xlim()

# 合理设置坐标轴范围
xmin ,xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()

dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2

xlim(xmin - dx, xmax + dx)
ylim(ymin - dy, ymax + dy)
```

#### [设置坐标轴范围 ax.set_ylim()](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_ylim.html#matplotlib.axes.Axes.set_ylim )

 Axes.set_ylim(*self*, *bottom=None*, *top=None*, *emit=True*, *auto=False*, ***, *ymin=None*, *ymax=None*) 



#### 设置刻度标签 ax.set_xticks() / ax.set_xticklabels()

```python
ax.set_xticks([0, 250, 500, 750, 1000])  # set_xticks设置要修改坐标标签的位置
ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')  # 根据上一行的位置设置对应的标签

# pyplot 方法 设置标签位置和相应的符号
xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])
```

#### 设置x轴名称 ax.set_xlabel()

```PYTHON
ax.set_xlabel('Stages')
```

#### 调整为四象限坐标轴形式

```PYTHON
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
```

#### 设置坐标轴标签

| 类型                | 说明                                       |
| :---------------- | :--------------------------------------- |
| `NullLocator`     | No ticks. ![img](assets/1540012569-1664-ticks-NullLocator.png) |
| `IndexLocator`    | Place a tick on every multiple of some base number of points plotted. ![img](assets/1540012569-4591-ticks-IndexLocator.png) |
| `FixedLocator`    | Tick locations are fixed. ![img](assets/1540012569-6368-ticks-FixedLocator.png) |
| `LinearLocator`   | Determine the tick locations. ![img](assets/1540012570-7138-ticks-LinearLocator.png) |
| `MultipleLocator` | Set a tick on every integer that is multiple of some base. ![img](assets/1540012570-8671-ticks-MultipleLocator.png) |
| `AutoLocator`     | Select no more than n intervals at nice locations. ![img](assets/1540012570-3960-ticks-AutoLocator.png) |
| `LogLocator`      | Determine the tick locations for log axes. ![img](assets/1540012571-8369-ticks-LogLocator.png) |

```PYTHON

# Setup a plot such that only the bottom spine is shown
def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)


plt.figure(figsize=(8, 6))
n = 8

# Null Locator
ax = plt.subplot(n, 1, 1)
setup(ax)
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.text(0.0, 0.1, "NullLocator()", fontsize=14, transform=ax.transAxes)
```



### 多图相关

#### [设定图纸1 plt.subplots()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots)

一行代码中同时返回figure和axes

```PYTHON
# 1
fig, axes = plt.subplots(2, 1)  # 设定两个子图

# 2
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)

# 3
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axes[0, 0].plot(x, y)
axes[1, 1].scatter(x, y)
```



#### [设定图纸2(推荐) fig.add_subplots()](https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot) 

对已存在的figure添加一个子图axes对象

```PYTHON
fig = plt.figure()
fig.add_subplot(221)

# equivalent but more general
ax1 = fig.add_subplot(2, 2, 1)

# add a subplot with no frame
ax2 = fig.add_subplot(222, frameon=False)

# add a polar subplot
fig.add_subplot(223, projection='polar')

# add a red subplot that share the x-axis with ax1
fig.add_subplot(224, sharex=ax1, facecolor='red')

#delete x2 from the figure
fig.delaxes(ax2)

#add x2 to the figure again
fig.add_subplot(ax2)
```



#### [添加每行不同数量的子图](https://matplotlib.org/gallery/subplots_axes_and_figures/axes_margins.html#sphx-glr-gallery-subplots-axes-and-figures-axes-margins-py )

```PYTHON
import numpy as np
import matplotlib.pyplot as plt


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


t1 = np.arange(0.0, 3.0, 0.01)

ax1 = plt.subplot(212)
ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
ax1.plot(t1, f(t1))

ax2 = plt.subplot(221)
ax2.margins(2, 2)           # Values >0.0 zoom out
ax2.plot(t1, f(t1))
ax2.set_title('Zoomed out')

ax3 = plt.subplot(222)
ax3.margins(x=0, y=-0.25)   # Values in (-0.5, 0.0) zooms in to center
ax3.plot(t1, f(t1))
ax3.set_title('Zoomed in')

plt.show()
```



#### 子图重叠

```PYTHON
plt.tight_layout()
```



### 图例

#### [添加图例 ax.legend()]( https://www.cnblogs.com/MTandHJ/p/10850415.html )

```PYTHON
# 方法1：添加subplot时候传入label参数
ax.plot(randn(1000).cumsum(), 'k', label='one')

# 方法2：调用ax.legend()
ax.legend(loc='best')

# 传入对应的线和label生成图例
ax.legend((line1, line2), ("line1", "line2"))

# 有副坐标轴下，图例放一起
lns1 = ax.plot(time, Swdown, '-', label = 'Swdown')
lns2 = ax.plot(time, Rn, '-', label = 'Rn')
ax2 = ax.twinx()
lns3 = ax2.plot(time, temp, '-r', label = 'temp')

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0) # 把三个图例放在一起
```

#### 多图图例调整 

```PYTHON
lines = []
labels = []
ax_list = [ax1, ax2, ax3]
for ax in ax_list:
    lines.append([l_1 for l_1 in ax.get_legend_handles_label()[0]])
    labels.append([a_1 for a_1 in ax.get_legend_handles_label()[1]])
    
lines = sum(lines, [])
labels = sum(labels, [])

ax1.legend(lines, labels, loc=0)
ax2.legend_.remove()
ax3.legend_.remove()
```

#### 移除图例

```PYTHON
ax3.legend_.remove()
```

#### [调整图例位置 ]( https://www.cnblogs.com/IvyWong/p/9916791.html )

 plt.legend(loc='String or Number', bbox_to_anchor=(num1, num2)) 

`bbox_to_anchor`参数 num1 用于控制 legend 的左右移动，值越大越向右边移动，num2 用于控制 legend 的上下移动，值越大，越向上移动。 

`loc`参数代表位置如下表

| String       | Number |
| ------------ | ------ |
| upper right  | 1      |
| upper left   | 2      |
| lower left   | 3      |
| lower right  | 4      |
| right        | 5      |
| center left  | 6      |
| center right | 7      |
| lower center | 8      |
| upper center | 9      |
| center       | 10     |

​     

<center class="half">
    <img src="assets/1392594-20191107215749853-533056299.png" alt="img" style="zoom: 45%;" /> 
    <img src="assets/20190920143244170.png" alt="在这里插入图片描述" style="zoom:45%;" />
</center>



#### 颜色设置(color参数)

`color`参数可以接受的输入如下：

- an RGB or RGBA (red, green, blue, alpha) tuple of float values in `[0, 1]` (e.g., `(0.1, 0.2, 0.5)` or `(0.1, 0.2, 0.5, 0.3)`);
- a hex RGB or RGBA string (e.g., `'#0f0f0f'` or `'#0f0f0f80'`; case-insensitive);
- a string representation of a float value in `[0, 1]` inclusive for gray level (e.g., `'0.5'`);
- one of `{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}`;
- a X11/CSS4 color name (case-insensitive);
- a name from the [xkcd color survey](https://xkcd.com/color/rgb/), prefixed with `'xkcd:'` (e.g., `'xkcd:sky blue'`; case insensitive);
- one of the Tableau Colors from the 'T10' categorical palette (the default color cycle): `{'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}` (case-insensitive);
- a "CN" color spec, i.e. `'C'` followed by a number, which is an index into the default property cycle (`matplotlib.rcParams['axes.prop_cycle']`); the indexing is intended to occur at rendering time, and defaults to black if the cycle does not include color.

```python
# 用“CN”的方法
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

th = np.linspace(0, 2*np.pi, 128)


def demo(sty):
    mpl.style.use(sty)
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.set_title('style: {!r}'.format(sty), color='C0')  # 这里代表用配色风格中的0号配色

    ax.plot(th, np.cos(th), 'C1', label='C1')  # 这里代表用配色风格中的1号配色
    ax.plot(th, np.sin(th), 'C2', label='C2')  # 这里代表用配色风格中的2号配色
    ax.legend()

demo('default')
demo('seaborn')
```





## 保存文件

#### 图存成PDF文件

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

#### 将图标保存到文件 plt.savefig

```python
plt.savefig('figpath.svg')
```



## 画图



#### 画各变量箱形图

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

#### 画各变量分布图

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

#### seaborn 画多个子图

```PYTHON
f, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.catplot(x="Sex", y="Survived", kind='bar',data=train_data, ax=axes[0])
sns.catplot(x="Ticket", y="Survived",kind='bar',  data=train_data, ax=axes[1]);
```



#### 根据二维数据画线 ax.plot(x, y)

```python
ax.plot(x, y, linesyple='--', color='g')
```

#### 给特殊点做标记

```PYTHON
t = 2*np.pi/3
plot([t,t],[0,np.cos(t)], color ='blue', linewidth=2.5, linestyle="--")
scatter([t,],[np.cos(t),], 50, color ='blue')

annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
         xy=(t, np.sin(t)), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plot([t,t],[0,np.sin(t)], color ='red', linewidth=2.5, linestyle="--")
scatter([t,],[np.sin(t),], 50, color ='red')

annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
```

 <div align=center><img src="assets/1540010726-1211-exercice-9.png" alt="img" style="zoom: 67%;" /> 




##  pandas 绘图函数

### [df.plot()]( https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#plotting )

#### [参数说明]( https://github.com/pandas-dev/pandas/blob/v0.25.3/pandas/plotting/_core.py#L504-L1533 )

| callable method                          |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`DataFrame.plot`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot)([x, y, kind, ax, ….]) | DataFrame plotting accessor and method   |
| [`DataFrame.plot.area`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.area.html#pandas.DataFrame.plot.area)(self[, x, y]) | Draw a stacked area plot.                |
| [`DataFrame.plot.bar`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html#pandas.DataFrame.plot.bar)(self[, x, y]) | Vertical bar plot.                       |
| [`DataFrame.plot.barh`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.barh.html#pandas.DataFrame.plot.barh)(self[, x, y]) | Make a horizontal bar plot.              |
| [`DataFrame.plot.box`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.box.html#pandas.DataFrame.plot.box)(self[, by]) | Make a box plot of the DataFrame columns. |
| [`DataFrame.plot.density`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.density.html#pandas.DataFrame.plot.density)(self[, bw_method, ind]) | Generate Kernel Density Estimate plot using Gaussian kernels. |
| [`DataFrame.plot.hexbin`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.hexbin.html#pandas.DataFrame.plot.hexbin)(self, x, y[, C, …]) | Generate a hexagonal binning plot.       |
| [`DataFrame.plot.hist`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.hist.html#pandas.DataFrame.plot.hist)(self[, by, bins]) | Draw one histogram of the DataFrame’s columns. |
| [`DataFrame.plot.kde`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.kde.html#pandas.DataFrame.plot.kde)(self[, bw_method, ind]) | Generate Kernel Density Estimate plot using Gaussian kernels. |
| [`DataFrame.plot.line`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html#pandas.DataFrame.plot.line)(self[, x, y]) | Plot Series or DataFrame as lines.       |
| [`DataFrame.plot.pie`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.pie.html#pandas.DataFrame.plot.pie)(self, \*\*kwargs) | Generate a pie plot.                     |
| [`DataFrame.plot.scatter`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html#pandas.DataFrame.plot.scatter)(self, x, y[, s, c]) | Create a scatter plot with varying marker point size and color. |
| [`DataFrame.boxplot`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html#pandas.DataFrame.boxplot)(self[, column, by, ax, …]) | Make a box plot from DataFrame columns.  |
| [`DataFrame.hist`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html#pandas.DataFrame.hist)(data[, column, by, grid, …]) | Make a histogram of the DataFrame’s.     |



#### 线性图

```python
# series会画一条线
s = Series(np.random.randn(10).cumsum(), index=np.arange(0,100,10))
s.plot

# dataframe会画很多条线，并自动创建图例
df = DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df.plot

```

#### 柱状图 [DataFrame.plot.bar](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html#pandas.DataFrame.plot.bar)

```PYTHON
fig, axes = plt.subplots(2, 1)  # 设定两个子图
data = Series(np.random.rand(16), index=list('asffdsfgge'))

data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)  # 在子图1画图
data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)  # 在子图2画图
```

#### 堆积柱状图 stacked=True

```PYTHON
df.plot(kind='barh', stacked=True, alpha=0.5)
```

#### 直方图 dataframe.hist(bins=)

```python
tips['tip_pct'] = tips['tip']/tips['total_bill']
tips['tip_pct'].hist(bins=50)
```

#### 密度图 kind='kde'

```python
tips['tip_pct'].plot(kind='kde')

# 将直方图和密度图画在同一幅图上
comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)的正态分布
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)的正态分布
values = Series(np.concatenate([comp1, comp2]))

values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k--')
```

#### 散点图 plt.scatter()

```PYTHON
plt.scatter(trans_data['m1'], trans_data['unemp'])
```

#### 散点矩阵图 pd.scatter_matrix()

```python
pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)
```

#### [热力图](https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py)

```PYTHON
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
```



#### [颜色设置](https://matplotlib.org/api/colors_api.html)

- an RGB or RGBA tuple of float values in `[0, 1]` (e.g., `(0.1, 0.2, 0.5)` or `(0.1, 0.2, 0.5, 0.3)`);
- a hex RGB or RGBA string (e.g., `'#0f0f0f'` or `'#0f0f0f80'`; case-insensitive);
- a string representation of a float value in `[0, 1]` inclusive for gray level (e.g., `'0.5'`);
- one of `{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}`;
- a X11/CSS4 color name (case-insensitive);
- a name from the [xkcd color survey](https://xkcd.com/color/rgb/), prefixed with `'xkcd:'` (e.g., `'xkcd:sky blue'`; case insensitive);
- one of the Tableau Colors from the 'T10' categorical palette (the default color cycle): `{'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}` (case-insensitive);
- a "CN" color spec, i.e. `'C'` followed by a number, which is an index into the default property cycle (`matplotlib.rcParams['axes.prop_cycle']`); the indexing is intended to occur at rendering time, and defaults to black if the cycle does not include color.





# Toolz 函数式编程库

#### 嵌套函数 pipe

Pipe a value through a sequence of functions

`pipe(data, f, g, h)` is equivalent to `h(g(f(data)))`

```PYTHON
from toolz.curried import pipe, map, compose, get  
>>> double = lambda i: 2 * i
>>> pipe(3, double, str)
    '6'
```



# 数据库操作

#### pgsql 操作

1. 创建一个数据库连接conn
2. 创建一个游标cur
3. 通过游标执行各种操作
4. commit操作
5. 关闭游标
6. 关闭数据库连接



```python
>>> import psycopg2

# Connect to an existing database
>>> conn = psycopg2.connect("dbname=test user=postgres")

# Open a cursor to perform database operations
>>> cur = conn.cursor()

# Execute a command: this creates a new table
>>> cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")

# Pass data to fill a query placeholders and let Psycopg perform
# the correct conversion (no more SQL injections!)
>>> cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",
...      (100, "abc'def"))

# Query the database and obtain data as Python objects
>>> cur.execute("SELECT * FROM test;")
>>> cur.fetchone()  # 返回元组
(1, 100, "abc'def")

# Make the changes to the database persistent
>>> conn.commit()

# Close communication with the database
>>> cur.close()
>>> conn.close()
```

