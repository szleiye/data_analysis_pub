[TOC]

## 参考资料

[Linux Basic - 不完全摘录1](https://mp.weixin.qq.com/s/QH5LqRgi6GVHb7nm6N7xQg)

[一篇精辟的Linux必知必会，推荐收藏](https://app.getpocket.com/read/3060678595)



## Linux 基础知识

#### 学习路径

![img](assets/linux_syntax/https___mmbiz.qpic.cn_mmbiz_png_UnzPPicpR0KVrhslf3yg0Lbn5bmmALg5OicTfBIDVCA6pKVgia7F4niczXyrMpib87wSGGAUHsQl84VVdjzAxQ56wvQ_640_wx_fmt=png)



#### 系统概念

* 客户端 X Client

    > 我们在电脑上操作Linux，需要一个客户端，我们称之为 X Client，常用的客户端桌面环境有XFCE、LXDE、KDE和GNOME等。

* Terminal

    > 我们在使用LInux的时候并不是直接与系统进行交互的，而是通过一个叫shell的中间程序来完成的，而这个程序是需要我们在一个窗口进行输入的显示的。终端的本质其实就是对应着LInux上的设备，Linux多用户登陆是可以通过不同的设备来完成，默认提供6个。

* Shell

    > 了解完Terminal，也来了解一下Shell，Shell的中文名是"壳"，蛋壳里就应该有”核“，也就是Linux内核，Shell提供给使用者使用界面，在UNIX/Linux 中比较流行的Shell有bash、zsh、ksh、csh等，Ubuntu终端默认使用bash。



## 基本操作

#### Shell 操作快捷键

| 按键            | 作用                                         |
| :-------------- | :------------------------------------------- |
| `Ctrl+c`        | 终止命令                                     |
| `Ctrl+d`        | 键盘输入结束或退出终端                       |
| `Ctrl+s`        | 暂停当前程序，暂停后按下任意键恢复运行       |
| `Ctrl+z`        | 将当前程序放到后台运行，恢复到前台为命令`fg` |
| `Ctrl+a`        | 将光标移至输入行头，相当于`Home`键           |
| `Ctrl+e`        | 将光标移至输入行末，相当于`End`键            |
| `Ctrl+k`        | 删除从光标所在位置到行末                     |
| `Alt+Backspace` | 向前删除一个单词                             |
| `Shift+PgUp`    | 将终端显示向上滚动                           |
| `Shift+PgDn`    | 将终端显示向下滚动                           |



#### 常用通配符

| 字符                    | 含义                                       |
| :---------------------- | :----------------------------------------- |
| `*`                     | 匹配 0 或多个字符                          |
| `?`                     | 匹配任意一个字符                           |
| `[list]`                | 匹配 list 中的任意单一字符                 |
| `[^list]`               | 匹配 除 list 中的任意单一字符以外的字符    |
| `[c1-c2]`               | 匹配 c1-c2 中的任意单一字符 如：[0-9][a-z] |
| `{string1,string2,...}` | 匹配 string1 或 string2 (或更多)其一字符串 |
| `{c1..c2}`              | 匹配 c1-c2 中全部字符 如{1..10}            |

```shell
# 找出当前目录下所有txt文件
ls *.txt

# 创建一些文件，比如相同前缀，只是后缀不同
touch diary_sam_{1..10}.txt
```



#### 寻找帮助 man

 `man` ，后面跟上命令名称即可。



#### 安装额外命令

``` shell
$ sudo apt-get update
$ sudo apt-get install sysvbanner
```



## 用户和文件权限管理

#### 查看当前用户 -who

关键命令用 `who am i`，即可输出当前的用户名，更多参数：

| 参数 | 说明                       |
| :--- | :------------------------- |
| `-a` | 打印能打印的全部           |
| `-d` | 打印死掉的进程             |
| `-m` | 同`am i`，`mom likes`      |
| `-q` | 打印当前登录用户数及用户名 |
| `-u` | 打印当前登录用户登录信息   |
| `-r` | 打印运行等级               |



#### 切换到 root 用户

在Linux中最高权限的用户角色就是root了，，建议平时还是使用普通角色来操作系统。新系统默认root没有密码，需要用到 root 的时候， 用`sudo`执行命令。

```SHELL
# sudo 获得5分钟root 权限
$  sudo

# 输入当前管理源密码进入root用户
sudo -i

# 设定root用户密码，并且切换到root
$  sudo passwd root
$  su
```



#### 用户的创建与删除

```shell
# 创建用户
$ sudo add user xxx

# 切换用户
su -l <username>

# 删除用户
sudo deluser <username>  # 默认保留想关路径下工作文件
```



#### 用户组创建、赋权和删除



## 文件和目录管理

| 常用命令 | 备注                                                         |
| -------- | ------------------------------------------------------------ |
| ls       | 查看当前目录下内容                                           |
| cd       | 切换目录                                                     |
| mkdir    | 新建文件夹                                                   |
| pwd      | 获取当前绝对路径                                             |
| rm       | `rm`是指remove，删除，可以用 `rm-r`删除文件夹，并且递归删除，**删除操作一律谨慎使用**。 |
| mv       | 移动，即剪贴，后面跟文件名和新的路径名                       |
| find     | 指定目录找文件， `find   path   -option`                     |
| touchu   | 修改文件或者目录的时间属性，包括存取时间和修改时间           |





#### 显示文件信息 ls

查看指定目录下信息，未指定则显示当前目录

```shell
ls -l

ls -lah
```

<img src="assets/linux_syntax/https___mmbiz.qpic.cn_mmbiz_png_UnzPPicpR0KVrhslf3yg0Lbn5bmmALg5OC2Y4nDWFsyXrgvkvQiatlo5vRVAoA1Vib0NHnlK5DoCGkqHnJDbf6weA_640_wx_fmt=png" alt="img" style="zoom:50%;" />

![img](assets/linux_syntax/https___mmbiz.qpic.cn_mmbiz_png_UnzPPicpR0KVrhslf3yg0Lbn5bmmALg5OmSJFNfgebUpWwxicwaNEz9PGiap4X90bOSPcncJpibJibpQWDzGUiasbEqA_640_wx_fmt=png)

#### 更改文件所有者

```shell
sudo chown <user1> <user2>
```



#### [修改文件权限 chmod](https://www.runoob.com/linux/linux-comm-chmod.html)



* 三组固定的权限，分别对应拥有者，所属用户组，其他用户
* 文件的读写执行对应字母 `rwx`，均拥有以二进制表示就是 `111`，用十进制表示就是 `7`
* 假设新建的文件 learning 的权限是 `rw-rw-r--`，换成对应的十进制表示就是 664

文件权限表示

<img src="assets/linux_syntax/https___mmbiz.qpic.cn_mmbiz_png_UnzPPicpR0KVrhslf3yg0Lbn5bmmALg5OlTXMDcN2r6s4g5WxGKVxsic36K2iaresOXnPNHYaCPgn5KKic9gn9KmcA_640_wx_fmt=png" alt="img" style="zoom:50%;" />

```shell
chmod 600 learning
ls -alh learning
```



## 文件查看和编辑

| 命令         | 备注                                                 |
| ------------ | ---------------------------------------------------- |
| wc           | 查看文档字数之类的值                                 |
| head 和 tail | 查看文件的前若干行或后若干行                         |
| cat 和 tac   | 遍历文件内容打印在 linux 界面，前者正向，后者反向    |
| nl           | 和 cat 类似，但是会加上行号                          |
| more 和 less | 也是遍历命令，more 按页展示带检索功能； less更加强大 |
| grep         | 遍历文件中带有特定字段的内容                         |
| sed          | 依照脚本指令编辑文件；建议具体查下用法               |
| awk          | 类似sql? 建议具体查下用法                            |
| stat         | 查看文件的细节情况，大小、创建时间、权限             |



#### 计算文档字数之类的值 wc

```SHELL
$ wc -c helloworld.txt # 字数

11 helloworld.txt

$wc -l helloworld.txt # 行数

1 helloworld.txt
```



## 系统管理

| 命令           | 备注                                                        |
| -------------- | ----------------------------------------------------------- |
| top 和 ps      | 查看进程状态，top 是带界面实时监控， ps是瞬时的             |
| vmstat         | 查看虚拟内存状态                                            |
| free           | 显示内存占用状态                                            |
| chown 和 chmod | `chown`是用户与用户组设置， `chmod`是设置读、写、运行的权限 |
| nohup          | 在命令之前带这个玩意，能让命令在后台一直运行                |



