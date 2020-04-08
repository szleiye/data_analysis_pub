#### 添加 SSH

检查是否已存在ssh

```shell
$ cd ~/.ssh
$ ls
```

创建SSH key

* -t 指定密钥类型，默认是 rsa ，可以省略。
* -C 设置注释文字，比如邮箱。
* -f 指定密钥文件存储文件名。

 ```shell
$ ssh-keygen -t rsa -C "your_email@example.com"
 ```

以上代码省略了 -f 参数，因此，运行上面那条命令后会让你输入一个文件名，用于保存刚才生成的 SSH key 代码，如：

```shell
Generating public/private rsa key pair.
# Enter file in which to save the key (/c/Users/you/.ssh/id_rsa): [Press enter]
# 可以不输入，用默认文件名
```

```shell
Enter passphrase (empty for no passphrase): 
# Enter same passphrase again:
# push时候需要的密码，可以不输入
```

 拷贝 SSH内容

```
$ clip < ~/.ssh/id_rsa.pub
```

测试 SSH key

```shell
$ ssh -T git@github.com
# 如果你看到 “access denied” ，者表示拒绝访问，那么你就需要使用 https 去访问，而不是 SSH 
```



 