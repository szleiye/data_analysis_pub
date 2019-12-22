[TOC]

# **SAS**
## 程序开头的注释

```MYSQL
*---------------------------------------------------*
| PURPOSE: 借新还旧和结清再贷客户
| PROGRAMMER: LEI WY 
| DATE: 2017-07-17
| REMARK:
| 
*---------------------------------------------------*;
```

```MYSQL
——————————————————————————————————————————————————————
	版本号：v0.12
	修改日期：2017-01-17
	修改人：雷文烨
	修改内容：分析借款人属性和逾期率的关系
—————————————————————————————————————————————————————
```

```MYSQL
/*********************************************************************************/
/*  生成该变量的映射表                                                       */
/*********************************************************************************/
```

```
---------------------------------------------------------------------------------------------------------------------------------
DATE: 2017-01-09
PURPOSE:
	将案件分成直批案件和非直批案件
	发放日期在2015.7.1-2016.6.3
	截止日期到2016.12.30时点的逾期情况
	1. 做一个vintage图
	2. 做一个曾经DPD15 和 曾经DPD30的分析
---------------------------------------------------------------------------------------------------------------------------------
```



## 批量读取文件名

```mysql
/*读取APPLY CSV数据*/
%let dir_apply = D:\work\data\apply;
options noxwait;
x "dir &dir_apply\*.csv /b > &dir_apply\filename.txt";
```

## 导入数据

`CALL EXECUTE` `INFILE`

```MYSQL
data _null_;
infile "&dir_apply\filename.txt";
input filename & $100.;
call execute('%importfile('||scan(filename,1,'.')||')');
run;
```

`LRECL`：设置读取数据的data lines 的长度

```SAS
proc import datafile="D:\work\raw_data\pingfenka\PINGFENKA_GBK20190105.csv" 
            out=PINGFENKA_GBK20190105 
            dbms=dlm 
            replace ;    delimiter='09'x;
			GETNAMES=NO;datarow=2;
run;
```

## 从lib中读取库的所有数据集名称

```SAS
PROC SQL;
CREATE TABLE APPROVE_MEM_LIST AS
SELECT (MEMNAME) INTO:APPROVE_LIST 
SEPARATED BY " "
FROM DICTIONARY.TABLES
WHERE LIBNAME = "APPROVE";
QUIT;
```



## 手动输入数据

`INPUT` `DATALINES` `CARDS`

```MYSQL
#用DATALINES 和 用CARDS 同样意思
DATA USPRESIDENTS;
	INPUT President $ Party $ Number;
	DATALINES;
Adams F 2
Lincoin R 16
Grant R 18
Kennedy D 35
	;
RUN;
```

## 字符串转为日期/字符串转为日期时间
`INPUT` 
```MYSQL
进件日期=input(substr(申请编号,3,8),yymmdd10.); 
input(eval_start_time,E8601DT19.)  as eval_start_time format datetime19.
```

## 用B表的数据来更新A表的数

`UPDATE` 

```mysql
/*改报表日期*/
PROC SQL;
UPDATE MAKEUP1 SET 报表日期 = '31MAR2016'D WHERE 报表日期 = '30APR2016'D;
UPDATE MAKEUP1 SET 逾期天数=. WHERE 申请号 NOT IN 
			("AI20140929000001"	"AI20141013000003"	"AI20141013000008"
				"AI20150526000105"	"AI20150529000097"	"AI20150707000042"
				"AI20150902000063"	"AI20151104000055"	"AI20151223000079"
				"AI20160119000050");

UPDATE MAKEUP1 SET 逾期天数= ( 
				SELECT 当前逾期天数 FROM OVERDUE.OVERDUE_20160401 
						WHERE MAKEUP1.申请号=OVERDUE_20160401.申请编号) 
						WHERE 申请号 NOT IN 
			("AI20140929000001"	"AI20141013000003"	"AI20141013000008"
				"AI20150526000105"	"AI20150529000097"	"AI20150707000042"
				"AI20150902000063"	"AI20151104000055"	"AI20151223000079"
				"AI20160119000050");
QUIT;
```

## 连接字符和数字组成新字符串

`CATS`  `WEEK` 

```MYSQL
cats("第",put(week(进件日期,'V'),4.),"周") as 周数
```

## 数字转字符串

`PUT`

```MYSQL
put(week(进件日期,'V'),4.)
```
## 字符串转数字
`INPUT`
```MYSQL
input(aaa,8.)


oldvar	input function	newvar
32000	newvar=input(oldvar, 5.);	32000
32000	newvar=input(oldvar, 5.2);	320.00
32,000	newvar=input(oldvar,comma6.);	32000
```
## TABULATE中用format指定变量顺序

>1. 定义格式`
>2. 添加顺序的新变量`
>3. 把顺序转换回原变量值`

`FORMAT` `INPUT`


```MYSQL
proc format;
invalue sequence
'第1周'=1	'第2周'=2	'第3周'=3	;
VALUE classcode
1='第1周'	2='第2周'	3='第3周';
run;
```

```MYSQL
DATA work.apply_anlys;
SET work.apply_anlys;
classrank = INPUT(周数,sequence.);
RUN;
```

```MYSQL
proc tabulate data = work.apply_anlys;
class 门店 业务种类 申请月份 案件类型 classrank;
format classrank classcode.;
table (all='合计' classrank=''),(all='合计'*N='案件数' 业务种类='业务类型'*rowpctn='%' 案件类型*rowpctn='%')
			/BOX="按周的进件量的分析 （按申请日期取周）" printmiss misstext=' ';
run;
```

## 添加序号

`_N_`

```mysql
DATA CT_LATEST_TEMP;
SET COPETIME.COPETIME_20161104;
	序号=_N_;
IF 阶段名称="问题件处理" AND 问题件类型='现场调查' THEN 阶段名称 = '现场调查';
IF 阶段名称="补件处理" AND 问题件类型='现场调查' THEN 阶段名称 = '现场调查';
RUN;
```

## 取领先一期的值

> 1. 添加序号
>
> 2. 逆序排列
>
> 3. 用滞后函数

`LAG` `DESCENDING`

```mysql
PROC SORT DATA=CT_LATEST_TEMP;
BY DESCENDING 序号;
RUN;

```

```mysql
data CT_TEMP;
set CT_LATEST_TEMP;
by notsorted 申请流水号;
INFORMAT 下阶段到达时间 DATETIME18.
下阶段处理时间 DATETIME18.
下阶段结束时间 DATETIME18.;
FORMAT 下阶段到达时间 NLDATMS33.
下阶段处理时间 NLDATMS33.
下阶段结束时间 NLDATMS33.;
下一阶段=lag(阶段名称);
if first.申请流水号 then call missing(下一阶段);
下阶段到达时间=LAG(任务到达时间);
if first.申请流水号 then call missing(下阶段到达时间);
下阶段处理时间=LAG(处理开始时间);
if first.申请流水号 then call missing(下阶段处理时间);
下阶段结束时间=LAG(处理结束时间);
if first.申请流水号 then call missing(下阶段结束时间);
run;
```

## 自定义函数和循环

`FCMP` `IF...THEN...` `SELECT...WHEN...`

```MYSQL
/*自定义函数-计算工作日数*/
proc fcmp outlib=function.funcsol.conversion;
	function DAYNUM(STAR_DAY,END_DAY);
	i=0;
	IF END_DAY<>. THEN 
		DO;
			DO DATE = STAR_DAY TO END_DAY;
				SELECT ; 
				WHEN (DATE>= '01Jan2016'd  AND  DATE <= '03Jan2016'd ) i=i+0;
				WHEN (DATE= '06Feb2016'd  OR  DATE = '14Feb2016'd ) i=i+1;
				OTHERWISE i=i+1;
				END;
			END;
		END;
	ELSE i=.;
  return (i);
  endsub;
 run;

 options cmplib=(function.funcsol);
```

## 用百分比显示
`FORMAT` `PICTURE`

```MYSQL
proc format;
picture hellpct
0-100='009.99%'(mult=100);
RUN;
```

## 设置TABULATE 的格式，有哪些值出现在表格里

```MYSQL
DATA CLASS_TEMP1;
DO 进件类型='AI','BJ','YA','YB','YS';
 DO 回退='回退','通过';
 OUTPUT;
 END;
END;
RUN; 
```

## 设置分部的输出格式

```MYSQL
PROC FORMAT;
	VALUE $BRANCH
		'福田门店','深圳','深圳宝安'	 =	'01深圳福田'
		'东莞','东莞门店'				=	'02东莞 '
		'佛山','佛山门店'				=	'03佛山 '
		'福州','福州门店'				=	'04福州 '
		'武汉','武汉门店'				=	'05武汉 '
		'长沙','长沙门店'				=	'06长沙 '
		'南昌','南昌门店'				=	'07南昌 '
		'厦门','厦门门店'				=	'08厦门 '
		'合肥','合肥门店'				=	'09合肥 '
		'广州','广州门店'				=	'10广州'
		'成都','成都门店'				=	'11成都'
		'石家庄','石家庄门店'		   =  '12石家庄'
		'上海门店'					 =	 '13上海'
		'贵阳门店'					=	'14贵阳'
		'北京门店'					=	'15北京'
		'郑州门店'					=	'16郑州'
;RUN;
```
## 删除字符串中指定的字符串

```MYSQL
DATA TEMP;
SET APPROVE_RJ;
FORMAT 否决意见详情_RE_TEMP $CHAR200. ;
/*暂时删除意见详情中 其他的'0011 其他-请注明' 原因*/
否决意见详情_RE_TEMP=TRANWRD(否决意见详情,'0011 其他-请注明','');
RUN;
```

## 保留指定字符

`COMPRESS`

```MYSQL
DATA APPROVE_RJ_TEMP;
SET APPROVE_RJ;
FORMAT RJ_CODE_TEMP $CHAR200. ;

否决意见详情_RE_TEMP=TRANWRD(否决意见详情_RE_TEMP,'-002','');

IF INDEX_其他>0 THEN 
RJ_CODE_TEMP=CATX(' ','0011',COMPRESS(否决意见详情_RE_TEMP,'-0123456789 ','K'));
ELSE RJ_CODE_TEMP=STRIP(COMPRESS(否决意见详情_RE_TEMP,'-0123456789 ','K'));

IF INDEX_分部否决>0 THEN 
RJ_CODE_TEMP=CATX(' ','0024',RJ_CODE_TEMP);
RJ_CODE_1 = SCAN(RJ_CODE_TEMP,1,' ');
RJ_CODE_2 = SCAN(RJ_CODE_TEMP,2,' ');
RJ_CODE_3 = SCAN(RJ_CODE_TEMP,3,' ');
RJ_CODE_4 = SCAN(RJ_CODE_TEMP,4,' ');
RUN;
```

## 取距离申请日前三个月的日期

```MYSQL
/* 计算3个月，6个月，12个月内的征信查询记录 */
DATA PRIMARY_QUERYRECORD_1;
SET PRIMARY_QUERYRECORD;
FORMAT DATE YYMMDD10.;
IF SEARCHDATE<=进件日期 AND SEARCHDATE^=.;

/*同一机构（不含本人查询）在 15 个自然日内因同一原因查询的，可以算作一次信用查询记录*/
RETAIN 有效查询 LAST_SEARCHDATE;
FORMAT LAST_SEARCHDATE YYMMDD10.;
BY NOTSORTED SERIALNO SEARCHBY SEARCHREASON ;
IF FIRST.SEARCHREASON THEN DO;LAST_SEARCHDATE=SEARCHDATE;有效查询=1;END;
ELSE IF INTCK('DAY',LAST_SEARCHDATE,SEARCHDATE)<15 THEN 有效查询=0;ELSE 有效查询=1;
IF 有效查询=1 THEN LAST_SEARCHDATE=SEARCHDATE;

IF SEARCHDATE>=INTNX('MONTH',进件日期,-3,'sameday')
	THEN QUERY_3=1;ELSE QUERY_3=0; 	
IF SEARCHDATE>=INTNX('MONTH',进件日期,-6,'sameday')
	THEN QUERY_6=1;ELSE QUERY_6=0; 	
IF SEARCHDATE>=INTNX('MONTH',进件日期,-12,'sameday')
	THEN QUERY_12=1;ELSE QUERY_12=0; 	

IF 有效查询=1;
RUN;
```



## TRANSPOSE 转置

```MYSQL
PROC SORT DATA=APPROVE_RJ_TEMP;BY 报表年月 进件日期 申请号;QUIT;
PROC TRANSPOSE DATA=APPROVE_RJ_TEMP
     OUT=APPROVE_RJ_TEMP_1 NAME=RJ_CODE_NUM;
   BY  报表年月 进件日期 申请号  否决意见详情 否决意见详情_RE_TEMP 核定贷款金额 核定业务类型   申请金额  审批意见 所属分部 所属分部RE 终审阶段 INDEX_分部否决 INDEX_其他 RJ_CODE_TEMP;
/*   COPY  报表年月 进件日期 申请号  否决意见详情 否决意见详情_RE_TEMP 核定贷款金额 核定业务类型   申请金额  审批意见 所属分部 所属分部RE 终审阶段 INDEX_分部否决 INDEX_其他 RJ_CODE_TEMP;*/
   VAR RJ_CODE_1-RJ_CODE_4;
RUN;
```

## TRANSPOSE转置，双ID作为字段名

```SAS
PROC TRANSPOSE DATA=上下限_20181116_汇总 OUT=上下限_20181116_汇总 delimiter=_;
BY 业务品种 业务类型 客户类型 客户评级   机构 版本;
VAR COL1;
ID  借款主体 _NAME_;
RUN; 
```

## 用ARRAY转置，多列转一列

```SAS
data EVAL_INNER_split;
set EVAL_INNER_split;
do i=1 to 27; /*每个组有54个数据*/
	array rj[1:27] reason_1-reason_27;
	if rj(i)^='' then do;rj_reason=rj(i);output;end;
end;
drop reason_1-reason_27 i;
run;
```



## 实现累加

```MYSQL
data a;
input date :yymmn6.
      amt  
          ;
format date yymmn6.;
cards;
201101    100 
201102    200
201103    300
201104    400
201105    500
201201    100
201202    200
201203    300
201204    400
201205    500
;

/*method one-datastep*/
data temp;
set a;
group=year(date);
run;
data result1;
  set temp;
  by group;
  if first.group then ytd_amt=amt;
  else ytd_amt+amt;
  drop group;
run;

/*method two-sql*/
proc sql;
    create table result2 as
      select distinct (a.date),a.amt, sum(b.amt) as ytd_amt
            from (select a.*,year(date) as group,monotonic() as n from a) a
                  join  (select a.*,year(date) as group ,monotonic() as n from a) b
                    on a.n ge b.n and a.group=b.group
                      group by a.group,a.n;
quit;
```

## 循环操作变量

```mysql
%macro loop(DSin);
proc sql noprint;
  select distinct name into :Var1-:Var999
  from dictionary.columns
  where libname = "%upcase(%scan(&DSin,1,'.'))"
  and memname = "%upcase(%scan(&DSin,2,'.'))";
quit;
%local i;
%do i=1 %to &sqlobs;
  %hello(&&Var&i);
%end;
%mend;
```

## 取逻辑库中的所有数据集的名称

```MYSQL
PROC SQL;
CREATE TABLE overdue_MEM_LIST AS
SELECT (MEMNAME) INTO:overdue_LIST 
SEPARATED BY " "
FROM DICTIONARY.TABLES
WHERE LIBNAME = "OVERDUE";
QUIT;
```

## 数据集变量值检查

```MYSQL
DATA LOAN_QC_CHK(DROP=申请号);
SET LOAN_QC_BASE;
RUN;

%MACRO LOOP(DSIN);
%let dsid=%sysfunc(open(&dsin,i));    
%let n=%sysfunc(attrn(&dsid,nvar));  
%let rc=%sysfunc(close(&dsid));  
%put nvar=&n;  

PROC SQL NOPRINT;
  SELECT DISTINCT NAME INTO :VAR1-:VAR&n
  FROM DICTIONARY.COLUMNS
  WHERE LIBNAME = "%UPCASE(%SCAN(&DSIN,1,'.'))"
  AND MEMNAME = "%UPCASE(%SCAN(&DSIN,2,'.'))";
QUIT;

DATA CHK_TEMP;
RUN;

%LOCAL I;
%DO I=11 %TO &n;
  PROC SQL;
  CREATE TABLE CHK_&I AS
  SELECT DISTINCT &&VAR&I FROM &DSIN;
  QUIT;

  DATA CHK_TEMP;
  MERGE CHK_TEMP CHK_&I;
  RUN;

  PROC DELETE DATA=CHK_&I;RUN;
%END;
%MEND;
%LOOP(WORK.LOAN_QC_CHK);
```

## 将数据集中变量名设定成宏变量

`proc content`

```mysql
PROC CONTENTS DATA=CONTI_VAR ORDER=VARNUM OUT=VAR_LIST(KEEP=VARNUM NAME);
RUN;

PROC SQL NOPRINT;
select put(max(VARNUM),2.) into: varnum from VAR_LIST;
quit;

proc sql NOPRINT;
SELECT distinct name INTO: Var1-: Var&varnum
FROM VAR_LIST;
QUIT;
```

## 对所有字段执行一次函数

```mysql
PROC SQL NOPRINT;
SELECT distinct name INTO: Var1-:Var3 
FROM VAR_LIST;
QUIT;

%macro temp;
%local j;
%do j=1 %to 3;
%ChcAnalyS_PLOT(CONTI_VAR,DPD_31,&&Var&j.,15,2,CHK);
%end;
%mend;

%temp;
```

## GPLOT画折线图 

`gplot` 

```MYSQL
proc gplot data=&DSChc ;
symbol1 interpol=join;
	title "&VarX";
   plot Percent_1*UpperBound;
run;
```

## SGPLOT画折线和柱状的组合图并且输出图片

`sgplot` `ods` `GRAPHICS`

```MYSQL

ods html file= "D:\work\ScoreCard_Development\images\CHK\aaa.HTM"  device=activex  style=journal  gpath="D:\work\ScoreCard_Development\images\CHK" IMAGE_DPI=300 ;
ODS GRAPHICS ON/width=3.25in imagefmt=jpg imagemap=on imagename="&VarX" border=off ;
title "ARIMA forecasts of CPI";
proc sgplot data=CHK  ;
   series x=负债比 y=Percent_1 /markers;
vbar N_1;
hbar Percent_1 BinTotal;
run;
ODS GRAPHICS/RESET;
ODS GRAPHICS off ;
ods html close;
```

## SGPLOT画Vintage图

```mysql

PROC SQL;
CREATE TABLE VINTAGE_DATA AS 
SELECT DISTINCT 账龄,发放季度,SUM(借款余额_REC*DPD_31) AS 借款余额_REC,SUM(借款发放金额) AS 借款发放金额,SUM(借款余额_REC*DPD_31)/SUM(借款发放金额) AS DPD31_PER
FROM AGING_REC   GROUP BY 账龄,发放季度;
QUIT;

DATA VINTAGE_DATA;
SET VINTAGE_DATA ;
N=_N_;
IF N>36 THEN N=.;
RUN;

PROC SQL;
CREATE TABLE VINTAGE_DATA_1 AS 
SELECT A.*,B.* FROM VINTAGE_DATA A LEFT JOIN DATA_ISU.VINTAGE_STAND B ON A.N=B.AGE
ORDER BY 账龄,发放季度;
QUIT;

ODS GRAPHICS ON /width=8in imagefmt=STATIC  imagemap=on ;
proc sgplot data=VINTAGE_DATA_1   ;

   series x=账龄 y=DPD31_PER /markers MARKERATTRS=(SYMBOL=diamondfilled) group=发放季度 NAME="DPD31" TRANSPARENCY=0.20;
   series x=N y=stand3 /markers MARKERATTRS=(SYMBOL=circle) LEGENDLABEL="stand3"   NAME="stand3" ;
   series x=N y=stand5 /markers MARKERATTRS=(SYMBOL=circle)  LEGENDLABEL="stand5" NAME="stand5";

   xaxis grid  type=discrete ;
   yaxis grid  values=(0 to 0.11 by 0.01);
     keylegend "DPD31" "stand3" "stand5";
run;
ODS GRAPHICS/RESET;
ODS GRAPHICS off ;
```



## 用FREQ 输出二联表的数据

`FREQ`

```MYSQL
	proc FREQ data=LOAN_QC_BASE ;
	 TABLE 按揭放款笔数*DPD_31/missing out=Temp_Cats;
	run;
```

## 储存表字段名

`dictionary` `columns`

```MYSQL
SELECT DISTINCT name INTO: SE_CONT_VAR SEPARATED BY ' ' FROM dictionary.columns  where memname='DSOUT_WOE';
```

## 创建空表

`SQL`

```MYSQL
PROC SQL;
CREATE TABLE CANDIDATE_MODEL (N num(20),C num(20),KS num(20));
QUIT;
```



## 运用data步将数据存成宏变量

```MYSQL
data _null_;
set lift_chk;
   call symput("bad_"||compress(_N_), bad);
run;
```

## 查看编码

```MYSQL
proc options option=locale;
run;

proc options option=encoding;
    run;
```

## DDE打开EXCEL

`DDE` `EXCEL`

```mysql
%MACRO STARTXL;
FILENAME SAS2XL DDE 'EXCEL|SYSTEM';

DATA _NULL_;
	FILE SAS2XL;
RUN;

OPTIONS NOXWAIT NOXSYNC;
%IF &SYSERR NE 0 %THEN %DO;
	X '"C:\Program Files\Microsoft Office\Office15\EXCEL.EXE"';
	DATA _NULL_;
	X=SLEEP(10);
	RUN;
	%END;
	%MEND STARTXL;
	%STARTXL;

	DATA _NULL_;
	FILE SAS2XL;
	PUT '[OPEN("D:\work\code\审批日报_20161129")]';
	RUN;
```

## [SAS程序实现日常报表的全自动化](http://bbs.pinggu.org/thread-1444265-1-1.html)

```SAS
/*  该sas文件旨在说明如何编写一次完成以后不用任何人手更改（包括更改日期、excel的格式等）的常规报表；
    如用邮件自动发送程序，则还可省略发送邮件的过程，做到一次编写，无需跟进（前提是程序经过检验完善）；
    可根据个人情况举一反三使用；
    本文来自"OUR SAS"群-"统计-小风"  */

options noxwait noxsync;

 /* 该文件为固定的报表模板，可以事先调整好单元格格式、字体颜色，事先写好其他不变的内容 */
x '"D:\报表模板.xlsx"'; 

 /* sas睡眠10秒，是为了给打开上述文件留时间 */
data _null_;
rc=sleep(10);              
run;

/* 此处设定各类时间，比如你要读取的文件是包含时间的，如test2012-05-16，就是用当天的时间、
sas程序运行的时间去得到这个"2012-05-16";另外一种是你要生成的excel文件是包含时间的，也在这里处理得出 */
data _null_;                
x=put(date()-1,yymmdd10.);   /* 比如每日运行这个程序，处理前一日的文件，就是用date()-1 */
y=substr(x,6,2)||substr(x,9,2)||"b";
z=compress(input(substr(x,6,2),best8.)||"."||input(substr(x,9,2),best8.));
call symput("path",y);
call symput("path2",z);
run;

%let log=alla_&path;        /* 我要处理的文件名就是 alla_0516b 这种形式 */

libname result "D:\test";
data temp1(compress=yes);
set result.&log;            /* 类似这样应用 */
run;

/* 中间是你数据处理的过程，省略 */

filename r1 dde 'excel|[报表模板.xlsx]自定义表名1!r5c1:r60c6' ;  /* 对某张表某些单元格进行写入 */
data _null_;
set result1;
file r1 notab linesize=2000;    /* DDE默认空格为分隔符，如果一个变量中间有空格将会分开到两个单元格，用notab即可避免，
linesize赋予一个足够大的值，则过长的变量不会错行 */
put date '09'x time '09'x source '09'x duration '09'x sql '09'x type;
run;

filename r1 dde 'excel|[报表模板.xlsx]自定义表名2!r5c1:r60c6' ;   /*  继续写入下一张表 */
data _null_;
set result2;
file r1 notab linesize=2000;
put date '09'x time '09'x source '09'x duration'09'x  sql '09'x type;
run;

filename r1 dde "excel|system";  
data _null_;
file r1;
put '[workbook.activate("自定义表名2")]';  /* 激活其中一张表 */
put '[row.height(0,"A1:A1",false,3)]';    /* 调整行高；类似这样的跟vba比较像 */
put '[workbook.activate("自定义表名1")]';
put '[row.height(0,"A1:A1",false,3)]';
x= compress('[save.as("D:\sql('||&path2||').xlsx")]');  /* 存储一个备份到某个路径，文件名为 sql(5.16).xlsx  */
put x;
y= compress('[save.as("E:\MailFile\sql('||&path2||').xlsx")]');  /* 存储到邮件文件夹，这样由邮件自动发送excel出去 */
put y;
put '[quit]';
run;

/*整个程序如上，然后txt写如下内容另存为bat文件，在windows-附件-系统工具-任务计划程序里面设置每日凌晨运行这个bat即可：
D:
cd: "D:\sas_program\sas\sasfoundation\9.2\"
sas.exe -sysin "D:\thisprogramname.sas" -altlog "D:\test\log.log"
*/

/* 上述bat第二行是你sas程序的路径，第三行表示执行的sas程序的路径和名字，然后将sas运行的日志写入到log.log中，
以便事后查看日志 */
```



## [SAS的DDE更好](http://bbs.pinggu.org/thread-136833-1-1.html) 

```SAS
有对SAS的MDX感兴趣的,可以联系我:nkwill@hotmail.com.初学者勿打扰.
/**************************************
*
*此程序实现对多个数据集输入到一个EXCEL工作簿中，
*唯一遗憾的是SHEET排序为倒序
*
********************************************/
OPTION SYMBOLGEN;
DATA A;
DO I=1 TO 100;
OUTPUT;
END;
RUN;
DATA B;
DO I=101 TO 500;
OUTPUT;
END;
RUN;
DATA C;
DO I=10 TO 500;
OUTPUT;
END;
RUN;
DATA D;
DO I=20 TO 500;
OUTPUT;
END;
RUN;
/**********以下程序实现将一个或者两个SAS数据集运用DDE输出到同一个EXCEL工作簿不同工作表中*/
options noxsync noxwait xmin; 
filename sas2xl dde 'excel|system'; 
%let tab='09'x; 
%MACRO TIME;
data _null_;
length fid rc start stop time 8;
fid=fopen('sas2xl','s');  /*此处fid=0,因为并没有start excel*/
if (fid le 0) then do;
rc=system('start excel');  /*启动excel,并保持10秒，以便EXCEL宏有足够的时间控制来自SAS程序的EXCEL宏参数*/
start=datetime();
stop=start+10;
do while (fid le 0);
fid=fopen('sas2xl','s');
time=datetime();
if (time ge stop) then fid=1;
end;
end;
rc=fclose(fid); 
run;
%mend;
%MACRO XLM;
data _null_;/*插入一个宏insert(3)*/
file sas2xl;
put '[workbook.next()]';
put '[workbook.insert(3)]'; 
run;
filename xlmacro dde 'excel|macro1!r1c1:r100c1' notab ;
data _null_;
file xlmacro;
put '=workbook.name("sheet1","第一")';
put '=workbook.name("sheet2","第二")';
put '=workbook.name("sheet3","第三")';
put '=workbook.name("sheet4","第四")';
put '=halt(true)';
put '!dde_flush';
file sas2xl;
put '[run("macro1!r1c1")]'
;
run; 
filename xlmacro clear; 
%MEND XLM;
%MACRO SHEET(N);

filename recrange dde "excel|[tt.xls]sheet&N!r4c1:r65000c1" notab;
data _null_; /***写入第一个数据集*/
set &&D&N; 
file recrange; 
put I 
;
run;
filename recrange clear;
filename recrange dde "excel|[tt.xls]sheet&N!r3c1:r3c1" notab;
data _null_; /*写入标签值*/
file recrange; 
put '手机号码'
;
run; 
filename recrange clear;

data _null_; /*制作列宽，3表示自动按原字段值调整*/
   file sas2xl; 
   put '[column.width(0,"c1:c1",false,3)]'; 
run; 
data _null_;

file sas2xl;
put '[workbook.insert(1)]';

run; 
%MEND SHEET;

%MACRO DDE1(d1,d2,d3,d4);
%TIME
data _null_;   /*创建一个新工作簿；并删除原有的缺省值3张表；建立一个新表并保存*/
file sas2xl;
put '[file.close(false)]';
put '[new(1)]';
put '[error(false)]';
put '[save.as("D:\tt")]';
run;
%SHEET(1)
%SHEET(2)
%SHEET(3)
%SHEET(4)

%XLM

data _null_;
file sas2xl;
put '[workbook.delete("macro1")]';
put '[save]';                      /*保存数据集*/
put '[file.close(false)]';        /*关闭文件*/
put '[quit]';                    /*退出EXCEL程序*/
run; 

%MEND DDE1;
%DDE1(A,B,C,D);
```



## 银行的Vintage参考线

```MYSQL
DATA DATA_ISU.VINTAGE_STAND;
	INPUT AGE STAND3 STAND5 ;
	datalines;
1 0 0   
2 0 0   
3 0.003500  0.005833 
4 0.005100  0.008500 
5 0.006500  0.010833 
6 0.009000  0.015000 
7 0.014500  0.024167 
8 0.019100  0.031833 
9 0.022600  0.037667 
10 0.023300  0.038833 
11 0.024200  0.040333 
12 0.025400  0.042333 
13 0.029200  0.048667 
14 0.030600  0.051000 
15 0.033200  0.055333 
16 0.033600  0.056000 
17 0.034200  0.057000 
18 0.036100  0.060167 
19 0.036400  0.060667 
20 0.036400  0.060667 
21 0.037300  0.062167 
22 0.038000  0.063333 
23 0.039100  0.065167 
24 0.038700  0.064500 
25 0.040500  0.067500 
26 0.040600  0.067667 
27 0.042200  0.070333 
28 0.041900  0.069833 
29 0.041600  0.069333 
30 0.041700  0.069500 
31 0.038700  0.064500 
32 0.039300  0.065500 
33 0.038700  0.064500 
34 0.040000  0.066667 
35 0.039600  0.066000 
36 0.033800  0.056333 
;
run;
```

## 删除数据集

```MYSQL
proc datasets library=work;
delete temp_freqs temp_Xcats temp_YCats;
quit;
```

## 打印前几行
`PRINT` `OBS`
```MYSQL
PROC PRINT DATA=APPR_MON.APPROVE_20170430(OBS=5);RUN;
```

## 巧用CONTENT输出表的所有变量名

`PROC CONTENTS` `short`

```MYSQL
proc contents data=ch10.telco short;
run;
```

## 将结果输出到excel

`ODS` `HTML`

```JAVA
ods listing close;
ods results off;
ods html
path='C:\Users\ThinkStation\Desktop\工作\风控数据表\'
body='result.xls';
proc tabulate data=copetime2 ;
class 申请周数 阶段名称 案件类型 入账结构 业务类型 是否直批 年份/preloadfmt;
var 工作停留时间 工作处理时间 自然停留时间 自然处理时间;
table (all='合计' 案件类型='')*(all='合计' 入账结构='')*(all='合计' 是否直批='')*(all='合计' 业务类型='')*阶段名称='',
年份=''*申请周数=''*工作停留时间*mean='' 
/misstext=' ' printmiss;
run;
ods html close;
ods results on;
ods listing;
```

## 运用EXCELXP 输出结果到excel

`ODS` `EXCELXP `

[Quick Reference for the TAGSETS.EXCELXP Tagset](http://support.sas.com/rnd/base/ods/odsmarkup/excelxp_help.html)

[Try This Demo: The ExcelXP Tagset and Microsoft Excel](http://support.sas.com/rnd/base/ods/odsmarkup/excelxp_demo.html)

[ODS and Microsoft Office Products](http://support.sas.com/rnd/base/ods/excel/)

```mysql

ods tagsets.excelxp file="D:\work\SAS_result\application_daily_report\rj_reason_&DATE_NUM..xls" style =  htmlblue  options(sheet_name="rj_all" zoom='70') ;
PROC TABULATE DATA=APPR_MON_TABULATE_all ORDER=FORMATTED classdata=class_rj;
CLASS  所属分部RE RJ_CODE 周_MARK/PRELOADFMT;
VAR 总数 否决率 CNT;
FORMAT 
	   RJ_CODE $REJECT_CODE.;
TABLE 所属分部RE=' '*RJ_CODE=' ',周_MARK=' '*(CNT='A/C'*SUM=' '*F=8.0 否决率='%'*SUM=' '*F=8.2)/BOX='全部' PRINTMISS MISSTEXT=' ';
RUN; 

ods tagsets.excelxp options(sheet_name="rj_na" zoom='70') ;
PROC TABULATE DATA=APPR_MON_TABULATE_na ORDER=FORMATTED classdata=class_rj;
CLASS  所属分部RE RJ_CODE 周_MARK/PRELOADFMT;
VAR 总数 否决率 CNT;
FORMAT 
	   RJ_CODE $REJECT_CODE.;
TABLE 所属分部RE=' '*RJ_CODE=' ',周_MARK=' '*(CNT='A/C'*SUM=' '*F=8.0 否决率='%'*SUM=' '*F=8.2)/BOX='NA' PRINTMISS MISSTEXT=' ';
RUN; 

ods tagsets.excelxp close;
```

## 结果输出到excel

```MYSQL
# 方法1
libname  myxls  EXCEL "D:\work\SAS_result\month_portfolio_monitoring\&LAST_MON_D_NUM._vintage_31.xls";
DATA myxls.VINTAGE_DATA_31(DROP=ORDER);
FORMAT 维度1 $CHAR100. 维度2 $CHAR100. 维度3 $CHAR100. V1 $CHAR100. V2 $CHAR100.  _NAME_ $CHAR100. V3 $CHAR100.;
RETAIN 维度1	维度2	维度3	V1	V2 _NAME_ V3;
SET VIN_Q31 VINTAGE_CITY_Q31 VINTAGE_CITY_B31 VINTAGE_CITY_B_ALL31 VINTAGE_CLIENTY_Q31 VINTAGE_SURVEY_Q31
VINTAGE_BUSITY_ALL31 VINTAGE_BUSITY_Q31 VINTAGE_CLIENTY_SURVEY_Q31  VINTAGE_BUSI_CLIEN_Q31;
IF V1=' ' THEN V1='合计';
IF V2=' ' THEN V2='合计';
IF V3=' ' THEN V3='合计';
RUN;
libname myxls clear;

# 方法2
proc export data=final outfile="D:\报表\mianqian.csv" dbms=csv label replace;
run;
```

## 关于输出log文件,限制log文件输出

```SAS
http://bbs.pinggu.org/thread-955059-1-1.html

options nonotes nosource nosource2 errors=10; 

options notes source source2 errors=20; 

```

## 去重

```SAS
proc sort data=test1 nodup out=aa2;by x y;run;
```

## update一次更新多个字段

```MYSQL
PROC SQL;
update tb_county t
   set (t.prov_name, t.city_name, t.xs_mc) = (select t.prov_name,
                                                     t.city_name,
                                                     t.xs_mc
                                                from tb_yzbm t
                                               where t.postcode = '230000')
 where t.xs_code = '2300';
 QUIT;
```

## 建立索引

```MYSQL
DATA class (index=studentID firstname);

PROC DATASETS library=<libref>;

            modify <dataset>;

           index create <variable>;

quit;

PROC SQL;
            create unique index studentID on class(studentID);
 quit;
```

## 字符的日期时间格式转换

```MYSQL
data aa;
a=dhms(input(scan('2011-09-01 14:20:31',1,' '), yymmdd10.),0,0, input(scan('2011-09-01 14:20:31',2,' '), time8.));
b=input('2011-09-01 14:20:31', b8601dt.);
format a b b8601dt.;
run;
```

## SAS 随机抽样

```SAS
PROC SURVEYSELECT
    DATA=  * 输入数据集;
    OUT=  * 输出数据集;
    METHOD=  * 抽样方法;
    SAMPSIZE=  * 选择项指定需要抽样的样本量;
    SAMPRATE= * ;
    REP=
    SEED=
    NOPRINT;
    ID variable; 指定抽取的样本所保留的源数据集变量
    STRATA variables;  指定分层变量
    CONTROL variables; 控制变量
    SIZE variables; 不等概抽样指标变量
RUN;
```

## 修改数据集变量或者label

> 需要修改数据集变量的label和format格式时，还是通过proc datasets过程进行修改效率比较快，它不需要记录进入pdv，比起data步更有效率。

```mysql

data test;           
     set sashelp.class;           
      label weight="体重（斤）";            
      format weight best6.2;    
run;      

proc datasets library=sashelp;            
     modify class;            
     label weight="体重（斤）";           
     format weight best6.2;   
run;    
quit;
```

## 报错终止程序

```SAS
%if &SYSERR > 6 %then %goto STOPLOG
```

## SAS数据集取前N条记录

```SAS
方法一：
data temp1;
set sashelp.air(firstobs=n obs=n/obs=n);
run;
/*firstobs.n < obs.n*/
方法二：
proc sql;
create table temp2 as 
select * from sashelp.air(firstobs=n obs=n/obs=n);
quit;
/*firstobs.n < obs.n*/
方法三：
proc sql inobs=n;
create table temp3 as
select * from sashelp.air;
quit;
方法四：
proc sql outobs=n;
create table temp4 as
select * from sashelp.air;
quit;
```

## 将一个数据集拆分成多个数据集

```SAS
data chk chk1;
set qry_record_report(obs=1000);
if rid^=' ' then output  chk;
if rid=' ' then output  chk1;
run;
```

## 直连本地库

```MYSQL
proc sql;
/*…SQL Pass-Through statements referring to mydbone…*/
connect to mysql  as result
(user=leiwy  password=eLison_6800 server='192.168.9.133'
        database=riskgkdb port=3306 DBCONINIT='set names gbk');

disconnect from result;
quit;


proc sql;
  connect to mysql
   (user=root  password=sas123 server=localhost
    database=world port=3306);
  create table b as select * from connection to mysql
  (select * from city);

   execute(create table cc as select * from aa
  )by mysql;

  disconnect from mysql;

 quit;
 
 # 直接从数据库连好后再取数
 proc sql;
connect to mysql  as database
(user=leiwy  password=eLison_6800 server='192.168.9.133' database=mddb port=3306 DBCONINIT='set names gbk');
create table ccard1_1 as select * from  connection  to database
(
select * from cr_loan_business_detail where etl_date='2019-01-13' limit 100
)
  ;
disconnect from database;
quit;

```

## SAS 自动执行

```bash
set "year=%date:~0,4%"
set "month=%date:~5,2%"
set "day=%date:~8,2%"
set "hour=%time:~0,2%"
set "minute=%time:~3,2%"
set "second=%time:~6,2%"

"C:\Program Files\SASHome\SASFoundation\9.3\sas.exe" -sysin "D:\SASProjects\data_analysis\exampleproject\20190505_month_report\取申请客户评分\01 取申请客户评分.sas" -log "D:\SASProjects\data_analysis\exampleproject\20190505_month_report\取申请客户评分\取申请客户评分%year%%month%%day%_%hour%%minute%%second%.log" -print "D:\SASProjects\data_analysis\exampleproject\20190505_month_report\取申请客户评分\取申请客户评分%year%%month%%day%_%hour%%minute%%second%.lst"
```



# VBA

## 隐藏工作表

```VB
　Sub MySheetsHide()
　　　MsgBox "第一次隐藏工作表sheet1"
　　　Worksheets("sheet1").Visible = False
  　　MsgBox "显示工作表sheet1"
　　　Worksheets("sheet1").Visible = True
  End Sub
```

## 另存XLS文件

```VB
Sub test()
     Dim fn$
     fn = Application.InputBox("请输入输出的文件名", "提示", , , , , , 2)
     fn = "d:\work\04月\好_" & fn & ".xls"
  ActiveWorkbook.SaveAs fn, 56 '51 xlsx  56 xls
  
End Sub
```

## 指定列变量顺序

```MYSQL
/*method 1:*/

data A;
  retain  agegr1，agegr1n,   agrgr2，agegr2n ,    agegr3 ，agegr3n;
  set B;
run;

/*method 2:*/

proc sql noprint ;
  create table a as
    select  agegr1，agegr1n,   agrgr2，agegr2n ,    agegr3 ，agegr3n
      from B ;
quit;
```

## 遍历工作表6种方法

```visual basic
'''1.
Sub 遍历工作表()
For Each sh In Worksheets    '数组
    sh.Select
    Call 设置A1格式
Next
End Sub
--------------------------------------
'''2.
Sub 遍历工作表()
For Each sh In Worksheets    '数组
    if sh.name <> 表名1 and sh.name <>表名 2 then
            sh.Select
            Call 设置A1格式
    end if 
Next
End Sub
--------------------------------------
'''3.
Sub 循环工作表()
    For Each sh In Worksheets
        If sh.Index > 2 Then    '限定工作表范围
       sh.Select
              Call 设置A1格式
        End If
    Next
End Sub
--------------------------------------
'''4.
Sub 遍历工作表()
For Each sh In Worksheets    '数组
    If sh.Name Like "*" & "表" & "*" Then     '如果工作表名称包含“表”
        sh.Select
        Call 设置A1格式
    End If
Next
End Sub
--------------------------------------
'''5.
Sub 遍历工作表()
For Each sh In Worksheets    '数组
    If Not sh.Name Like "*" & "表" & "*" Then     '如果工作表名称不包含“表”
        sh.Select
        Call 设置A1格式
    End If
Next
End Sub
--------------------------------------
'''6.
Sub 遍历工作表()
For Each sh In Worksheets    '数组
    If sh.Name <> "价格表" And sh.Name <> "人员表" Then    '指定不参与循环的工作表名称，可一个或多个，自行增减
        sh.Select
        Call 设置A1格式
    End If
Next
End Sub

```



## 快速提取sheet表标题

```VB
    Sub createmulu()
    For i = 1 To Sheets.Count
    Cells(i, 1) = Sheets(i).Name
    Next
    End Sub
```



# Drools

## accumulate实现累加

```java
//Example 1
$rc : RuleResultContainer()
$laf : LoanApplyFact($adate : applyDate, $amount : applyAmount)
$sumAmount : Number(doubleValue > Math.max(100000, $amount))
    from accumulate(
        ExecutePublicFact(filingDate != null, executeAmount != null,
            DateUtils.getMonth(filingDate, $adate) > 24, $eamount : executeAmount),
        sum($eamount)
    )
  
//Example 2  
  $num : Number(intValue >= 4) from accumulate (
    CreditTxnLoanDetail(loan24Status != null, loan24Status.length() ==24,
                        $loan24status : RepaymentStatusUtils.sumOverdueNum(loan24Status,0,24)),
    sum($loan24status)
  )
```

## list用法

```java
List<CivilRefereeInstrumentsFact> instrumentsFacts = lawxpDTO.getCivilRefereeInstrumentsFacts();
if (instrumentsFacts != null) {
    for (CivilRefereeInstrumentsFact instrumentsFact : instrumentsFacts) {
        insert(instrumentsFact);
    }
}
```

## 符合条件的贷款笔数相加

```SAS
$list1 : List() from collect (
CreditTxnCCDetail(accountStaus != "正常")
)

$list2 : List() from collect (
CreditTxnQuasiCCDetail(accountStaus != "正常")
)

QualityControlInfo(hasLoan == false)

Number(intValue() == 0) from $list1+$list2
```



## 同一机构放款金额累加

`set` `function` `map`

```java
$details : Set() from collect (
            CreditTxnLoanDetail($institute == inst, LoanUtils.isCreditLoan(guarType, loanType, term),
        )

Number(doubleValue < 100000) from SumByInst($details)

function Double SumByInst(Set details) {
    Map<String, Double> countMap = new HashMap<>();

    System.out.println(details.size());

    for (java.lang.Object obj : details) {
        CreditTxnLoanDetail detail = (CreditTxnLoanDetail) obj;
        Double count = countMap.get(detail.getInst());
        if (count == null) {
            count = 0.0d;
        }
        countMap.put(detail.getInst(), count + detail.getIssueAmount());
    }

    System.out.println(countMap);

    double maxSum = 0;
    for (java.lang.Double value : countMap.values()) {
        if (value != null && value > maxSum) {
            maxSum = value;
        }
    }

    return maxSum;
}
```

## 计算逾期总数/数字的个数

```JAVA
    /**
     * 计算逾期总数
     *
     * @param loan24Status 24月还款状态
     * @param minLastMonth 抽取区间（小）
     * @param maxLastMonth 抽取区间（大）
     * @return 逾期总数
     */
    public static int sumOverdueNum(String loan24Status, int minLastMonth, int maxLastMonth) {
        String needStatus = extractNeedStatus(loan24Status, minLastMonth, maxLastMonth);
        Matcher matcher = NUMBER_PATTERN.matcher(needStatus);
        int sumNum = 0;
        while (matcher.find()) {
            sumNum += 1;
        }
        return sumNum;
    }
```

## 求和大于某个数

```JAVA
        $num3 : Number() from accumulate (
            CreditTxnQuasiCCDetail(loan24Status != null, loan24Status.length() ==24,
                $loan24status2 : RepaymentStatusUtils.sumOverdueNum(loan24Status,0,24)),
                sum($loan24status2)
        )

        $num_sum : Number(intValue > 4) from $num1+$num2+$num3
```

## Function 的用法(匹配查询记录和详情中的机构)

```JAVA

function Integer count(List records, List details) {

    Integer count = 0;

    for(Object record1: records){
        CreditApprovalQueryRecord record = (CreditApprovalQueryRecord)record1;
        for(Object detail1: details){
          CreditTxnLoanDetail detail = (CreditTxnLoanDetail)detail1;
          if (detail.getInst().contains(record.getOperator())) {
            count += 1;
            break;
          }
        }
    }

    return count;
}
```

## 直接用定义的变量计算后设定规则

```JAVA
rule "ruleSZS_006"

    when
        $rc : RuleResultContainer()
        $num1 : Number() from accumulate (
            CreditTxnLoanDetail(loan24Status != null, loan24Status.length() ==24,
                $loan24status : RepaymentStatusUtils.sumOverdueNum(loan24Status,0,24)),
                sum($loan24status)
        )
        $num2 : Number() from accumulate (
            CreditTxnCCDetail(loan24Status != null, loan24Status.length() ==24,
                $loan24status1 : RepaymentStatusUtils.sumOverdueNum(loan24Status,0,24)),
                sum($loan24status1)
        )
        $num3 : Number() from accumulate (
            CreditTxnQuasiCCDetail(loan24Status != null, loan24Status.length() ==24,
                $loan24status2 : RepaymentStatusUtils.sumOverdueNum(loan24Status,0,24)),
                sum($loan24status2)
        )

        $num_sum : Number(intValue > 4) from $num1+$num2+$num3

    then
//    System.out.println($num_sum);
        $rc.addResult("ruleSZS_006：石嘴山银行否决规则", RuleResult.REJECT);
end

```

## accumulate 进行累加

```JAVA
        $num : Number(intValue >= 3) from accumulate (
            CreditTxnLoanDetail(loan24Status != null, loan24Status.length() ==24
                , loanType in ("个人住房贷款","个人住房公积金贷款","个人商用房（包括商住两用）贷款")
                , $loan24status : RepaymentStatusUtils.sumOverdueNum(loan24Status,0,24)),
                sum($loan24status)
        )

```

## 去重计数

```JAVA
        $instset : Set() from accumulate (
            CreditTxnLoanDetail(
                 $inst:inst
                ,DateUtils.getLogicMonth(loanDate, $reDate) <= 12
                ,(inst not contains "银行" || inst not contains "住房公积")
                )
                ,collectSet($inst)
        )

        $list : Number(intValue >= 7) from $instset.size() ;
```

## 打印数值

```JAVA
 System.out.println(drools.getRule().getName() + " -- " + $loaninstset + " --- " +$ccinstset+"---");
```

## 新建一个逾期列表

```
OverdueDetail overdueDetail1 = new OverdueDetail();
ArrayList<OverdueDetail> overdueDetails = new ArrayList<>();
overdueDetails.add(overdueDetail1);
```

## 设置日期

```
Calendar calendar = Calendar.getInstance();
calendar.set(2018,Calendar.NOVEMBER,16);
```



# Python

## 多环境管理

[Anaconda使用教程](http://www.afox.cc/archives/390)

[Anaconda多环境多版本python配置指导](https://www.jianshu.com/p/d2e15200ee9b)



## jupyter notebook 格式设置

```
>jt -t oceans16 -nfs 10 -tfs 11 -ofs 10
```



## pandas 无法读入中文名称的文件

```PYTHON
df = pd.read_csv(path,encoding = 'gbk', engine='python')
```

## get dummy variables

```python
values = np.random.randn(10)
#RESULT
# array([ 0.4082, -1.0481, -0.0257, -0.9884, 0.0941, 1.2627, 1.29 ,
# 0.0824, -0.0558, 0.5366])

values

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

pd.get_dummies(pd.cut(values, bins))
#RESULT
# (0.0, 0.2] (0.2, 0.4] (0.4, 0.6] (0.6, 0.8] (0.8, 1.0]
# 0 0 0 1 0 0
# 1 0 0 0 0 0

```

## 重命名dataframe列名

```python
#将Age转成dummy
Age=BURN1000['AGE']
Age_Dummy=pd.get_dummies(pd.cut(BURN1000['AGE'], bin_age))
Age_Dummy.columns=['Age_Q1','Age_Q2','Age_Q3','Age_Q1']
Age_Dummy.head()
```

```python
df = df.rename(columns={'$a': 'a', '$b': 'b'})
# OR
df.rename(columns={'$a': 'a', '$b': 'b'}, inplace=True)
```

## 连接mysql(以mysql-connector-python为例)

```python
# 导入MySQL驱动:
>>> import mysql.connector
# 注意把password设为你的root口令:
>>> conn = mysql.connector.connect(user='root', password='password', database='test', use_unicode=True)
>>> cursor = conn.cursor()
# 创建user表:
>>> cursor.execute('create table user (id varchar(20) primary key, name varchar(20))')
# 插入一行记录，注意MySQL的占位符是%s:
>>> cursor.execute('insert into user (id, name) values (%s, %s)', ['1', 'Michael'])
>>> cursor.rowcount
1
# 提交事务:
>>> conn.commit()
>>> cursor.close()
# 运行查询:
>>> cursor = conn.cursor()
>>> cursor.execute('select * from user where id = %s', ('1',))
>>> values = cursor.fetchall()
>>> values
[(u'1', u'Michael')]
# 关闭Cursor和Connection:
>>> cursor.close()
True
>>> conn.close()
```

##  连接mysql(以sqlalchemy为例)

```PYTHON
连接mysql(以mysql-connector-python为例)
```

## 看每个变量的情况
```PYTHON
print(EvalQryRecord.info(), '\n')
for i in EvalQryRecord.columns:
    if len(EvalQryRecord[i].drop_duplicates()) < 50:
        print("*"*50, '\n', EvalQryRecord[i].value_counts() )
```

## lambda

```MYSQL
EvalQryRecord.loc[lambda df: pd.isnull(df["input_time"])]
```

## 重复标记

> Pandas提供了duplicated、Index.duplicated、drop_duplicates函数来标记及删除重复记录
>
> duplicated函数用于标记Series中的值、DataFrame中的记录行是否是重复，重复为True，不重复为False
>
> pandas.DataFrame.duplicated(self, subset=None, keep='first')
>
> pandas.Series.duplicated(self, keep='first')
>
> 其中参数解释如下：
>
> subset：用于识别重复的列标签或列标签序列，默认所有列标签
>
> keep=‘frist’：除了第一次出现外，其余相同的被标记为重复
>
> keep='last'：除了最后一次出现外，其余相同的被标记为重复
>
> keep=False：所有相同的都被标记为重复

```PYTHON
import numpy as np  
import pandas as pd   
#标记DataFrame重复例子  
df = pd.DataFrame({'col1': ['one', 'one', 'two', 'two', 'two', 'three', 'four'], 'col2': [1, 2, 1, 2, 1, 1, 1],  
                   'col3':['AA','BB','CC','DD','EE','FF','GG']},index=['a', 'a', 'b', 'c', 'b', 'a','c'])  
#duplicated(self, subset=None, keep='first')  
#根据列名标记  
#keep='first'  
df.duplicated()#默认所有列，无重复记录  
df.duplicated('col1')#第二、四、五行被标记为重复  
df.duplicated(['col1','col2'])#第五行被标记为重复  
#keep='last'  
df.duplicated('col1','last')#第一、三、四行被标记重复  
df.duplicated(['col1','col2'],keep='last')#第三行被标记为重复  
#keep=False  
df.duplicated('col1',False)#Series([True,True,True,True,True,False,False],index=['a','a','b','c','b','a','c'])  
df.duplicated(['col1','col2'],keep=False)#在col1和col2列上出现相同的，都被标记为重复  
type(df.duplicated(['col1','col2'],keep=False))#pandas.core.series.Series  
#根据索引标记  
df.index.duplicated()#默认keep='first',第二、五、七行被标记为重复  
df.index.duplicated(keep='last')#第一、二、三、四被标记为重复  
df[df.index.duplicated()]#获取重复记录行  
df[~df.index.duplicated('last')]#获取不重复记录行  
#标记Series重复例子  
#duplicated(self, keep='first')  
s = pd.Series(['one', 'one', 'two', 'two', 'two', 'three', 'four'] ,index= ['a', 'a', 'b', 'c', 'b', 'a','c'],name='sname')  
s.duplicated()  
s.duplicated('last')  
s.duplicated(False)  
#根据索引标记  
s.index.duplicated()  
s.index.duplicated('last')  
s.index.duplicated(False)  
```



### 正则表达式提取字段

```python
str_temp = '''select tb1.serialno, tb1.customer_name, tb1.birthday, tb1.qry_date, tb1.qry_result, tb1.loan_max_amt, tb1.status, 
                    tb1.cert_id, tb1.eval_result, tb2.rid, tb2.report_time
                    from mddb.cr_credit_report_qry as tb1 left join mddb.pbcc_report_basics as tb2
								on tb1.customer_name = tb2.query_name and tb1.birthday = tb2.date_of_birth
                where qry_source = "友金云贷"  and (ABS(datediff(tb1.qry_date, tb2.report_time))<15 or report_time is null)'''
dt_yunLoanCreditResult = loadDailyData.f_load_bySql(str_temp)
dt_yunLoanCreditResult[["是征信否决", "否决原因"]] = dt_yunLoanCreditResult["eval_result"].str.extract(
                        '{"reject":([\s\S]*?),[\s\S]*?resultList[\s\S]*?"subRejectCodeList":\[([\s\S]*?)\]}', expand=False)
dt_yunLoanCreditResult["是征信否决"] = dt_yunLoanCreditResult["是征信否决"].replace({"false":0,'true':1}).astype('int')
dt_yunLoanCreditResult = dt_yunLoanCreditResult.sort_values(["serialno", "是征信否决"]).\
                            drop_duplicates(["serialno"], keep = 'last').set_index("serialno", drop=False) #按照是否征信否决排序， 如果是征信否决，就会取到是征信否决那一条

sr_eval_resultExtract = dt_yunLoanCreditResult["eval_result"].str.findall('(rule[\s\S]*?_[0-9]{1,5})[\s\S]*?status":"([\s\S]*?)"')

for key, value in sr_eval_resultExtract.iteritems():
    for i in value:
        lst_neiPingResult.append([key, i[0],i[1]])
dt_neiPingResult = pd.DataFrame(lst_neiPingResult , columns = ["serialno", "预警码", "类型"])
dt_neiPingResult = pd.merge(dt_neiPingResult, dt_yunLoanCreditResult[["serialno", "qry_date"]], on='serialno', ).\
                        assign(qry_date = lambda df:df["qry_date"].apply(pd.to_datetime), how="left")
dt_neiPingResult["征信返回时间戳"] = dt_neiPingResult["qry_date"].apply(lambda x: str(x.year) +
                                            "第{}周".format('0'+str(x.weekofyear) if x.weekofyear<10 else x.weekofyear))
dt_neiPingResult["是否有查征信"] = dt_neiPingResult["serialno"].isin(dt_yunLoanCreditResult.query("rid == rid")["serialno"])
dt_resultDnyReason = tabulate(dt_neiPingResult, ["是否有查征信","征信返回时间戳", "预警码", "类型"], lambda grp: pd.Series({"计数":grp.shape[0]})).\
                        unstack(1).unstack(2)
```

### 查找重复值

```PYTHON
names = df.name.value_counts()
names[names > 1]
```

# SQL

## 增

1. 增加部分列的值，其他默认NULL

```MYSQL
INSERT INTO train30day.input_base(id,MID) VALUES('0','120')
```

2. 增加全部列的值

```MYSQL
INSERT INTO train30day.input_base VALUES ('0','120','男')
```

3. 通过外部表导入，其他列默认NULL

```MYSQL
INSERT INTO  train30day.input_base(MID,SEX,AGE,DEGREE)
SELECT MID,SEX,AGE,DEGREE
FROM train30day.input_base WHERE MENBER_GRADE>5
```



## 删(DELETE/TRUNCATE/DROP)

1. 删除age为NULL的记录

   ```MYSQL
   DELETE FROM input WHERE age is NULL;
   ```

2. 清空input 表的数据

   ```MYSQL
   TRUNCATE TABLE input
   ```

3. 删除input 表

   `DROP`

   ```MYSQL
   DROP TABLE TABLE_NAME [RESTRICT|CASCADE]
   #RESTRICT选项 若表被视图或者约束引用，删除失败
   #CASCADE选项 删除相关表和视图
   DROP TABLE INPUT
   ```

   

## 改

1. 新增列

   ```MYSQL
   ALTER TABLE input add UUID VARCHAR(255) COMMENT '数据库唯一表示' FIRST;
   ALTER TABLE input add UUID VARCHAR(255) COMMENT '数据库唯一表示' AFTER id;

   ```

2. 删除列

   ```MYSQL
   ALTER TABLE train30day.input_base DROP UUID

   ```

3. 单表更新数据

   ```MYSQL
   UPDATE train30day.input_base SET menber_grade=5 where TRIM(sex)='女';
   ```

   

4. 多表关联更新数据

   ```MYSQL
   UPDATE train30day.input_base AS s1,train30day.input AS s2 SET s1.degree=s2.degree 
   WHERE s1.mid=s2.mid AND s1.age=s2.age

   ```

5. 对字段名的修改

   ```MYSQL
   ALTER TABLE train30day.input_base CHANGE member_grade grade TINYINT(10);
   ```

   

6. 对数据类型的修改

   ```MYSQL
   #方式1
   ALTER TABLE train30day.input_base CHANGE member_grade member_grade TINYINT(1);
   
   #方式2
   ALTER TABLE  train30day.input_base CHANGE MODIFY member_grade TINYINT(1);
   ```

7. 增加主键

    ```mysql
    ALTER TABLE riskgkdb.wy_ys_30min_list
    ADD  PRIMARY KEY (n)
    ```



## 查

1. 对表结构的查询

   ```MYSQL
   DESC train30day.input_base
   ```

   


## 修改字符集 - ALTER TABLE

`ALTER TABLE`

```MYSQL
ALTER TABLE TABLE_NAME [MODIFY] [COLUMN COLUMN_NAME][DATATYPE|NULL NOT NULL]
[RESTRICT|CASCADE]
[DROP] [CONSTRAINT CONSTRAINT_NAME]
[ADD] [COLUMN] COLUNM DEFINITION
```



 ```mysql
#修改数据库字符集
ALTER DATABASE db_name DEFAULT CHARACTER SET character_name [COLLATE ...];

#把表默认的字符集和所有字符列（CHAR,VARCHAR,TEXT）改为新的字符集
ALTER TABLE tbl_name CONVERT TO CHARACTER SET character_name [COLLATE ...];
ALTER TABLE logtest CONVERT TO CHARACTER SET utf8 COLLATE utf8_general_ci;

#只是修改表的默认字符集
ALTER TABLE tbl_name DEFAULT CHARACTER SET character_name [COLLATE...];
ALTER TABLE logtest DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;

#修改字段的字符集
ALTER TABLE tbl_name CHANGE c_name c_name CHARACTER SET character_name [COLLATE ...];
ALTER TABLE logtest CHANGE title title VARCHAR(100) CHARACTER SET utf8 COLLATE utf8_general_ci;

#查看数据库编码
SHOW CREATE DATABASE db_name;

#查看表编码
SHOW CREATE TABLE tbl_name;

#查看字段编码
SHOW FULL COLUMNS FROM tbl_name;
 ```
## 查看Mysql的编码格式

```MYSQL
SHOW VARIABLES LIKE "%char%";
```

[MySQL 数据库 - 大小写, collate, collation, 校对规则, 字符集](http://www.oursg.com/Article/Details/9B30680333DFB2F0DDB1A2CA3D43F3BF#TMd_AutoId_6)

## 创建表

```MYSQL
CREATE TABLE user_invest_detail(
    id BIGINT(30) NOT NULL AUTO_INCREMENT COMMENT'自增id' ,
    investid BIGINT(30) DEFAULT NULL COMMENT'投资订单id',    
    userid BIGINT(30) DEFAULT NULL COMMENT'会员id',
    invest_time VARCHAR(40) DEFAULT NULL COMMENT'投资时间',
    invest_amount INT(10) DEFAULT NULL COMMENT'投资金额',
    invest_product VARCHAR(40) DEFAULT NULL COMMENT'投资的产品',
    invest_type VARCHAR(40) DEFAULT NULL COMMENT'投资类型0:新手投资,1:短期投资,2:年化投资',
    PRIMARY KEY (id)
    )ENGINE=INNODB AUTO_INCREMENT=101 DEFAULT CHARSET=utf8;
```

## 改密码

```mysql
alter user xiongwei identified by "5n1ivEpy*UkM"; #修改标红的为新密码
```

## 存储过程

```mysql
CREATE OR REPLACE PROCEDURE "loan"."prc_test01"("p_etl_date" date)
 AS $BODY$
<<label>>

/*
功能描述：pgsql开发模块
开发人员：陈玉金
当前版本：v1.1
修改时间：2018.11.02
特别说明：
*/

--定义变量
declare v_rowcnt int;
declare v_second int;
declare v_btime timestamp;
declare v_etime timestamp;
declare v_step int default 0;
declare v_errno varchar(10);
declare v_errmsg varchar(100);

begin
--判断输入参数
if p_etl_date >= current_date then
raise notice 'p_etl_date必须小于当前日期：%',p_etl_date;
exit label;
end if;

--这里可以插入日志表--

--阶段1
v_step:=1;
v_btime := clock_timestamp();
--数据处理
delete from t6411_sum where etl_date=p_etl_date;

--诊断信息
get diagnostics v_rowcnt := row_count;
v_etime := clock_timestamp();
v_second := extract(epoch from (v_etime - v_btime));
raise notice '[ S:%, T:%, R:% ]',v_step,v_second,v_rowcnt;

--阶段2
v_step:=2;
v_btime := clock_timestamp();
--数据处理
with t1 as (select p_etl_date,f03,sum(f16),count(*) from t6411 group by f03)
insert into t6411_sum select * from t1 ;

--诊断信息
get diagnostics v_rowcnt := row_count;
v_etime := clock_timestamp();
v_second := extract(epoch from (v_etime - v_btime));
raise notice '[ S:%, T:%, R:% ]',v_step,v_second,v_rowcnt;


--完成
v_step:=100;
--这里可以插入日志表--
raise notice '[ 成功 ]';

--异常处理
	exception 
	when others then
	  --这里可以插入日志表--
		raise notice '[ 失败：step-%, % ]',v_step,SQLERRM;
	
end;
$BODY$
  LANGUAGE plpgsql
```

``` mysql
USE [master]
GO
/****** Object:  StoredProcedure [dbo].[UP_CreateDB]    Script Date: 05/07/2015 13:33:26 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create procedure [dbo].[UP_CreateDB](@v_dbname nvarchar(255),@v_path nvarchar(255),@v_dbfilenum int)
/************************************************************
  名称: UP_CreateDB
  功能描述：根据传入的参数，创建数据库
  入参描述：
    @v_dbname 需要创建的数据库名称
    @v_path 创建数据库的物理路径
    @v_dbfilenum 创建数据库的物理文件数目
  
  修订记录：
  版本号     编辑时间     编辑人      修改描述
  1.0.0     2015.04.17    贺亮      创建此存储过程
 
  调用示例:
      exec master.dbo.UP_CreateDB 'GTA_MFL1_TRDTIME_201503','D:\MFL1_TRDSEC',3
  ***********************************************************/
as
begin
    declare @v_sql varchar(8000),
            @v_num int
    set @v_sql = NULL
    set @v_num = 1
    --数据库已存在则返回
    if exists(select * from master.dbo.sysdatabases where name = @v_dbname)
        begin
            print '数据库已存在！'
            return
        end
     else
        begin
        
            set @v_sql = 'CREATE DATABASE '+@v_dbname+' 
                          ON PRIMARY
                               (
                                 NAME = '''+@v_dbname+'_01'',
                                 FILENAME = '''+@v_path+'\'+@v_dbname+'_01.GTA'',
                                 SIZE = 100MB,
                                 FILEGROWTH = 10%
                                ),'
            while @v_num < @v_dbfilenum
                begin
                    set @v_num = @v_num + 1
                    set @v_sql = @v_sql +'
                             FILEGROUP data_'+RIGHT('0'+CAST(@v_num as nvarchar(10)),2)+'
                               (
                                 NAME = '''+@v_dbname+'_'+RIGHT('0'+CAST(@v_num as nvarchar(10)),2)+''',
                                 FILENAME = '''+@v_path+'\'+@v_dbname+'_'+RIGHT('0'+CAST(@v_num as nvarchar(10)),2)+'.GTA'',
                                 SIZE = 100MB,
                                 FILEGROWTH = 10%
                                ),'  
                end
                
            set @v_sql = LEFT(@v_sql,LEN(@v_sql)-1)+'
                          LOG ON 
                               (
                                 NAME = '''+@v_dbname+'_log'',
                                 FILENAME = '''+@v_path+'\'+@v_dbname+'_log.ldf'',
                                 SIZE = 1MB,
                                 FILEGROWTH = 10%
                                 --MAXSIZE = 2024MB 
                                )'                      
            exec (@v_sql)
        end
end
```

``` mysql

Delimiter //
		alter procedure riskgkdb.myproc()
			Begin
				Select count(app_num)  from mddb.cr_loan_repay_detail
				where etl_date='2017-10-22';
			End
			//
Delimiter;

call riskgkdb.myproc()

-- IN参数
DELIMITER //
  CREATE PROCEDURE riskgkdb.in_param(IN p_in int)
    BEGIN
    SELECT p_in;
    SET p_in=2;
    SELECT p_in;
    END;
    //
DELIMITER ;
#调用
SET @p_in=1;
CALL riskgkdb.in_param(@p_in);
SELECT @p_in;

-- OUT参数 
DELIMITER //
  CREATE PROCEDURE riskgkdb.out_param(OUT p_out int)
    BEGIN
      SELECT p_out;
      SET p_out=2;
      SELECT p_out;
    END;
    //
DELIMITER ;
#调用
SET @p_out=1;
CALL riskgkdb.out_param(@p_out);
SELECT @p_out;
```

# git

## 删除本地仓库

```git
    find . -name ".git" | xargs rm -Rf
```

## Git global setup

```git
git config --global user.name "lei"
git config --global user.email "leiwy@yonyou.com"
```

## Create a new repository

```
git clone git@gitlab.example.com:leiwy/sas_data_analysis_pub.git
cd sas_data_analysis_pub
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

## Existing folder

```
cd existing_folder
git init
git remote add origin git@gitlab.example.com:leiwy/sas_data_analysis_pub.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

## Existing Git repository

```bash
cd existing_repo
git remote rename origin old-origin
git remote add origin git@gitlab.example.com:leiwy/sas_data_analysis_pub.git
git push -u origin --all
git push -u origin --tags
```


## [intellj Idea git ignore文件的.idea不起作用解决](https://www.cnblogs.com/sunlightlee/p/5803862.html)

```bash
# 先清除掉已经跟踪的文件，然后再添加到gitignore
git rm -r --cached 目录名称/文件名称
```

## [Git-常用命令及说明](https://frainmeng.github.io/2015/10/29/Git-常用命令及使用说明/)

## [git中文乱码解决方案](https://blog.csdn.net/xl_lx/article/details/78223349)
```bash
# 修改etc\gitconfig：
[gui]
encoding = utf-8
[i18n]
commitencoding = utf-8
[svn]
pathnameencoding = utf-8
说明：打开 Git 环境中的中文支持。pathnameencoding设置了文件路径的中文支持。
```

# markdown

## 行内与独行

1. 行内公式：将公式插入到本行内，符号：`$公式内容$`，如：$xyz$
2. 独行公式：将公式插入到新的一行内，并且居中，符号：`$$公式内容$$`，如：$$xyz$$

## 上标、下标与组合

1. 上标符号，符号：`^`，如：$x^4$
2. 下标符号，符号：`_`，如：$x_1$
3. 组合符号，符号：`{}`，如：${16}_{8}O{2+}_{2}$

## 汉字、字体与格式

1. 汉字形式，符号：`\mbox{}`，如：$V_{\mbox{初始}}$
2. 字体控制，符号：`\displaystyle`，如：$\displaystyle \frac{x+y}{y+z}$
3. 下划线符号，符号：`\underline`，如：$\underline{x+y}$
4. 标签，符号`\tag{数字}`，如：$\tag{11}$
5. 上大括号，符号：`\overbrace{算式}`，如：$\overbrace{a+b+c+d}^{2.0}$
6. 下大括号，符号：`\underbrace{算式}`，如：$a+\underbrace{b+c}_{1.0}+d$
7. 上位符号，符号：`\stacrel{上位符号}{基位符号}`，如：$\vec{x}\stackrel{\mathrm{def}}{=}{x_1,\dots,x_n}$

## 占位符

1. 两个quad空格，符号：`\qquad`，如：$x \qquad y$
2. quad空格，符号：`\quad`，如：$x \quad y$
3. 大空格，符号`\`，如：$x \  y$
4. 中空格，符号`\:`，如：$x : y$
5. 小空格，符号`\,`，如：$x , y$
6. 没有空格，符号``，如：$xy$
7. 紧贴，符号`\!`，如：$x ! y$

## 定界符与组合

1. 括号，符号：`（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)`，如：$（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)$
2. 中括号，符号：`[]`，如：$[x+y]$
3. 大括号，符号：`\{ \}`，如：${x+y}$
4. 自适应括号，符号：`\left \right`，如：$\left(x\right)$，$\left(x{yz}\right)$
5. 组合公式，符号：`{上位公式 \choose 下位公式}`，如：${n+1 \choose k}={n \choose k}+{n \choose k-1}$
6. 组合公式，符号：`{上位公式 \atop 下位公式}`，如：$\sum_{k_0,k_1,\ldots>0 \atop k_0+k_1+\cdots=n}A_{k_0}A_{k_1}\cdots$

## 四则运算

1. 加法运算，符号：`+`，如：$x+y=z$
2. 减法运算，符号：`-`，如：$x-y=z$
3. 加减运算，符号：`\pm`，如：$x \pm y=z$
4. 减甲运算，符号：`\mp`，如：$x \mp y=z$
5. 乘法运算，符号：`\times`，如：$x \times y=z$
6. 点乘运算，符号：`\cdot`，如：$x \cdot y=z$
7. 星乘运算，符号：`\ast`，如：$x \ast y=z$
8. 除法运算，符号：`\div`，如：$x \div y=z$
9. 斜法运算，符号：`/`，如：$x/y=z$
10. 分式表示，符号：`\frac{分子}{分母}`，如：$\frac{x+y}{y+z}$
11. 分式表示，符号：`{分子} \voer {分母}`，如：${x+y} \over {y+z}$
12. 绝对值表示，符号：`||`，如：$|x+y|$

## 高级运算

1. 平均数运算，符号：`\overline{算式}`，如：$\overline{xyz}$
2. 开二次方运算，符号：`\sqrt`，如：$\sqrt x$
3. 开方运算，符号：`\sqrt[开方数]{被开方数}`，如：$\sqrt[3]{x+y}$
4. 对数运算，符号：`\log`，如：$\log(x)$
5. 极限运算，符号：`\lim`，如：$\lim^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
6. 极限运算，符号：`\displaystyle \lim`，如：$\displaystyle \lim^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
7. 求和运算，符号：`\sum`，如：$\sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
8. 求和运算，符号：`\displaystyle \sum`，如：$\displaystyle \sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
9. 积分运算，符号：`\int`，如：$\int^{\infty}_{0}{xdx}$
10. 积分运算，符号：`\displaystyle \int`，如：$\displaystyle \int^{\infty}_{0}{xdx}$
11. 微分运算，符号：`\partial`，如：$\frac{\partial x}{\partial y}$
12. 矩阵表示，符号：`\begin{matrix} \end{matrix}`，如：$\left[ \begin{matrix} 1 &2 &\cdots &4\5 &6 &\cdots &8\\vdots &\vdots &\ddots &\vdots\13 &14 &\cdots &16\end{matrix} \right]$

## 逻辑运算

1. 等于运算，符号：`=`，如：$x+y=z$
2. 大于运算，符号：`>`，如：$x+y>z$
3. 小于运算，符号：`<`，如：$x+y<z$
4. 大于等于运算，符号：`\geq`，如：$x+y \geq z$
5. 小于等于运算，符号：`\leq`，如：$x+y \leq z$
6. 不等于运算，符号：`\neq`，如：$x+y \neq z$
7. 不大于等于运算，符号：`\ngeq`，如：$x+y \ngeq z$
8. 不大于等于运算，符号：`\not\geq`，如：$x+y \not\geq z$
9. 不小于等于运算，符号：`\nleq`，如：$x+y \nleq z$
10. 不小于等于运算，符号：`\not\leq`，如：$x+y \not\leq z$
11. 约等于运算，符号：`\approx`，如：$x+y \approx z$
12. 恒定等于运算，符号：`\equiv`，如：$x+y \equiv z$

## 集合运算

1. 属于运算，符号：`\in`，如：$x \in y$
2. 不属于运算，符号：`\notin`，如：$x \notin y$
3. 不属于运算，符号：`\not\in`，如：$x \not\in y$
4. 子集运算，符号：`\subset`，如：$x \subset y$
5. 子集运算，符号：`\supset`，如：$x \supset y$
6. 真子集运算，符号：`\subseteq`，如：$x \subseteq y$
7. 非真子集运算，符号：`\subsetneq`，如：$x \subsetneq y$
8. 真子集运算，符号：`\supseteq`，如：$x \supseteq y$
9. 非真子集运算，符号：`\supsetneq`，如：$x \supsetneq y$
10. 非子集运算，符号：`\not\subset`，如：$x \not\subset y$
11. 非子集运算，符号：`\not\supset`，如：$x \not\supset y$
12. 并集运算，符号：`\cup`，如：$x \cup y$
13. 交集运算，符号：`\cap`，如：$x \cap y$
14. 差集运算，符号：`\setminus`，如：$x \setminus y$
15. 同或运算，符号：`\bigodot`，如：$x \bigodot y$
16. 同与运算，符号：`\bigotimes`，如：$x \bigotimes y$
17. 实数集合，符号：`\mathbb{R}`，如：`\mathbb{R}` 
18. 自然数集合，符号：`\mathbb{Z}`，如：`\mathbb{Z}` 
19. 空集，符号：`\emptyset`，如：$\emptyset$

## 数学符号

1. 无穷，符号：`\infty`，如：$\infty$
2. 虚数，符号：`\imath`，如：$\imath$
3. 虚数，符号：`\jmath`，如：$\jmath$
4. 数学符号，符号`\hat{a}`，如：$\hat{a}$
5. 数学符号，符号`\check{a}`，如：$\check{a}$
6. 数学符号，符号`\breve{a}`，如：$\breve{a}$
7. 数学符号，符号`\tilde{a}`，如：$\tilde{a}$
8. 数学符号，符号`\bar{a}`，如：$\bar{a}$
9. 矢量符号，符号`\vec{a}`，如：$\vec{a}$
10. 数学符号，符号`\acute{a}`，如：$\acute{a}$
11. 数学符号，符号`\grave{a}`，如：$\grave{a}$
12. 数学符号，符号`\mathring{a}`，如：$\mathring{a}$
13. 一阶导数符号，符号`\dot{a}`，如：$\dot{a}$
14. 二阶导数符号，符号`\ddot{a}`，如：$\ddot{a}$
15. 上箭头，符号：`\uparrow`，如：$\uparrow$
16. 上箭头，符号：`\Uparrow`，如：$\Uparrow$
17. 下箭头，符号：`\downarrow`，如：$\downarrow$
18. 下箭头，符号：`\Downarrow`，如：$\Downarrow$
19. 左箭头，符号：`\leftarrow`，如：$\leftarrow$
20. 左箭头，符号：`\Leftarrow`，如：$\Leftarrow$
21. 右箭头，符号：`\rightarrow`，如：$\rightarrow$
22. 右箭头，符号：`\Rightarrow`，如：$\Rightarrow$
23. 底端对齐的省略号，符号：`\ldots`，如：$1,2,\ldots,n$
24. 中线对齐的省略号，符号：`\cdots`，如：$x_1^2 + x_2^2 + \cdots + x_n^2$
25. 竖直对齐的省略号，符号：`\vdots`，如：$\vdots$
26. 斜对齐的省略号，符号：`\ddots`，如：$\ddots$

## 希腊字母

| 字母 | 实现       | 字母 | 实现       |
| ---- | ---------- | ---- | ---------- |
| A    | `A`        | α    | `\alhpa`   |
| B    | `B`        | β    | `\beta`    |
| Γ    | `\Gamma`   | γ    | `\gamma`   |
| Δ    | `\Delta`   | δ    | `\delta`   |
| E    | `E`        | ϵ    | `\epsilon` |
| Z    | `Z`        | ζ    | `\zeta`    |
| H    | `H`        | η    | `\eta`     |
| Θ    | `\Theta`   | θ    | `\theta`   |
| I    | `I`        | ι    | `\iota`    |
| K    | `K`        | κ    | `\kappa`   |
| Λ    | `\Lambda`  | λ    | `\lambda`  |
| M    | `M`        | μ    | `\mu`      |
| N    | `N`        | ν    | `\nu`      |
| Ξ    | `\Xi`      | ξ    | `\xi`      |
| O    | `O`        | ο    | `\omicron` |
| Π    | `\Pi`      | π    | `\pi`      |
| P    | `P`        | ρ    | `\rho`     |
| Σ    | `\Sigma`   | σ    | `\sigma`   |
| T    | `T`        | τ    | `\tau`     |
| Υ    | `\Upsilon` | υ    | `\upsilon` |
| Φ    | `\Phi`     | ϕ    | `\phi`     |
| X    | `X`        | χ    | `\chi`     |
| Ψ    | `\Psi`     | ψ    | `\psi`     |
| Ω    | `\v`       | ω    | `\omega`   |

## 公式

https://blog.csdn.net/dss_dssssd/article/details/82692894

