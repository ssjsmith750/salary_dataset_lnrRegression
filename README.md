# salary_dataset_lnrRegression


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

     

data = pd.read_csv("/content/Salary_dataset.csv")
data
     
Unnamed: 0	YearsExperience	Salary
0	0	1.2	39344.0
1	1	1.4	46206.0
2	2	1.6	37732.0
3	3	2.1	43526.0
4	4	2.3	39892.0
5	5	3.0	56643.0
6	6	3.1	60151.0
7	7	3.3	54446.0
8	8	3.3	64446.0
9	9	3.8	57190.0
10	10	4.0	63219.0
11	11	4.1	55795.0
12	12	4.1	56958.0
13	13	4.2	57082.0
14	14	4.6	61112.0
15	15	5.0	67939.0
16	16	5.2	66030.0
17	17	5.4	83089.0
18	18	6.0	81364.0
19	19	6.1	93941.0
20	20	6.9	91739.0
21	21	7.2	98274.0
22	22	8.0	101303.0
23	23	8.3	113813.0
24	24	8.8	109432.0
25	25	9.1	105583.0
26	26	9.6	116970.0
27	27	9.7	112636.0
28	28	10.4	122392.0
29	29	10.6	121873.0

data1=data.drop(['Unnamed: 0'],axis=1) # To drop unnecesary column
     

data1
     
YearsExperience	Salary
0	1.2	39344.0
1	1.4	46206.0
2	1.6	37732.0
3	2.1	43526.0
4	2.3	39892.0
5	3.0	56643.0
6	3.1	60151.0
7	3.3	54446.0
8	3.3	64446.0
9	3.8	57190.0
10	4.0	63219.0
11	4.1	55795.0
12	4.1	56958.0
13	4.2	57082.0
14	4.6	61112.0
15	5.0	67939.0
16	5.2	66030.0
17	5.4	83089.0
18	6.0	81364.0
19	6.1	93941.0
20	6.9	91739.0
21	7.2	98274.0
22	8.0	101303.0
23	8.3	113813.0
24	8.8	109432.0
25	9.1	105583.0
26	9.6	116970.0
27	9.7	112636.0
28	10.4	122392.0
29	10.6	121873.0

data1.describe() # to find the summary
     
YearsExperience	Salary
count	30.000000	30.000000
mean	5.413333	76004.000000
std	2.837888	27414.429785
min	1.200000	37732.000000
25%	3.300000	56721.750000
50%	4.800000	65238.000000
75%	7.800000	100545.750000
max	10.600000	122392.000000

data1.YearsExperience.quantile(0.75)# check the 75th quantile
     
7.8

data1.YearsExperience.quantile(0.25)
     
3.3000000000000003

iqr = data1.YearsExperience.quantile(0.75) - data.YearsExperience.quantile(0.25)
iqr
     
4.5

upper_thershold = data1.YearsExperience.quantile(0.75) + (1.5*iqr) #q3
lower_thershold = data1.YearsExperience.quantile(0.25) - (1.5*iqr) #q1

upper_thershold,lower_thershold
     
(14.55, -3.4499999999999997)
NO outliers checking for null values


data1.isnull().sum()
     
YearsExperience    0
Salary             0
dtype: int64
NO Null Values also Delete the duplicate values


data1 = data1.drop_duplicates()
data1.shape # used to delete the duplicate values
     
(30, 2)
#ETA

data1.dtypes
     
YearsExperience    float64
Salary             float64
dtype: object

data1.plot(x='YearsExperience',y = 'Salary', style='o')
plt.title('Salary According to Years of Experience')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
     

There is a relationship Btw the Feature and Target also linear relationship can be seen


data1.corr()
     
YearsExperience	Salary
YearsExperience	1.000000	0.978242
Salary	0.978242	1.000000
There is a linear relationship between YearsExperience and Salary


data.Salary.values
     
array([ 39344.,  46206.,  37732.,  43526.,  39892.,  56643.,  60151.,
        54446.,  64446.,  57190.,  63219.,  55795.,  56958.,  57082.,
        61112.,  67939.,  66030.,  83089.,  81364.,  93941.,  91739.,
        98274., 101303., 113813., 109432., 105583., 116970., 112636.,
       122392., 121873.])
preparing the data


X = data1.loc[:, ['YearsExperience']].values # select all rows and columns
y = data1.loc[:, ['Salary']].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state= 8)
     

y_test
     
array([[105583.],
       [ 39344.],
       [ 54446.],
       [ 81364.],
       [ 67939.],
       [ 39892.],
       [121873.],
       [ 46206.]])

X_train.shape, X_test.shape
     
((22, 1), (8, 1))

X_test
     
array([[ 9.1],
       [ 1.2],
       [ 3.3],
       [ 6. ],
       [ 5. ],
       [ 2.3],
       [10.6],
       [ 1.4]])
Modelling

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # predicted scores = m* YearsExperience + c
regressor.fit(X_train, y_train)
     
LinearRegression()

print(regressor.intercept_) # c
     
[23185.78510289]

print(regressor.coef_) # slope - m
     
[[9824.98564969]]

regressor .predict([[9]])
     
array([[111610.65595008]])

y_pred = regressor.predict(X_test)
y_pred
     
array([[112593.15451505],
       [ 34975.76788251],
       [ 55608.23774686],
       [ 82135.69900102],
       [ 72310.71335133],
       [ 45783.25209717],
       [127330.63298958],
       [ 36940.76501245]])

df = pd.DataFrame([{'Actual': y_test, 'Predicted': y_pred}],index=[1])
df
     
Actual	Predicted
1	[[105583.0], [39344.0], [54446.0], [81364.0], ...	[[112593.15451504805], [34975.76788251415], [5...

regressor.predict([[12]])
     
array([[141085.61289914]])

from sklearn import metrics 
print('R2- SCORE:', metrics.r2_score(y_test,y_pred))
regressor.score(X_test,y_test) 
     
R2- SCORE: 0.9644656253169838
0.9644656253169838
