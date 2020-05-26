"""
This dataset is created for prediction of Graduate Admissions from an Indian perspective.

Content
The dataset contains several parameters which are considered important during the application for Masters Programs.
The parameters included are :

GRE Scores ( out of 340 )
TOEFL Scores ( out of 120 )
University Rating ( out of 5 )
Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
Undergraduate GPA ( out of 10 )
Research Experience ( either 0 or 1 )
Chance of Admit ( ranging from 0 to 1 )
Acknowledgements
This dataset is inspired by the UCLA Graduate Dataset. The test scores and GPA are in the older format.
The dataset is owned by Mohan S Acharya.
"""

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
df=pd.read_csv("datasets/Admission_Predict.csv")
print(df.head())
print(df.isnull().sum())
print(df.columns)
df.columns = (['serial_no','GRE','TOEFL','university_rating','SOP','LOR','CGPA','research','COA'])
df = df.drop('serial_no',axis=1)
df.head()
sns.distplot(df. GRE,bins=20, kde=False,color='purple')
#plt.show()
con_feat = ['GRE','TOEFL','CGPA', 'COA']
sns.pairplot(df[con_feat])
#plt.show()
y=df['COA']
x=df.drop(['COA'],axis=1)
cat_feature = [col for col in df.columns if df[col].dtypes == "O"]
print(cat_feature)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=10)
model= RandomForestRegressor()
model.fit(x_train,y_train)
predict=model.predict(x_test)
rmse=metrics.mean_squared_error(y_test,predict)
print(rmse)
model1 = LinearRegression()
model1.fit(x_train,y_train)
predict1 = model1.predict(x_test)
rmse1 =metrics.mean_squared_error(y_test,predict1)                   
print(rmse1)
#cv1 =cross_val_score(model1,cv=5)

