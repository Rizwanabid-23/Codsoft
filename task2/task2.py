import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load and preprocess the data
df_train_data = pd.read_csv('D:/Codsoft/task2/datasets/fraudTrain.csv')
df_train_data.drop('trans_date_trans_time', axis=1, inplace=True)

plt.hist(df_train_data['is_fraud'])
plt.show()

train_x,test_x=train_test_split(df_train_data,test_size=0.3)
pred_var=['amt','zip','lat','long','city_pop','unix_time','merch_lat','merch_long']
outcome_var='is_fraud'

model=LogisticRegression()
model.fit(train_x[pred_var],train_x[outcome_var])
prediction=model.predict(train_x[pred_var])
print("Variance score: {}".format(model.score(train_x[pred_var],train_x[outcome_var])))

input=[[21.92,32960,27.633,-80.4031,105638,1371939613,27.339943,-81.199244]]  #fraud input
output=model.predict(input) 
print(input,np.round_(output,0))


#the dataset for this code can be found here:https://www.kaggle.com/datasets/kartik2112/fraud-detection