#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# ##### A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So BoomBikes want to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. 
# 
# 
# 

# ## Business Requirement 
# 
# ##### We are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market.

# ## Steps Fallowed in solving this case study 
# 
# 1. Reading, understanding and visualizing the data
# 2. Preparing the data for model training (train-test split, rescaling)
# 3. Training the model
# 4. Residual analysis
# 5. Prediction and evaluation of the test set
# 

# ### Step 1: Reading, understanding and visualizing the data

# In[129]:


# Importing all the required libraries

import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

# Importing stats model 
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Importing sklearn 
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[130]:


#Reading the dataset

boombike = pd.read_csv("day.csv")


# #### Data Inspection

# In[131]:


boombike.head()


# In[132]:


boombike.shape


# In[133]:


boombike.describe()


# In[134]:


# Get to know about type of Data
boombike.info()


# ###### Since we have the required infromation about date, month and year in separate columns and instant is not required . 
# ##### After reading the data dropping instant and dteday from the data frame. 

# In[135]:


boombike = boombike.drop(['instant','dteday'],axis =1)


# Dropping causal and registered column since count we will get from cnt column and we can have only one target in the data

# In[136]:


boombike.head()


# #### Visualisation of Data 

# In[137]:


#visualising the numeric variables of the dataset using pairplot 
sns.pairplot(boombike, x_vars=['temp','atemp','hum','windspeed'],y_vars='cnt')
plt.show()


# with the above plots we can say there is strong correlation between the count variable and temperature, humidity and windspeed.

# In[138]:


# Let's see the correlation between variables.
plt.figure(figsize=(20, 12))
sns.heatmap(boombike.corr(), cmap="viridis", annot = True)
plt.show()


# In the Heatmap above darkest color shows the negative corelation and lightest color shows positive corelation. 
# 

# In[139]:


# Box Plots for categorical variables

plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
sns.boxplot(x='season',y='cnt',data=boombike)
plt.subplot(3,3,2)
sns.boxplot(x='yr',y='cnt',data=boombike)
plt.subplot(3,3,3)
sns.boxplot(x='mnth',y='cnt',data=boombike)
plt.subplot(3,3,4)
sns.boxplot(x='holiday',y='cnt',data=boombike)
plt.subplot(3,3,5)
sns.boxplot(x='weekday',y='cnt',data=boombike)
plt.subplot(3,3,6)
sns.boxplot(x='weathersit',y='cnt',data=boombike)

plt.show()


# ###### Observations from the above box plots 
# 
# 1. People have rented bikes more in Summer and Fall 
# 2. More bikes were rented in 2019 compared to 2018 
# 3. Bikes were rented more in Aug, Sep and Oct 
# 4. In clear weather bike rentals is more and in Heavy rain people have not taken bike for rent 

# ### Step 2: Data Preparation for modeling 

# In[140]:


#dropping the variables atemp as its similar to temp 

boombike = boombike.drop(['atemp'], axis=1)


# In[141]:


#Dropping causal and registered since we can have only one target in the data 
boombike = boombike.drop(['casual','registered'],axis =1)


# In[142]:


boombike.head(5)


# ###### From data dictionalry we can see that
# 
#     categorical variables :
#         1. season : season (1:spring, 2:summer, 3:fall, 4:winter).
#         2. yr : year (0: 2018, 1:2019).
#         3. mnth : month ( 1 to 12).
#         4. weekday : day of the week.
#         5. holiday : 0 and 1
#         6. workingday : 0 and 1
#         7. weathersit : 
#                 - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#                 - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#                 - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#                 - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog.
#         

# In[143]:


#mapping categorical variables with their subcategories to help with visualization analysis 
boombike['season'] = boombike.season.map({1: 'spring', 2: 'summer',3:'fall', 4:'winter' })
boombike['mnth'] = boombike.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
boombike['weathersit'] = boombike.weathersit.map({1: 'Clear',2:'Mist + Cloudy',3:'Light Snow',4:'Heavy rain'})
boombike['weekday'] = boombike.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})

boombike.head()


# 
# 
# 

# Creating dummy variables for the variables for non binary categorical variables like month, weekday, weathersit, seasons.

# In[144]:


month = pd.get_dummies(boombike.mnth, drop_first=True)
weekday = pd.get_dummies(boombike.weekday, drop_first=True)
weathersit = pd.get_dummies(boombike.weathersit, drop_first=True)
season = pd.get_dummies(boombike.season, drop_first=True)


# In[145]:


# adding dummy variable to boombike data frame 

boombike = pd.concat([boombike,month, weekday, weathersit, season], axis=1)
boombike.head()


# In[146]:


# dropping the catergorical variables season,mnth,weekday,weathersit from boombike as we have added the dummies to boombikes 
boombike.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)
boombike.head()


# #### Splitting into Train and Test 

# In[147]:


boombike_train, boombike_test = train_test_split(boombike, train_size=0.7, random_state=100)


# In[148]:


boombike_train.shape


# In[149]:


boombike_test.shape


# #### Scaler 

# In[150]:


scaler = MinMaxScaler()


# In[151]:


#normalize temp, atemp, hum and windspeed numerical columns 

scaler_variables = ['hum', 'windspeed', 'temp', 'cnt']
boombike_train[scaler_variables] = scaler.fit_transform(boombike_train[scaler_variables])


# In[152]:


boombike_train.describe()


# ### Step 3 : Building a Model 

# ###### I am using hybrid approch to Build the model 

# In[153]:


# Creating feature and target variable datasets in y_train and X_train

y_train = boombike_train.pop('cnt')
X_train = boombike_train


# In[154]:


# Adding a constant 
X_train_sm=sm.add_constant(X_train)

# Creating First Model 
lr = sm.OLS(y_train, X_train_sm).fit()

# Displying Params 
lr.params


# In[155]:


lr.summary()


# #### Using RFE:
# 
# We have found out the R-squared value of 85.3% from manual approch  for feature selection. We will now using the automated approach for selecting the values required for building the most optimized models and dropping the values which are redundant for our model building approach. 
# 
# We will now use RFE for automated approach, along with VIF to determine the variables to drop.

# In[156]:


#creating the RFE object
lm = LinearRegression()
lm.fit(X_train, y_train)

#setting feature selection variables to 15 which is half of the feature set available
rfe = RFE(lm,n_features_to_select = 15) 
#fitting rfe ofject on our training dataset
rfe = rfe.fit(X_train, y_train)


# In[157]:


#checking the elements selected and the ones rejected in a list after rfe
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[158]:


#getting the selected feature variables in one one variable
rfe_vars = X_train.columns[rfe.support_]
rfe_vars


# In[159]:


#building model using selected RFE variables
#creating training set with RFE selected variables
X_train_rfe = X_train[rfe_vars]


# In[160]:


#adding constant to training variable
X_train_rfe = sm.add_constant(X_train_rfe)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train, X_train_rfe).fit()

#params
lr.params


# In[161]:


#summary of model
lr.summary()


# In[162]:


#calculating the VIF of the model
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# ###### Conditions used for dropping a feature variable 
#     1. High p-value, high VIF : definitely drop
#     2. High-Low:
#             High p, low VIF : remove these first, then build model again and check VIF again
#             Low p, high VIF : remove these later
#     3. Low p-value, low VIF : keep variable
#     
#     I have considered cut-off for P-value as 0.05 and Cut-off for VIF as 5 for my model building 

# based on the above condition in our model currently Holiday is having higher P-value of 0.386 and workday,sat,sun and spring are above VIF cut-off level . for next model I am removing "Holiday" feature which satify condition 2 . 

# ##### Model 2: Building the model by dropping Holiday 

# In[163]:


# Dropping holiday variable
X_train_rfe.drop(columns='holiday', inplace=True)

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr2 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr2.summary()


# In[164]:


#calculating the VIF for the model2
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# After model 2 all the features have p-value less then the cut-off value but Hum,Temp, Workingday still has VIF greater then Cut-off value. But we R2 and adjusted R2 remains same .
# 
# So i am building one more model by dropping "Hum" which is having highest VIF . 

# ##### Model 3: Building model by dropping hum 

# In[165]:


# Dropping hum variable
X_train_rfe.drop(columns='hum', inplace=True)

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr3 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr3.summary()


# In[166]:


#calculating the VIF for the model3
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# After this mode there is a drop in r2 and adj.r2 by 0.006 which is very minimal ,
# but still working day and temp has VIF more then cutoff value . So dropping workingday which is having highest VIF in next model 

# ##### Model 4: Building by dropping workingday 

# In[167]:


# Dropping Workingday variable
X_train_rfe.drop(columns='workingday', inplace=True)

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr4 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr4.summary()


# In[168]:


#calculating the VIF for the model4
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# with the removal of workingday r2 and adj.r2 dropped by 0.005 , but VIF has reduced significantly. Building new model by Removing Sat which is having higher P-value then cut-off 
# 

# ##### Model 5: Building model  by removing Sat 

# In[169]:


# Dropping Workingday variable
X_train_rfe.drop(columns='Sat', inplace=True)

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr5 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr5.summary()


# In[170]:


#calculating the VIF for the model5
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# Temp is slightly above VIF cut-off so lets try by removing temp 

# ##### Model 6: Building by dropping temp

# In[171]:


# Dropping Workingday variable
X_train_rfe.drop(columns='temp', inplace=True)

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr6 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr6.summary()


# ###### if we drop temp R2 and Adj.R2 will drop by 10% and increase in P-value of the features , so i am adding back temp and trying to build new model by removing  July which is having slightly higher P-value  

# In[172]:


# adding temp again 
X_train_rfe['temp'] = X_train['temp']


# In[173]:


#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)


# ##### Model 7: Building new model by removing July 

# In[174]:


# Dropping Workingday variable
X_train_rfe.drop(columns='July', inplace=True)

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr7 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr7.summary()


# In[175]:


#calculating the VIF for the model7
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# By removing July all the remaining features VIF came below cut off with R2 and Adj.R2 values of 83%. Lets try to improve the model by adding Oct which is next in the ranking list of features.

# #### Model 8: Building model by adding 'Oct' . 

# In[176]:


# Adding sunday variable
X_train_rfe['Oct']=X_train['Oct']

# Adding a constant to X_train_rfe
X_train_rfe=sm.add_constant(X_train_rfe)

# Creating a LR object which we will use to fit the line.
lr8 = sm.OLS(y_train, X_train_rfe).fit()

#Checking the summary
lr8.summary()


# In[177]:


#calculating the VIF for the model7
#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)

# the independent variables set
X = X_train_rfe

# VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X.columns

# calculating VIF for each feature
vif['VIF'] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# With the addition of Oct there is not much improvement in R2 and Adj.R2 but it has increased P-value of Oct so dropping Oct.

# In[178]:


X_train_rfe.drop(columns='Oct', inplace=True)


# In[179]:


## Building the final model 
X_train_rfe=sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()

lm.summary()


# ### Step 4: Residual analysis

# In[180]:


# Adding a constant to X_train_rfe

y_train_pred = lm.predict(X_train_rfe)


# In[181]:


res = (y_train - y_train_pred)


# In[182]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# ### Step5 : Predictions & Evaluation on the Test Set

# In[183]:


scaler_variables


# In[184]:


#Applying scaling on test data
boombike_test[scaler_variables] = scaler.fit_transform(boombike_test[scaler_variables])


# In[185]:


boombike_test.describe()


# In[186]:


# Splitting target and feature variables
y_test = boombike_test.pop('cnt')
X_test = boombike_test[:]


# In[187]:


#Dropping constant variable

X_train_rfe.drop(columns='const', inplace=True)


# In[188]:


# Creating X_test_new dataframe by dropping variables from X_test
X_test_sm = X_test[X_train_rfe.columns]

# Adding a constant variable 
X_test_sm = sm.add_constant(X_test_sm)


# In[189]:


# Making predictions

y_pred = lm.predict(X_test_sm)


# #### Model Evaluation 

# In[190]:


print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_pred))


# In[194]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
sns.regplot(x=y_test, y=y_pred, ci=52, fit_reg=True, line_kws={"color": "red"})
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 16)               
plt.xlabel('y_test', fontsize = 15)                          
plt.ylabel('y_pred', fontsize = 15) 
plt.show()


# ######  R-squared value of 83.33 % on train data and 79.35% on test data.

# In[192]:


train_mse = (mean_squared_error(y_true=y_train, y_pred=y_train_pred))
test_mse = (mean_squared_error(y_true=y_test, y_pred=y_pred))
print('Mean squared error of the train set is', train_mse)
print('Mean squared error of the test set is', test_mse)


# ###### Mean Squared Error close to 0 on the training dataset, meaning our model is able to correctly predict all variances in the data.
# ###### MSE for test dataset is 0.01 which is close to zero,our model is able perform similarly on unknown data sets too.
# 

# ### Summary 

# 1. The R-squared value of the train set is 83.33% & on the test set has a value of 79.35%, which suggests that
#    our model explains the variance quite accurately on the test set and thus we can conclude that it is a 
#    good model.
# 2. MES is almost 0 on both the training and testing datasets, which suggests that the variance is accurately
#    predicted on the test set. The p-values and VIF were used to select the significant variables. 
#    RFE was also conducted for automated selection of variables.
# 3. bike demands for the BoomBikes is company is dependent on the temperature
# 4. More bike rentals will be on Winter compared to other seasons 
# 5. Months of Aug to October has higher rentals 
