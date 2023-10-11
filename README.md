# Boombikes-LinearRegression-Assignment 
> Linear Regression performed on the Boombikes bike rental dataset as part of an assignment


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)


## General Information
- Multiple linear regression is performed on the dataset.
- The project is done as part of coursework in the Machine Learning module. 
- We are trying to find the number of rentals issued from the company based on numerous independent values such as temperature, weather, humidity, holiday, etc. 
- The Boombikes bike rental dataset is being used. 


## Conclusions
- The R-squared value of the train set is 83.33 whereas the test set has a value of 79.33% which suggests that our model broadly explains the variance quite accurately on the test set and thus we can conclude that it is a good model.

- Our developed model's mean squared error is almost 0 on both the training and testing datasets which suggests that the variance is accurately predicted on the test set. The p-values and VIF were used to select the significant variables. RFE was also conducted for automated selection of variables.

- The major steps included in the python notebook are data interpretation, data visualisation, data pre-processing, model training, feature selection, residual analysis, model evaluation on the test set. 

- Concepts such as p-value, VIF, RFE were used and model building was done using statsmodels library


## Technologies Used
- pandas
- seaborn
- matplotlib
- statsmodels
- sci-kit learn
- numpy



## Contact
Created by [@Darshan-BV] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->