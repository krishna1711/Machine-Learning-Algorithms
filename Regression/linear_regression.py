##########################################################################################################################
#########################                      Linear Regression                                 #########################
##########################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
#Read a csv File
df =pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

### Un Comment To Check the Plots
'''#Histogram Plotting
cdf.hist()
plt.show()

##########Scatter Plot Of All columns with target variable################
plt.scatter(cdf['ENGINESIZE'],cdf['CO2EMISSIONS'])
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()
##########################################################################
plt.scatter(cdf['FUELCONSUMPTION_COMB'],cdf['CO2EMISSIONS'])
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.show()

##########################################################################
plt.scatter(cdf['CYLINDERS'],cdf['CO2EMISSIONS'])
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.show()
##########################################################################
'''

mask = np.random.rand(len(df)) < 0.8 ###rand(n): generates n values with random numbers btw(0,1)
train = cdf[mask]                    ###training data set
test = cdf[~mask]                    ###test data set

#############Training dataset Scatter Plot ################################
plt.scatter(train['ENGINESIZE'],train['CO2EMISSIONS'])
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Training DataSet')
plt.show()


################Sklearn to Model the data##################################
from sklearn import linear_model as lm
regr = lm.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print(f"Coefficients: {regr.coef_}")
print(f"Intercept: {regr.intercept_}")


################Plot the line between x1 and y^ of train data################
plt.scatter(train['ENGINESIZE'],train['CO2EMISSIONS'],color='orange')
plt.plot(train_x,(regr.intercept_[0] + regr.coef_[0][0] * train_x),color='black')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Trained Data Model')


#############Predicting using test data set##################################
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_pred = regr.predict(test_x)

print("Mean Absolute error: %.2f" %(np.mean(np.absolute(test_y_pred - test_y))))
print("MSE Mean Square Error: %.2f" % (np.mean((test_y_pred-test_y)**2)))
'''
    R2Score is the R-squared value
    Represents how close the data are to be fitted regression line.
    Higher R-squared better model.
    Best possible score is 1.0
    -ve value shows or means more worse
'''
print("R2Score: %.2f" % r2_score(test_y_pred,test_y))
#cdf[np.array[True,False]]
