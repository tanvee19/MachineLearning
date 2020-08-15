
"""
Q. (Create a program that fulfills the following specification.)
Female_Stats.Csv

Female Stat Students

 

Import The Female_Stats.Csv File

The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,

Column2 = Student’s Guess At Her Mother’s Height, And

Column 3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

 

Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are Significant For A Students’ Height Or Not
When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.

"""


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 

dataset = pd.read_csv('Female_Stats.csv') 
features = dataset.iloc[:, 1:3].values  
labels = dataset.iloc[:, 0:1].values 
regressor = LinearRegression()  
regressor.fit(features, labels) 
#mean_original=np.mean(regressor.predict(features))
mean_mom=list(regressor.coef_)[0][0]
mean_dad=list(regressor.coef_)[0][1]

print("If we increase mom height by one then increment in height : ",mean_mom)
print("If we increase dad height by one then increment in height : ",mean_dad)


"""

Q. (Create a program that fulfills the following specification.)
bluegills.csv

How is the length of a bluegill fish related to its age?

In 1981, n = 78 bluegills were randomly sampled from Lake Mary in Minnesota. The researchers (Cook and Weisberg, 1999) measured and recorded the following data (Import bluegills.csv File)

Response variable(Dependent): length (in mm) of the fish

Potential Predictor (Independent Variable): age (in years) of the fish

    How is the length of a bluegill fish best related to its age? (Linear/Quadratic nature?)
    What is the length of a randomly selected five-year-old bluegill fish? Perform polynomial
    regression on the dataset.

NOTE: Observe that 80.1% of the variation in the length of bluegill fish is reduced by 
taking into account a quadratic function of the age of the fish.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('bluegills.csv')
features = dataset.iloc[:, 0:1].values
labels = dataset.iloc[:, 1].values


from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(features, labels)
plt.scatter(features, labels)
plt.plot(features, lin_reg_1.predict(features), color = 'red')

x = [5]
x=np.array(x)
print(lin_reg_1.predict(x.reshape(1,-1)))

from sklearn.preprocessing import PolynomialFeatures
poly_object = PolynomialFeatures(degree = 5)
features_poly = poly_object.fit_transform(features)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poly, labels)
print(lin_reg_2.predict(poly_object.transform(x.reshape(1,-1))))


plt.scatter(features, labels)
plt.plot(features, lin_reg_2.predict(poly_object.transform(features)), color = 'black')


s1 = lin_reg_1.score(features, labels)
s2 = lin_reg_2.score(features_poly, labels)

print(s1)
print(s2)
if s1>s2:
    print("Linear is best")
elif s2>s1:
    print("Quadratic is best")
    


'''

import statsmodels.api as sm
features = sm.add_constant(features)


features_opt = features[:, [0,1]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
'''

"""

Q. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) predictive of his or her intelligence?

 

Import the iq_size.csv file

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

    What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
    Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or
    brain size.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('iq_size.csv')
features = dataset.iloc[:, 1:].values
labels = dataset.iloc[:, 0:1].values


'''
import statsmodels.api as sm
features = sm.add_constant(features)



l=[0, 1, 2, 3]
features_opt = features[:,l ]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
c = list(regressor_OLS.pvalues)

for i in range(len(c)):
    if (c[i] >0.0500000000000000):
        number = np.where(c==c[i])[0]
        l.remove(number)
    else:
        pass
'''

from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(features[:,0:3], labels)
number1 = np.where(lin_reg_1.coef_[0] == max(lin_reg_1.coef_[0]))[0]
print((dataset.columns[number1+1])[0])

x = [90,70,150]
x=np.array(x)
print(lin_reg_1.predict(x.reshape(1, -1)))












