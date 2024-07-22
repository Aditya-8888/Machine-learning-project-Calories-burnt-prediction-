# Machine-learning-project-Calories-burnt-prediction-

# following are the libraries and model are import in the calories burnt prediction that is given in the (.ipynb file)

# numpy: which provides support for mathmatical and numerical operations in python.
#import numpy as np 

#pandas: which offers data structure and tools for data manipulation and analysis.

#import pandas as pd  

#Matplotlib: It allows for creating various types of plots and visualization,such as line,scatter plots,histograms,etc.

import matplotlib.pyplot as plt  

#seaborn: Which is built on top of Matplotlib and offers additional plotting functionalites and aestheric improvements over Matplotlib. 

import seaborn as sns            

# Used for splitting datasets into training and testing sets, which is essential for evaluating machine learning models.

from sklearn.model_selection import train_test_split 

""" Import the metrics modulte from the scikit-learn library, which contain various evaulation metrics for assessing the performance of 
 machine learning models.Includes metrics such as mean absolute error,mean squared error, R-squared score,etc."""
 
from sklearn import metrics

from sklearn.metrics import r2_score 

# It converts categorical column into numerical column.

from sklearn.preprocessing import LabelEncoder 

from sklearn.linear_model import LinearRegression,Lasso,Ridge
