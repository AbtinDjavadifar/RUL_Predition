# Aircraft Engine Run-to-Failure Prediction

## Project summary

Degradation modeling and prediction of remaining useful life (RUL) are crucial to prognostics and health management of aircraft engines. The objective of this project is to introduce a machine learning-based prognostic approach to model an exponential degradation process due to wear as well as predicting the RUL of aircraft engines. Project was started by finding more important variables based on their effect on health index value for each data point. To make data ready before applying the machine learning algorithms, Feature Tools package in python has been used to truncate data and calculate the RUL for each engine based on the number of remaining life cycles. As truncating data leads to creation of data sets with various lengths, feature engineering has been applied to take the characteristic features of engines, making the input data have the same size and shape. Finally, Random Forests (RFs), Support Vector Regression (SVR), Classification and Regression Tree (CART), and Multi Layer Perceptron (MLP) algorithms have been employed to estimate the RUL of test data set. Tests were conducted with various number of variables and aggregation primitives. Also different error metrics were used to evaluate prediction accuracy.

## Problem description

Aircraft or turbine engine failures may result in significant economic losses and even accidents in extreme cases. While the reliability of turbine engines in use on modern turbine-powered aircrafts has been improved over the past few decades, abnormal engine degradation can occur at any time because of a variety of mechanical problems
The RUL of an aircraft engine is defined as the amount of time in hours or cycles from the current time to the end-of-life in which an aircraft engine is expected to serve its intended function. Predictive maintenance requires health monitoring systems and predictive modeling technologies. The existing literature pertaining to RUL prediction for aircraft engines can be classified into two categories: model-based and data-driven prognostics. Model-based prognostic methods describe system behavior and system degradation using physics-based models typically in combination with state estimators such as the Kalman filter, the particle filter, and the hidden Markov model. Data-driven prognostic methods represent the system degradation process using machine learning algorithms. Current data-driven methods are developed based on classical machine learning algorithms such as neural networks, support vector machines (SVM), and decision trees (Li et al. 2019). One of the primary limitations associated with classical machine learning algorithms is that they are not able to predict the RUL of aircraft engines with enough
accuracy. For most complex systems like aircraft engines, finding a suitable model that allows the injection of health-related changes certainly is a challenge. So, it is desired to find a robust and accurate solution to estimate remaining life of an unspecified system using historical data only, irrespective of the underlying physical process (Saxena et al. 2008).

## Research method

One of the main factors for success of data-driven approaches is availability of clean and well-organized data sets. So, in the first part, health index was calculated for each cycle and then Random Forests algorithm was used to find more important variables which affect the health of engine. Then, the data were loaded, and time indices were assigned to each cycle based on its time order. This value was used in the next step, where a random cycle for each engine has been selected and any data point after that cycle was ignored. Next, the sensors measurements for each engine have been converted to some new features, leading to creation of an input data set consisting of data points with same size and shape. Finally, different machine learning algorithms have been applied on training data and their performance in prediction has been evaluated using the test data set.

## Results
### Variable selection

In the model training stage, not all the variables are useful. Considering some variables may even reduce prediction accuracy because these variables may not be correlated to the degradation behavior of aircraft engines. To select the most effective variables, RFs were used to measure the importance of measurement variables with respect to their performance on prediction accuracy. The importance of variables is shown in a bar chart:

<p align="center">
    <img src="Aircraft_Engine_RUL_Predition/Doc_Files/Variable Importance Diagram.png" width=600></br>
</p>

Based on this criterion, the most important variable is Corrected fan speed (NRF). The least important variable is Demanded fan speed. Comparing obtained results with other sources shows that we were able to correctly calculate the HI for each row of data and then find the importance of each variable based on its effect on HI value, using RFs algorithm. These variables are shown in Table 3. The observed difference in the order of variables importance can be caused by different settings used for RFs algorithm.

