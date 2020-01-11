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
    <img src="Doc_Files/Variable Importance Diagram.png"></br>
</p>

Based on this criterion, the most important variable is Corrected fan speed (NRF). The least important variable is Demanded fan speed. Comparing obtained results with other sources shows that we were able to correctly calculate the HI for each row of data and then find the importance of each variable based on its effect on HI value, using RFs algorithm. These variables are shown in the following table. The observed difference in the order of variables importance can be caused by different settings used for RFs algorithm.

<p align="center">
    <img src="Doc_Files/table 3.PNG"></br>
</p>

Next figure shows the health indices of 249 training aircraft engines. These health indices were transformed from the original data using the T-matrix transformation. Large variations in the health indices were observed at the beginning of the degradation processes of the training units when the original 21 variables were used to compute HI. In addition, sudden decreases in the health indices were observed at the end of the degradation processes of the training units. However, the variations in the health indices should be respectively small because the C-MAPSS tool models a gradual degradation process due to wear. These observations indicate that some redundant variables might be used for computing the health indices.

<p align="center">
    <img src="Doc_Files/HIs - 21 variables.png"></br>
</p>

In the next step, only 7 more important parameters were considered, and other sensor measurements were removed. The new calculated health indices are show in the following figure. When 7 variables were used to compute the health indices, smaller variations in the health indices were observed. So, it is expected to get better results in RUL prediction after removing redundant data. All the calculations and plotting commands for this section can be found in “Variable_Importance.py” code.

<p align="center">
    <img src="Doc_Files/HIs - 7 variables.png"></br>
</p>

###Machine Learning Baselines
The calculated values for baselines (calculated with different aggregation primitives) are shown in following tables.

* Prediction error values of baselines calculated with [Min, Max, Last] aggregation primitives
<p align="center">
    <img src="Doc_Files/table 4.PNG"></br>
</p>

* Prediction error values of baselines calculated with [Min, Max, Last, Mean, Std] aggregation primitives
<p align="center">
    <img src="Doc_Files/table 5.PNG"></br>
</p>

It can be seen that using more aggregation primitives leads to a slight enhancement of detection accuracy. A comparison between results shows that number of variables did not have a great impact on performance. The minimum prediction error in each column is shown with green color. These values can be used as criteria for evaluating the performance of machine learning algorithms in next section.

### RUL prediction

The RUL prediction was conducted in different condition to evaluate the effect of each parameter on performance. All the techniques have been applied once on the whole data set and another time on the limited data set including only 7 more important variables.
From the feature engineering point of view, 2 different set of aggregation primitives were evaluated:
1) Min, Max, Last
2) Min, Max, Last, Mean, Std
These are just two samples from hundreds possible combination of primitives to see the effect of different features on prediction.
In each case, MAE, RMSE, and RE metrics were used to measure the prediction accuracy.

Finally, following machine learning approaches were applied to find the best fit to this project.
1) Random Forests (RFs)
2) Support Vector Regression (SVR)
3) Classification and Regression Tree (CART)
4) Multi Layer Perceptron (MLP)
The results are shown in the following tables (calculated with different aggregation primitives).

* Prediction error values of base learners calculated with [Min, Max, Last] aggregation primitives
<p align="center">
    <img src="Doc_Files/table 6.PNG"></br>
</p>

* Prediction error values of base learners calculated with [Min, Max, Last, Mean, Std] aggregation primitives
<p align="center">
    <img src="Doc_Files/table 7.PNG"></br>
</p>

At first glance, it can be seen that Random Forests algorithms is performing considerably better than other base learners. However, CART and MLP usually achieve less relative error.
All the calculations for this section can be found in “RUL_Estimator.py” manuscript.

## Conclusion

While working on engine run-to-failure data set, one of the major steps is creating a pipeline which handles the time series. For this purpose, it is needed to truncate each of time series and consider the remaining number of cycles as its remaining useful life. As a result, the sensor measurements will not have the same shape and size after truncation. Here is when feature engineering comes to the scene and helps to create new features from available data in each sub-set of data. However, creating useful features is another challenge that needs human expertise or huge amount of calculation, trying to find it by trail and error. In this project, 2 different set of aggregation primitives were used for feature creation. The result shows that blindly adding extra features does not lead to better performance in prediction. So, it is important to evaluate the impact of each newly generated feature before adding it to the data. This can be done by using the Complexity time series primitive to create more advanced features or using Recursive Feature Elimination along with Random Forest Regression module to assign scores to created features.
Different number of base variables (7 & 21) were also used for prediction. Although the preliminary results showed that using only 7 more important variables leads to less amount of variation in health indices, in practice, employing this smaller data set did not enhance the accuracy. But it is noteworthy that using smaller data sets decreases the volume of calculation and optimizes the prediction process. 
Using 3 different metrics for error shows different aspects of prediction accuracy. It is interesting that RFs are performing well based on the Root Mean Square Error metric, but considering the Relative Error, MLP and CART are greatly performing better than RFs. It should be noted that underestimating the RUL is preferable to overestimating it as it does not lead to failure of the engine and just stops it earlier.
4 different machine learning algorithms has been used for prediction of RUL. Obtained results show that most of the algorithms are working better than baselines which proves they are making an improvement in prediction. It can be seen that RFs have a better performance compared to others. In the next degree, CART and SVR are doing well, specifically while using more aggregation primitives. The parameter settings of each algorithm have been selected based on experience or by trial and error. However, using more advanced techniques for tuning these parameters can lead to better results in future.
By having the data pipeline ready now, more machine learning algorithms can be taken into account in next steps. Also, using an ensemble of these algorithms can be another important step toward improving the prediction by combining the advantages of different base learners.

## Future work

Here is some suggestion for improving the results in future works:
1. Creating different sets of cut off times and using them for cross validation to find the best set.
2. Using K-Means clustering technique and make new entities from clusters by grouping the engines with similar operational settings.
3. using the Complexity time series primitive to create more advanced features from data series.
4. Using Recursive Feature Elimination along with Random Forest Regression module to assign scores to created feature and selecting the features with higher impacts.
5. Using Gaussian Process or other techniques to tune hyperparameters (like number of estimators or maximum of features for Random Forest regression).
6. Employing an ensemble of various machine learning algorithms to find the optimum combination of different base learners and assigning appropriate weights to each of them.

## References

Breiman, L. (2001). "Random forests." Mach.Learning, 45(1), 5-32.
Hu, C., Youn, B. D., Wang, P., and Yoon, J. T. (2012). "Ensemble of data-driven prognostic algorithms for robust prediction of remaining useful life." Reliab.Eng.Syst.Saf., 103 120-135.
Li, Z., Goebel, K., and Wu, D. (2019). "Degradation Modeling and Remaining Useful Life Prediction of Aircraft Engines Using Ensemble Learning." Journal of Engineering for Gas Turbines and Power, 141(4), 041008.
Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation." 2008 international conference on prognostics and health management, IEEE, 1-9.
