## Airline Passenger Satisfaction Clustering & Classification Project - Overview:

* Created a model as well as customer segments to identify which customers are reviewing their flight experience as satisfactory in addition to using 
identified customer segments for marketing and business decision purposes.

* Performed EDA to extract and investigate insights within the data. I managed to find some interesting details of the data, including which customers were more likely to be satisfied with their flight experience. Following this, I thought to also perform clustering techniques on the data to identify if there are any review patterns within the data.

* Following this, I built a classifier model to predict instances of when a person may leave a satisfied review using a myriad of different algorithms.

* NOTE: I did not collect or generate this data personally. The data used for this project comes from Kaggle at the following link:
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, scipy, matplotlib, seaborn, sklearn, uumap, statsmodels

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Article which provided useful information on how to apply UMAP in a supervised manner
https://towardsdatascience.com/umap-dimensionality-reduction-an-incredibly-robust-machine-learning-algorithm-b5acb01de568

## Data Cleaning

Columns detaling flight delay times contained some missing values. After looking at the distribution of values in these columns and their relation to other variables, I umputed these missing values with 0.

## EDA
While graphing the data, I managed to find some interesting insights primarily regarding the age along with reasons for travel. I was able to illustrate which passengers were more likely to leave satisfied reviews. 

Using displots, it can been seen that regardless of gender, elderly folks traveling for personal reasons are more likely to fly business class. Of course, those traveling for business purposes are as well, but their ages are much more distributed. In addition, generally those who left higher ratings for Inslight Wifi service were more likely to be satisfied with their flight experience.

Those traveling for business purposes seemed to be more likely to leave poorer ratings for flight time convenience. Travelers sitting in business class seats were also more likely to have satisfied flight experiences.

![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/eda_density_charts.png "Density Charts")
![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/satisfied_chart.png "Satisfaction Chart")
![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/eda_convenience_chart.png "Flight Convenience")
![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/eda_flight_satisfaction_class.png "Class Satisfaction")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/catboost-roc_updated.png "CatBoost ROC AUC Score")

## Model Building
Before building any models, I included the linear discriminants from my LDA application as well as clusters created from applying KMeans Clustering to the dataset as new features. I then scaled the data using MinMaxScaler for the Support Vector Machine implementation, and StandardScaler for all other models attempted. 

* I began model testing with the Support Vector Machine, since we are interested in creating an optimal class seperability, and then attempted K-Nearest Neighbors and Logistic Regression to compare it with these different algorithms. Oddly enough, the models performed better without stratification, but in practice, it may be more beneficial and conducive to practicality to stratify the the testing data since there were more normal records than pathological records. 

* This then led me to try Random Forest and AdaBoost Classifier in tandem with StratifiedKFold to ensure balanced classes while training. The Random Forest Classifier performed better on all folds.

* I then concluded using optuna with XGBoost and CatBoost Classifiers. As expected, the CatBoost Classifier yielded the best results out of all the models attempted. 

(<i>As a potential point to try in the future, I wonder how well the models would perform if including some newly sampled data to balance the records on the class targets. SMOTE is a potential method to perform this</i>).

## Model Performance
The Random Forest and CatBoost classifier models had the best two performances, with CatBoost being the top model. These models performed a little better over the previous models of Support Vector Machine, K-Nearest Neighbors, and Logistic Regression Models attempted. Below are the recorded <b>Weighted F1 Scores and Accuracies</b> for each of the models performed:

* Support Vector Machine F1 Score: 91%, Accuracy: 91.54% --- <i>with SMOTE</i> F1 Score: 96%, Accuracy: 96.46%

* K-Nearest Neighbors F1 Score: 92%, Accuracy: 92.11% --- <i>with SMOTE</i> F1 Score: 98%, Accuracy: 97.91%

* Logistic Regression F1 Score: 91%, Accuracy: 90.60% --- <i>with SMOTE</i> F1 Score: 87%, Accuracy: 86.71%

* Random Forest Classifier F1 Score: 94.40%, Accuracy: 94.40% --- <i>with SMOTE</i> F1 Score: 97.82%, Accuracy: 97.82%

* AdaBoost Classifier F1 Score: 84.24%, Accuracy: 84.24% --- <i>with SMOTE</i> F1 Score: 85.80%, Accuracy: 85.80%

* XGBoost Classifier F1 Score: 92.24%, Accuracy: 92% --- <i>with SMOTE</i> F1 Score: 97.48%, Accuracy: 97%

* CatBoost Classifier F1 Score: 95.06%, Accuracy: 95.06% --- <i>with SMOTE</i> F1 Score: 98.99%, Accuracy: 98.99%

I used Optuna with XGBoost and CatBoost to build an optimized model since these algorthms include a myriad of attributes to test when trying to optimize them (hyperparameters).

## Productionization
I lasted created a Flask API hosted on a local webserver. For this step I primarily followed the productionization step from the YouTube tutorial series found in the refernces above. This endpoint could be used to access, given cardiotocography exam results, a prediction for whether that pregnancy has normal health conditions or a pathological concern.

<b>UPDATE:</b> Below is an example prediction using the Flask API:

![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/fetal_homepage.png "Website Homepage")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/fetal_prediction.png "Website Prediction Page")

## Future Improvements
If I'm able to return to this project, I would like to create a Flask API with both clustering and classification models as a customer review assignment tool which could serve as a means to continuously append records to the existing clusters as well as for the classification model to further discern a customer's satisfaction level.
