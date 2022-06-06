## Airline Passenger Satisfaction Clustering & Classification Project - Overview:
* This project's goal is to identify factors that lead to flight experience satisfaction, and potentially identify groups of flight experiences with consistent satisfaction scores.

* Performed an exploratory data analysis to extract and investigate insights within the data. I managed to find some interesting details of the data, including which customers were more likely to be satisfied with their flight experience. 

* Created classifier models to identify which customers are reviewing their flight experience as satisfactory in addition to using finding customer segments which
can be used for marketing and business decision purposes.

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
![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/eda_satisfied.png "Wifi Ratings")

## Clustering / Customer Segmentation
In addition to building classifiers for this data, I wanted to also attempt to split these reviews into potential customer segments to identify areas of marketing efforts and understand what certain customers are satisfied and dissatisfied with. I used the UMAP dimensionality reduction technique over PCA since PCA required about 11 components to retain 80% of the variance in the data. 

Following this, I juxtaposed the performances of K-Means and DBSCAN on the components. Since the shapes returned from reducing the dimensionality to 2 resulted in more peculiarly shaped clusters rather than simple spheres, DBSCAN did a better job differentiating between the clusters formed. 

I'll also include a visual depicting the two UMAP components formed when reducing the dimensionality of the whole dataset to 2 components:

![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/umap_2_comps.png "UMAP Components")
![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/umap_kmeans.png "K-Means Applied")
![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/umap_dbscan.png "DBSCAN Applied")


## Model Building
Before I starting building any model, I resplit the data into training and test sets for the sole purpose of ensuring the target variable was evenly distributed among both sets.

As I begun the build classifier models to classify satisfied and dissatisfied records, I first constructed a Logit model using the statsmodels package to see if any attributes were seen as insignificant to that algorithm. It found a few attributes with high p-values (statistically insignificant), which mainly were some of the 1-5 rating variables. <i>Food and drink, Ease of online booking, and Inflight entertainment</i> are a few examples. I believe since these attributes are common among flights no matter what class of seat a passenger sits in, that is why the model deems these attributes as insignificant.

After using the Logit model to investigate attribute significance, I moved into performing different algorithms on the data, being Logistic Regression, KNN, Support Vector Machine, Random Forest, and AdaBoost Classifiers. Each algorithm returned quite strong results, which I will list below.

## Model Performance
Each model build returned strong performance metric values, with Random Forest and AdaBoost Classifier performing the best of all classifiers. Below are the models' accuracy and F1 scores:

* Logistic Regression) Accuracy: 98.42 | F1 Score: 98.18

* KNN) Accuracy: 96.74 | F1 Score: 96.16

* Support Vector Machine) Accuracy: 91.12 | F1 Score: 90.76

* Random Forest Classifier) Accuracy: 98.76 | F1 Score: 98.58

* AdaBoost Classifier) Accuracy: 98.83 | F1 Score: 98.65

I also made a ROC AUC curve to compare the models and decide which algorthm performed the "best" on the data.

![alt text](https://github.com/elayer/Airline-Passenger-Satisfaction-Clustering-Classification/blob/main/auc_curve.png "AUC Curve")


## Future Improvements
If I'm able to return to this project, I would like to create a Flask API with both clustering and classification models as a customer review assignment tool which could serve as a means to continuously append records to the existing clusters as well as for the classification model to further discern a customer's satisfaction level.
