# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model created by Litchi Sun Zulato. It is a GradientBoostingClassifier using default hyperparameters and 100 estimators in scikit-learn 1.1.2.

## Intended Use
This model should be used to predict if a person's income is less or greater than 50k a year.

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 32561 rows and 15 columns, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
Accuracy:
[0.86987153, 0.8599254,  0.867385,   0.86406962, 0.85702445, 0.8503937, 0.84832159, 0.85536676, 0.86199751, 0.87603648]
Average acc:
0.8610392040000001
Precision: 0.6114819759679573
Recall: 0.793073593073593
F1: 0.6905390124387486

## Ethical Considerations
Although some attributes such as race, sex and native-country may seem relevant to the model and help with a greater accuracy, the model won't take into considerations any kind of prejudice or marginalization that person went through in their life.

## Caveats and Recommendations
Based on the previous point it is advised to drop the appointed features for a less prejudiced model.