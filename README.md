# Profitable-customers-analysis

Topic : Analysis of profitable customers of an online food order and delivery service.

Author : Abinav Ramesh Sundararaman

Date : 3rd March, 2021

## Background

A bunch of data for a food-delivery service is collected with the help of a survey. The surveys are then manually typed into a computer by the company's office in a foreign country.Through the internal data on these users, customers have been separated these into two classes:
1. `great_customers`: those who are very profitable to the company
2. `no_great_customers`: and those who are not very profitable to the company

The aim is to build a predictive model for `great_customers` and also to find the characteristics of customers that make them profitable 

## Details of Functions performed in this project

1. Loading Project configurations (Like project name, file path and other global attributes. They can be defined in the .json file)
2. Created Dated directories to store outputs per day 
3. Loading data and Exception handling if File does not exist in the given path
4. Data Pre-processing
- Remove Duplicates
- Treat high cardinality features (By reducing their cardinality)
- Build pipelines for treating Continuous and Categorical variables
5. Treating Feature imbalance by applying SMOTE technique (Synthetic Minority Oversampling)
6. Perform Random Forest modelling and Feature importances
- Store Performance metrics of Random Forest into the dated directory corresponding to the date of the run
- Store Feature importances into the dated directory corresponding to the date of the run
7. Perform SVM Classifier
- Store Feature importances into the dated directory corresponding to the date of the run
8. Written test cases to test DataLoaders(`test_DataLoaders.py`) and Preprocess module (`test_Preprocess.py`)

## Sample output of Performance metrics


|**accuracy**  |**prec**|**recall**|**f1**|**auc** |**recall**|
|--------------|--------|----------|------|--------|----------|
|0.9           |0.62    |0.58      |0.58  | 0.58   |0.8       |

## Sample output of Feature importances


|**feature**     |**Importances**|
|----------------|---------------|
|age             |0.2            |
|marital-status  |0.1            |

### Additional Learnings : 

Abstract Classes in Python : 
https://towardsdatascience.com/abstract-base-classes-in-python-fundamentals-for-data-scientists-3c164803224b

Directory Structure based on dates : 
https://stackoverflow.com/questions/34411061/python-create-directory-structure-based-on-the-date

Scikit learn pipelines : 
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

Sklearn Pipeline: Get feature names after OneHotEncode In ColumnTransformer
https://stackoverflow.com/questions/54646709/sklearn-pipeline-get-feature-names-after-onehotencode-in-columntransformer

Unit testing in Python : 
https://www.youtube.com/watch?v=6tNS--WetLI