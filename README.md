# Binary-Classification-ML

## Setup

```
# Let's start off with all the basic imports
# Make sure you run this cell!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

```

## Main Objective

In this assignment, we are going to build a function that will take in a Pandas data frame containing data for a binary classification problem. Our function will try out and tune many different models on the input data frame it receives and at the end it is going to return the model it thinks is best, as well as an expectation of its performance on new and unseen data in the future. To achieve this mighty task we are going to build several helper functions that our main function is going to have access to.

### Extract Function

```
def extract_x_and_y(df, y_column):
    y=df[y_column]
    del df[y_column]
    x=df
    return(x,y)
    
```

### Split Function

```
def split_x_and_y(X, y, test_size = 0.2, random_state = 42):
    # % of the sample size
    train_size=int(len(X)*test_size)
    
    #Make our results reproducible
    np.random.seed(random_state)
    
    #Select randomly the rows for the training dataset
    rows_array=np.random.choice(len(X),size=train_size,replace=False)
    
    #Create x,y train datasets
    X_train=X.iloc[rows_array]
    y_train=y.iloc[rows_array]
    
    #Select the rest arrays for the test dataset
    total_rows=np.arange(len(X))
    test_arrays=np.delete(total_rows,rows_array)
    
    #Create x,y test datasets
    X_test=X.iloc[test_arrays]
    y_test=y.iloc[test_arrays]
    
    return(X_train,y_train,X_test,y_test)
    
```

### Models Classifiers

Create a function specify_models() that takes no parameters at all and returns a list of model definitions for each of 
the above classifiers, where each model definition is the dictionary structure described previously.

```
def specify_models():
    
    knear={'name':'K Nearest Neighbors Classifier',
           'class':sklearn.neighbors.KNeighborsClassifier(),
            'parameters':{'n_neighbors':range(1,12)}
          }
           
    svc_linear={'name':'Support Vector Classifier with Linear Kernel',
               'class':sklearn.svm.LinearSVC(),
                'parameters':{'C':[0.001,0.01,0.1,1,10,100]}
          }  
    
    sv_radial={'name':'Support Vector Classifier with Radial Kernel',
               'class':sklearn.svm.SVC(kernel='rbf'),
                'parameters':{'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100]}
          }      
    
    loglas={'name':"Logistic Regression with LASSO",
             'class':sklearn.linear_model.LogisticRegression(penalty='l1'),
             'parameters':{'C':[0.001,0.01,0.1,1,10,100]}
            }  
    
    sgdc={'name':"Stochastic Gradient Descent Classifier",
            'class':sklearn.linear_model.SGDClassifier(),
            'parameters':{'max_iter':[100,1000],'alpha':[0.0001,0.001,0.01,0.1]}
            }  
    
    decis_tree={'name':"Decision Tree Classifier",
            'class':sklearn.tree.DecisionTreeClassifier(),
            'parameters':{'max_depth':range(3,15)}
            } 
    
    ranfor={'name':"Random Forest Classifier",
            'class':sklearn.ensemble.RandomForestClassifier(),
            'parameters':{'n_estimators':[10,20,50,100,200]}
            } 
    
    extrerantree={'name':"Extremely Randomized Trees Classifier",
                    'class':sklearn.ensemble.ExtraTreesClassifier(),
                    'parameters':{'n_estimators':[10,20,50,100,200]}
                 } 
   
    
    lis=list([knear,svc_linear,sv_radial,loglas,sgdc,decis_tree,ranfor,extrerantree])
    
    return(lis)

```
### TRAIN THE MODEL

What we have right now a list of dictionaries. Each dictionary essentially has the ingredients for us to train a model and tune the right parameters for that model. So, what we need now is a function, train_model() that takes in the following parameters:

    model_dict : We will pass in the dictionaries from the list you just created one by one to this parameter
    X: The input data
    y: The target variable
    metric : The name of a metric to use for evluating performance during cross validation. Please give this parameter a default value of 'f1' which is the F measure.
    k : The number of folds to use with cross validation, the default should be 5

This function should essentially just call GridSearchCV() by correctly passing in the right information from all the different input parameters. The function should then return:

    name : The human readable name for the model type that was trained
    best_model : The best model that was trained
    best_score : The best score (for the metric provided) that was found


```
from sklearn.model_selection import GridSearchCV

def train_model(model_dict, X, y, metric = 'f1', k = 5):
    name=model_dict['name']
    param_grid = model_dict['parameters']
    clf=GridSearchCV(estimator=model_dict['class'], param_grid=param_grid, cv= k, scoring=metric)
    best_score= clf.fit(X,y).best_score_
    best_model= clf
    return(name, best_model, best_score)
```

### Central Component

```
def train_all_models(models, X, y, metric ='accuracy', k = 5):
    #Initialize the list
    final_list=list()
    
    for i in range(0,len(models)):
        tr_model=train_model(models[i] ,X ,y , metric = metric, k=k)
        final_list.append(tr_model)
        
    #Sort the final list    
    final_list=sorted(final_list, key=lambda score: score[2], reverse=True)
    return(final_list)
```

### Classifier Function

```
def auto_train_binary_classifier(df, y_column, models, test_size = 0.2, random_state = 42, 
                                 metric = 'f1', k = 5):
    
    #Use the first function to split df to data and response
    extr_df=extract_x_and_y(df, y_column)
    
    #Use the second function to split the dataframe to training and test
    split_df=split_x_and_y(extr_df[0], extr_df[1], 
                           test_size = test_size, 
                           random_state = random_state
                          )
    
    #Train all the models
    final_model=train_all_models(models, split_df[0],split_df[1], metric = metric, k = k)
    
    #Take the best model, it's name and the score
    best_model_name=final_model[1][0]
    best_model=final_model[1][1]
    train_set_score=final_model[1][2]
    
    ##################################
    # Test set performance
    ##################################
    
    predicted=final_model[1][1].predict(split_df[2])
    test_set_score=sklearn.metrics.accuracy_score(split_df[3], predicted)
    
    return(best_model_name, best_model, train_set_score, test_set_score)

```
 
 ### Testing
 
This section is an opportunity for you to test what you have implemented in this assignment. There are no more questions in this assignment, this section is only there to help you. In the code below, we've loaded up a data set into a Pandas dataframe and we call your auto_train_binary_classifier() function to see the result. Use this as an opportunity to see if your function returns an output that you expect.
 
 ```
from sklearn.datasets import load_breast_cancer, load_iris
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = pd.Series(cancer.target)

# The next commands will only work once you've implemented these functions above.
models = specify_models()
best_model_name, best_model, train_set_score, test_set_score = auto_train_binary_classifier(cancer_df, 'target', models)
print(best_model_name)
print(best_model)
print(train_set_score)
print(test_set_score)
 
 ```
 
 
