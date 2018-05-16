#########################################
#Libraries
#########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
from sklearn.model_selection import GridSearchCV

#########################################
def extract_x_and_y(df, y_column):
    y=df[y_column]
    del df[y_column]
    x=df
    return(x,y)

#########################################
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

#########################################
#Try it to with only Logistic Regression model
#########################################
loglas ={'name':"Logistic Regression with LASSO",
         'class':sklearn.linear_model.LogisticRegression(penalty='l1'),
         'parameters':{'C':[0.001,0.01,0.1,1,10,100]}
        }

#########################################
#Add more models in order to create the Binary Classifier
#########################################
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

#########################################
def train_model(model_dict, X, y, metric = 'f1', k = 5):
    name=model_dict['name']
    param_grid = model_dict['parameters']
    clf=GridSearchCV(estimator=model_dict['class'], param_grid=param_grid, cv= k, scoring=metric)
    best_score= clf.fit(X,y).best_score_
    best_model= clf
    return(name, best_model, best_score)

#########################################
def train_all_models(models, X, y, metric ='accuracy', k = 5):
    #Initialize the list
    final_list=list()

    for i in range(0,len(models)):
        tr_model=train_model(models[i] ,X ,y , metric = metric, k=k)
        final_list.append(tr_model)

    #Sort the final list
    final_list=sorted(final_list, key=lambda score: score[2], reverse=True)
    return(final_list)

#########################################
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


    ##################################
    # Test
    ##################################
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
