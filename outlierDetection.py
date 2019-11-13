import numpy as np
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

numberOfFeatures = 7

train_dataset = np.genfromtxt("OutlierData/trainDataO.csv", delimiter=",")
test_dataset = np.genfromtxt("OutlierData/test.csv", delimiter=",")

def outlinerFinderAndBoxplot():

    train_label = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    model = tree.DecisionTreeRegressor()

    # Train the model using the training sets
    model.fit(train_features,train_label)

    test_label = test_dataset[:,numberOfFeatures]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = model.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)

    print ("prediction: ",metrics.r2_score(predicted_results, test_label)*100)

    sns.boxplot(data= np.array(train_features))
    plt.show()


def outlinerRemoval():

    train_label = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    rowsToDelete = train_features[:,0]<7
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]

    rowsToDelete = train_features[:,1] > -5    
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]

    rowsToDelete = train_features[:,2] < 5    
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]    

    rowsToDelete = train_features[:,6] < 7    
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]

    sns.boxplot(data= np.array(train_features))
    plt.show()

    model = tree.DecisionTreeRegressor()

    # Train the model using the training sets
    model.fit(train_features,train_label)

    test_label = test_dataset[:,numberOfFeatures]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = model.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)

    print ("improved prediction: ", metrics.r2_score(predicted_results, test_label)*100)


def outlinerRemovalAndKNReg():

    k = 2

    train_label = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    rowsToDelete = train_features[:,0]<7
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]

    rowsToDelete = train_features[:,1] > -5    
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]

    rowsToDelete = train_features[:,2] < 5    
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]    

    rowsToDelete = train_features[:,6] < 7    
    train_features = train_features[rowsToDelete]    
    train_label = train_label[rowsToDelete]

    neigh = KNeighborsRegressor(n_neighbors=k)

    # Train the model
    neigh.fit(train_features,train_label)

    test_label = test_dataset[:,numberOfFeatures]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = neigh.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)

    print ("improved prediction: ", metrics.r2_score(predicted_results, test_label)*100)


outlinerRemovalAndKNReg()