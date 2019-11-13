import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

train_dataset = np.genfromtxt("classificationData-1/trainingData2.csv", delimiter=",")
test_dataset = np.genfromtxt("classificationData-1/testData2.csv", delimiter=",")

numberOfFeatures = 5

def basicKNNClassify():
    model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')

    train_class = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    # Train the model using the training sets
    model.fit(train_features,train_class)

    test_class = test_dataset[:,numberOfFeatures]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = model.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)
    return (metrics.accuracy_score(predicted_results, test_class)*100)


def decisionTreeClassification():

    model = tree.DecisionTreeClassifier()

    train_class = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    # Train the model using the training sets
    model.fit(train_features,train_class)

    test_class = test_dataset[:,5]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = model.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)
    return (metrics.accuracy_score(predicted_results, test_class)*100)

def naiveBayesClassify():

    model = GaussianNB()

    train_class = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    # Train the model using the training sets
    model.fit(train_features,train_class)

    GaussianNB(priors=None, var_smoothing=1e-09)

    test_class = test_dataset[:,5]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = model.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)
    return (metrics.accuracy_score(predicted_results, test_class)*100)


def svmClassify():

    model = SVC(gamma="auto")

    train_class = train_dataset[:,numberOfFeatures]
    train_features = train_dataset[:,:-1]

    # Train the model using the training sets
    model.fit(train_features,train_class)

    GaussianNB(priors=None, var_smoothing=1e-09)

    test_class = test_dataset[:,5]
    test_features = test_dataset[:,:-1]

    results = []
    for tf in test_features:
        result = ""
        result = model.predict([tf])
        results.append(result[0])

    predicted_results = np.array(results)
    return (metrics.accuracy_score(predicted_results, test_class)*100)    

print("basic classification: ",basicKNNClassify())
print("decision tree classification: ", decisionTreeClassification())
print("naive bayes classification: ", naiveBayesClassify())
print("support vector classification: ", svmClassify())