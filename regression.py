import numpy as np

from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

regression_dataset = np.genfromtxt("regressionData-1.csv", delimiter=",")

print(regression_dataset.shape[0]*80/100)
print(regression_dataset.shape[0]*20/100)

numberOfFeatures = 11

split_ds = train_test_split(regression_dataset, test_size=int(regression_dataset.shape[0]*20/100), random_state = 1)

train_ds = np.array(split_ds[0])
test_ds = np.array(split_ds[1])

train_class = train_ds[:,numberOfFeatures]
train_features = train_ds[:,:-1]


neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(train_features, train_class)

test_class = test_ds[:,numberOfFeatures]
test_features = test_ds[:,:-1]

results = []
for tf in test_features:
    result = ""
    result = neigh.predict([tf])
    results.append(result[0])

predicted_results = np.array(results)
print(predicted_results)
print(test_class)
print (metrics.r2_score(predicted_results, test_class)*100)

#print(train_ds)
#print(test_ds)

#train_ds, test_ds = train_test_split(regression_dataset, train_size=int(regression_dataset.shape[0]*80/100), test_size=int(regression_dataset.shape[0]*20/100))

#print(neigh)

#testdata = np.array([3,43,22,6,23])

#print(neigh.predict(testdata))