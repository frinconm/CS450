from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()

# Randomizing dataset
combined = list(zip(iris.data, iris.target))
np.random.shuffle(combined)

iris.data[:], iris.target[:] = zip(*combined)

# Preparing training / test sets
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.3,
                                                                    random_state=33, shuffle=True)
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

targets_predicted = model.predict(data_test)

data_list = iris.data.tolist()
data_test_list = data_test.tolist()

corrects = 0

print('***************************')
print('GaussianNB')
print('**************************')

for x in range(len(data_test)):
    # print(data_test[x], end=' ')
    # print(iris.target_names[targets_predicted[x]], end=' => ')
    if iris.target[data_list.index(data_test_list[x])] == targets_predicted[x]:
        # print('CORRECT')
        corrects += 1
    # else:
    #     print('INCORRECT')

print('Accuracy: {}/{} => {}'.format(corrects, len(data_test), corrects / len(data_test)))

# KNNeighborsClassifier
class HardCodedModel:

    def __get_distances(self, test_ele):

        distances_array = []

        for x in self.data_trained:
            # Euclidean distance
            dist = np.linalg.norm(test_ele - x)
            # Adding distance to array
            distances_array = np.append(distances_array, dist)

        # Adding target axis to distances
        distances_array = np.array([distances_array, self.target_trained])
        distances_array = np.swapaxes(distances_array, 0, 1)

        # Sorting array by distance
        distances_array = distances_array[distances_array[:, 0].argsort()]

        return distances_array

    def __init__(self, data, target):
        self.data_trained = np.array(data)
        self.target_trained = np.array(target)

    def predict(self, test, k):

        target = []

        for ele in test:
            # Getting distances
            distances = self.__get_distances(ele)
            # Trimming only the first k distances
            possible_targets = distances[:k, 1]
            # Getting the frequencies
            values, counts = np.unique(possible_targets, return_counts=True)
            # Creating array of frequencies
            value_count = np.array([values, counts])
            value_count = np.swapaxes(value_count, 0, 1)
            # Sorting by frequency
            value_count = value_count[value_count[:, 1].argsort()]
            # Adding the most frequent target
            target = np.append(target, value_count[::-1, 0][0])

        target = target.astype(int)
        return target

class KNNeighborsClassifier:

    def fit(self, data, target):
        return HardCodedModel(data, target)

# Change k here
k = 9
classifier = KNNeighborsClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test, k)

print('***************************')
print('Custom K Nearest Neighbors')
print('**************************')
print('k = {}'.format(k))
corrects = 0

for x in range(len(data_test)):
    # print(data_test[x], end=' ')
    # print(iris.target_names[targets_predicted[x]], end=' => ')
    if iris.target[data_list.index(data_test_list[x])] == targets_predicted[x]:
        # print('CORRECT')
        corrects += 1
    # else:
        # print('INCORRECT')

print('Accuracy: {}/{} => {}'.format(corrects, len(data_test), corrects / len(data_test)))

classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

print('***************************')
print('SKLearn K Nearest Neighbors')
print('**************************')
corrects = 0

for x in range(len(data_test)):
    # print(data_test[x], end=' ')
    # print(iris.target_names[targets_predicted[x]], end=' => ')
    if iris.target[data_list.index(data_test_list[x])] == targets_predicted[x]:
        # print('CORRECT')
        corrects += 1
    # else:
        # print('INCORRECT')

print('Accuracy: {}/{} => {}'.format(corrects, len(data_test), corrects / len(data_test)))