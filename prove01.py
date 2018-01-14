from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
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

for x in range(len(data_test)):
    print(data_test[x], end=' ')
    print(iris.target_names[targets_predicted[x]], end=' => ')
    if iris.target[data_list.index(data_test_list[x])] == targets_predicted[x]:
        print('CORRECT')
        corrects += 1
    else:
        print('INCORRECT')

print('Accuracy: {}/{} => {}'.format(corrects, len(data_test), corrects / len(data_test)))

# My classifier
class HardCodedModel:

    def __init__(self):
        pass

    def predict(self, test):
        target = [2] * len(test)
        return target


class HardCodedClassifier:

    def fit(self, data, target):
        return HardCodedModel()


classifier = HardCodedClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

print('****************')

corrects = 0

for x in range(len(data_test)):
    print(data_test[x], end=' ')
    print(iris.target_names[targets_predicted[x]], end=' => ')
    if iris.target[data_list.index(data_test_list[x])] == targets_predicted[x]:
        print('CORRECT')
        corrects += 1
    else:
        print('INCORRECT')

print('Accuracy: {}/{} => {}'.format(corrects, len(data_test), corrects / len(data_test)))
