from nltk.corpus import names
import nltk
import random

def gender_features(name):
    features = {}
    features['fl'] = name[0].lower()
    features['ll'] = name[-1].lower()
    features['fw'] = name[:2].lower()
    features['lw'] = name[-2:].lower()
    return features

names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])

random.shuffle(names)

featuresets = [(gender_features(n), g) for (n, g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.MaxentClassifier.train(train_set)

print classifier.classify(gender_features('Neo'))
print classifier.classify(gender_features('Trinity'))
print nltk.classify.accuracy(classifier, test_set)
print classifier.show_most_informative_features(5)
