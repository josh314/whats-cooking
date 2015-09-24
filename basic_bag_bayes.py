"""
This script uses a basic bag of words model for features. There is no attempt to normalize the ingredients' text. Each ingredient is used as separate term and feature. The input feature matrix trains a Naive-Bayes classifier
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import sklearn.cross_validation as CV
from sklearn import metrics

print("Importing training data...")
with open('/Users/josh/dev/kaggle/whats-cooking/data/train.json','rt') as file:
    recipes_train_json = json.load(file)

ingredients_master_list = sorted(list({ingredient for recipe in recipes_train_json for ingredient in recipe['ingredients']}))

# Stuff this into a dataframe. Convert the individual recipe lists into strings with separators inserted between ingredients (preserving whitespace within ingredients) and put into a single `ingredients` column.
sep = "::"
recipes_train = pd.DataFrame([{'id':recipe['id'], 'cuisine': recipe['cuisine'], 'ingredients':sep.join(recipe['ingredients'])} for recipe in recipes_train_json],columns=['id','cuisine','ingredients'])

def tokenize(s):
    return s.split(sep)

# Split data into a training set and a validation set
train_data = recipes_train['ingredients'].values
train_target = recipes_train['cuisine'].values
fit_data, val_data, fit_target, val_target = CV.train_test_split( recipes_train['ingredients'].values, recipes_train['cuisine'].values, test_size=0.2)


# Build Naive-Bayes Classifier pipeline
text_clf = Pipeline([('vect', CountVectorizer(vocabulary=ingredients_master_list,tokenizer=tokenize)),
                      ('clf', MultinomialNB()),
])

# Fit pipeline
print()
print("Fitting Naive Bayes classifier to training data...")
text_clf = text_clf.fit(fit_data, fit_target)

# Validation data predictions & evaluation
print("Applying classifier to valdation data...")
predicted = text_clf.predict(val_data)
print("Reporting...")
print(metrics.classification_report(val_target, predicted))
print("Confusion Matrix:")
print(metrics.confusion_matrix(val_target, predicted))


# Import competition test data
print("Importing competition test data...")
with open('/Users/josh/dev/kaggle/whats-cooking/data/test.json','rt') as file:
    recipes_test_json = json.load(file)

# Load into dataframe and combine strings as done for training data
recipes_test = pd.DataFrame([{'id':recipe['id'], 'ingredients':sep.join(recipe['ingredients'])} for recipe in recipes_test_json],columns=['id','ingredients'])

# Test data predictions & evaluation
test_data = recipes_test['ingredients'].values
print("Applying to test data...")
predicted = text_clf.predict(test_data)

# Out to file
recipes_test.drop('ingredients',axis=1,inplace=True)
recipes_test.insert(1,'cuisine',predicted)
print("Saving predictions to file")
recipes_test.to_csv('/Users/josh/dev/kaggle/whats-cooking/sub/basic_bag_bayes.csv',index=False)


