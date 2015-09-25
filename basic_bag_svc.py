"""
This script uses a basic bag of words model for features. There is no attempt to normalize the ingredients' text. Each ingredient is used as separate term and feature. The input feature matrix trains an SVM on part of the data set
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import sklearn.cross_validation as CV
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

# Seed for randomization. Set to some definite integer for debugging and set to None for production
seed = None

print("Importing training data...")
with open('/Users/josh/dev/kaggle/whats-cooking/data/train.json','rt') as file:
    recipes_train_json = json.load(file)

ingredients_master_list = sorted(list({ingredient for recipe in recipes_train_json for ingredient in recipe['ingredients']}))

# Stuff this into a dataframe. Convert the individual recipe lists into strings with separators inserted between ingredients (preserving whitespace within ingredients) and put into a single `ingredients` column.
sep = "::"
recipes_train = pd.DataFrame([{'id':recipe['id'], 'cuisine': recipe['cuisine'], 'ingredients':sep.join(recipe['ingredients'])} for recipe in recipes_train_json],columns=['id','cuisine','ingredients'])

# Tokenizer for the ingredient list strings
def tokenize(s):
    return s.split(sep)


# Build SGD Classifier pipeline
text_clf = Pipeline([('vect', CountVectorizer(vocabulary=ingredients_master_list,tokenizer=tokenize)),
                      ('clf', SVC(C=100.0, gamma=1e-3)),
])
# Grid search over svm classifiers. 
parameters = {
#    'clf__C': np.logspace(1.5, 3, 6),
#    'clf__gamma': ( 1e-4, 1e-3, 1e-2),
}

# Obtain training data subset for fitting
#fit_data, _, fit_target, _ = CV.train_test_split( recipes_train['ingredients'].values, recipes_train['cuisine'].values, train_size=0.4, random_state=seed)
fit_data = recipes_train['ingredients'].values
fit_target = recipes_train['cuisine'].values

# Init GridSearchCV with k-fold CV object
cv = CV.KFold(len(fit_data), n_folds=3, shuffle=True, random_state=seed)
gs_clf = GridSearchCV(
    estimator=text_clf,
    param_grid=parameters,
    n_jobs=-1,
    cv=cv,
    scoring='accuracy',
    verbose=2    
)

# Fit on data subset
print("\nPerforming grid search over hyperparameters...")
gs_clf.fit(fit_data, fit_target)

print("\nTop scoring models under cross-validation:\n")
top_grid_scores = sorted(gs_clf.grid_scores_, key=lambda x: x[1], reverse=True)[:min(25,len(gs_clf.grid_scores_))]
for x in top_grid_scores:
    print(x)

# Import competition test data
print("\nImporting competition test data...")
with open('/Users/josh/dev/kaggle/whats-cooking/data/test.json','rt') as file:
    recipes_test_json = json.load(file)

# Load into dataframe and combine strings as done for training data
recipes_test = pd.DataFrame([{'id':recipe['id'], 'ingredients':sep.join(recipe['ingredients'])} for recipe in recipes_test_json],columns=['id','ingredients'])

# Test data predictions & evaluation
test_data = recipes_test['ingredients'].values
print("Applying to test data...")
predicted = gs_clf.predict(test_data)

# Out to file
recipes_test.drop('ingredients',axis=1,inplace=True)
recipes_test.insert(1,'cuisine',predicted)
print("Saving predictions to file...\n")
recipes_test.to_csv('/Users/josh/dev/kaggle/whats-cooking/sub/basic_bag_svm.csv',index=False)


