"""
This script uses a basic bag of words model for features. There is no attempt to normalize the ingredients' text. Each ingredient is used as separate term and feature. The input feature matrix trains a SGD classifier
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
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
                      ('clf', SGDClassifier(loss='log', penalty='elasticnet', n_iter=5, alpha=1e-4, random_state=seed)),
])
# Grid search over svm classifiers. 
parameters = {
    'clf__loss': ('hinge', 'log', 'modified_huber'),
    'clf__l1_ratio': np.linspace( 0.0, 0.5, 11),
    'clf__alpha': np.logspace( -5, -3, 5),
}

# Split data into a fitting set and a validation set
#fit_data, val_data, fit_target, val_target = CV.train_test_split( recipes_train['ingredients'].values, recipes_train['cuisine'].values, test_size=0.05, random_state=seed)
fit_data = recipes_train['ingredients'].values
fit_target = recipes_train['cuisine'].values

# Init GridSearchCV with k-fold CV object
cv = CV.KFold(len(fit_data), n_folds=5, shuffle=True, random_state=seed)
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
recipes_test.to_csv('/Users/josh/dev/kaggle/whats-cooking/sub/basic_bag_sgd_grid.csv',index=False)


