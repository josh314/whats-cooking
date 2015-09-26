"""

Feature extraction:
 From the list of unprocessed ingredient strings:
 *) Tokenizes individual ingredient strings.
 *) Gets n=1,2,3 n-grams.
 *) Terms in n-grams are joined by '::'
 *) All n-grams for a given recipe are then joined into a single string separated by spaces 
    e.g. ['garlic powder', 'onions'] --> 'garlic powder garlic::powder onions'
 *) These strings are treated as a corpus of documents to which a bag of words model is applied
 *) Classification performed with a SGD classifier

"""

import json
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import sklearn.cross_validation as CV
from sklearn.grid_search import GridSearchCV

# Seed for randomization. Set to some definite integer for debugging and set to None for production
seed = None

### Text processing functions ###

def normalize(string):#Remove diacritics and whatevs
    return "".join(ch.lower() for ch in unicodedata.normalize('NFD', string) if not unicodedata.combining(ch))

def tokenize(string):#Ignores special characters and punct
    return re.compile('\w\w+').findall(string)

def ngrammer(tokens):#Gets all grams in each ingredient
    return [":".join(tokens[idx:idx+n]) for n in np.arange(1,1 + min(1,len(tokens))) for idx in range(len(tokens) + 1 - n)]

print("Importing training data...")
with open('/Users/josh/dev/kaggle/whats-cooking/data/train.json','rt') as file:
    recipes_train_json = json.load(file)

#Small subset for Debugging purposes
#recipes_train_json = recipes_train_json[:100]
    
# Build the grams for the training data
print('\nBuilding n-grams from input data...')
for recipe in recipes_train_json:
    recipe['grams'] = [term for ingredient in recipe['ingredients'] for term in ngrammer(tokenize(normalize(ingredient)))]

# Build vocabulary from training data grams
vocabulary = sorted(list({term for recipe in recipes_train_json for term in recipe['grams']}))

# Stuff everything into a dataframe. 
ids_index = pd.Index([recipe['id'] for recipe in recipes_train_json],name='id')
recipes_train = pd.DataFrame([{'cuisine': recipe['cuisine'], 'ingredients': " ".join(recipe['grams'])} for recipe in recipes_train_json],columns=['cuisine','ingredients'], index=ids_index)

# Build SGD Classifier pipeline
text_clf = Pipeline([('vect', CountVectorizer(vocabulary=vocabulary)),
                      ('clf', SGDClassifier(loss='log', penalty='elasticnet', n_iter=5, alpha=1e-4, random_state=seed)),
])
# Grid search over svm classifiers. 
parameters = {
#    'clf__penalty': ('l1', 'l2'),
    'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),#'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'
    'clf__l1_ratio': np.linspace( 0, 1.0, 6),
    'clf__alpha': np.logspace( -5, -2, 5),
}

# Split data into a fitting set and a validation set
#fit_data, val_data, fit_target, val_target = CV.train_test_split( recipes_train['ingredients'].values, recipes_train['cuisine'].values, test_size=0.05, random_state=seed)
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

# Build the grams for the test data
print('\nBuilding n-grams from input data...')
for recipe in recipes_test_json:
    recipe['grams'] = [term for ingredient in recipe['ingredients'] for term in ngrammer(tokenize(normalize(ingredient)))]

# Test data dataframe. 
test_ids_index = pd.Index([recipe['id'] for recipe in recipes_test_json],name='id')
recipes_test = pd.DataFrame([{'ingredients': " ".join(recipe['grams'])} for recipe in recipes_test_json],columns=['ingredients'], index=test_ids_index)

# Test data predictions & evaluation
test_data = recipes_test['ingredients'].values
print("Applying to test data...")
predicted = gs_clf.predict(test_data)

# Out to file
recipes_test.drop('ingredients',axis=1,inplace=True)
recipes_test.insert(0,'cuisine',predicted)
print("Saving predictions to file...\n")
recipes_test.to_csv('/Users/josh/dev/kaggle/whats-cooking/sub/ngram_bag_sgd_grid.csv',index=True)
