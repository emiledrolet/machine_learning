# Have to Use Jupyter Notebook
# The score is 0.77033 on Kaggle.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

file_path = 'train.csv'

data = pd.read_csv(file_path)

data.style.set_table_styles(
    [{
        'selector': 'th',
        'props': [('font-size', '8pt')]
    },
    {
        'selector': 'td',
        'props': [('font-size', '8pt')]
    }]
)

# Feature X and y

y = data.Survived
X = data.drop(['Name', 'Ticket', 'Cabin', 'Survived'], axis=1)


# Split data 80/20

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

## Pre-processing

# Numericals Columns

numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

# Categorical Columns

categorial_cols = [col for col in X_train.columns if X_train[col].dtype in ['object']]

# Pre-processing num_cols

numerical_transformer = SimpleImputer(strategy='median')

# Pre-processing cat_cols

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Bundle pre-processing

preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorial_cols)
        ]
)


# Model
model = XGBClassifier()


# Pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


# GridSearch
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.02, 0.03, 0.04]
}


grid_search = GridSearchCV(estimator=my_pipeline, param_grid=param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_train,y_train)

print(grid_search.best_estimator_)

best = grid_search.best_estimator_

predicts = best.predict(X_valid)


score = accuracy_score(y_valid, predicts)
print(f"The score is {score}")


# Submission

submission_data = pd.read_csv('test.csv')

submission_template = pd.read_csv('gender_submission.csv')

X_test = submission_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

predicts_test = best.predict(X_test)


submission_df = pd.DataFrame(
    {
        'PassengerId' : X_test['PassengerId'],
        'Survived' : predicts_test
    }
)

print(submission_df)

submission_df.to_csv('submission.csv', index=False)
