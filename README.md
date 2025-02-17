
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        df= pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
df.head()

df['Category'].value_counts()

df.isnull().sum()

X=df['Message']
Y=df['Category']
len(X)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y,  test_size=0.20 , random_state=42)

Train=len(X_train)
test=len(X_test)
print(Train,test)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

pipeMNB = Pipeline([
('tfidf', TfidfVectorizer()),('clf', MultinomialNB())
])
pipeCNB = Pipeline([
('tfidf', TfidfVectorizer()),('clf', ComplementNB())
])
pipeSVC = Pipeline([
('tfidf', TfidfVectorizer()),('clf', LinearSVC())
])

#MultinomialNB
pipeMNB.fit(X_train, Y_train)
predictMNB = pipeMNB.predict(X_test)
#ComplementNB
pipeCNB.fit(X_train, Y_train)
predictCNB = pipeCNB.predict(X_test)
#LinearSVC
pipeSVC.fit(X_train, Y_train)
predictSVC = pipeSVC.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print(f"MNB: {accuracy_score(Y_test, predictMNB):.4f}")
print(f"CNB: {accuracy_score(Y_test, predictCNB):.4f}")
print(f"SVC: {accuracy_score(Y_test, predictSVC):.4f}")

classification= classification_report(Y_test, predictSVC)
print(classification)

message = "Call 927363663 to recieve your price"
result = pipeSVC.predict([message])
print("Result: ", result[0])
