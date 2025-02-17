# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv
df= pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
df.head()
Category	Message
0	ham	Go until jurong point, crazy.. Available only ...
1	ham	Ok lar... Joking wif u oni...
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
3	ham	U dun say so early hor... U c already then say...
4	ham	Nah I don't think he goes to usf, he lives aro...
df['Category'].value_counts()
Category
ham     4825
spam     747
Name: count, dtype: int64
df.isnull().sum()
Category    0
Message     0
dtype: int64
X=df['Message']
Y=df['Category']
len(X)
5572
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y,  test_size=0.20 , random_state=42)
Train=len(X_train)
test=len(X_test)
print(Train,test)
4457 1115
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
MNB: 0.9650
CNB: 0.9830
SVC: 0.9919
classification= classification_report(Y_test, predictSVC)
print(classification)
              precision    recall  f1-score   support

         ham       0.99      1.00      1.00       966
        spam       0.99      0.95      0.97       149

    accuracy                           0.99      1115
   macro avg       0.99      0.98      0.98      1115
weighted avg       0.99      0.99      0.99      1115

message = "Call 927363663 to recieve your price"
result = pipeSVC.predict([message])
print("Result: ", result[0])
Result:  spam
