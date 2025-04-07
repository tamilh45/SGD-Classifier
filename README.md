# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1. start 

step 2. Import Necessary Libraries and Load Data 

step 3. Split Dataset into Training and Testing Sets 

step 4. Train the Model Using Stochastic Gradient Descent (SGD) 

step 5. Make Predictions and Evaluate Accuracy 

step 6. Generate Confusion Matrix 

step 7. end

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Tamil Pavalan M
RegisterNumber:  212223110058
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df=pd.DataFrame(data=iris.data,columns = iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

```

## Output:

## Head
## Accuracy and Confusion matrix

![Screenshot 2025-04-07 105715](https://github.com/user-attachments/assets/b209b5d5-4c0d-4fc3-bd9f-ac508d7de3ea)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
