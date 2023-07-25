import numpy as np 
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

#load the dataset 
data=load_wine()
X,y=data.data,data.target 

#split thee data into traing and testing data sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#scale  the data 
scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
# they will comwe in the same range 

#create the KNN classifer 
knn=KNeighborsClassifier()
#calling the instance from kneighbour 

#hyperparameter  tuning  with the gridsearchcv
param_grid={'n_neighbors':np.arange(1,21)} # try k values from 1 to 20 
grid_search=GridSearchCV(knn, param_grid,cv=5) 
grid_search.fit(X_train_scaled,y_train)
#to get the bestt valuee fromthe grid seacrh 
best_k=grid_search.best_params_['n_neighbors']
print("best k value:",best_k)
#train the classifer  with the best k value on the scaled training data
knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled,y_train)
# we are train the model based on best value 
#make predections on the scaled test data
y_pred=knn.predict(X_test_scaled)
#  calculate the accuracy of rthe moodel 
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)

#print the  classification report   
target_names=data.target_names
print("Classified report:")
print(classification_report(y_test, y_pred,target_names=target_names))

#visualition
#barchart to show the count of each class int he target variables 
plt.figure(figsize=(6,4))
sns.countplot(  x=y,palette='coolwarm')
plt.xticks(ticks=np.unique(y)  , labels=target_names, rotation=45)
plt.xlabel('class')
plt.ylabel('coutn')
plt.title('Class Distribution')
plt.show()
#confusion heat maP
conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='coolwarm',xticklabels=target_names,yticklabels=target_names)
plt.xlabel('predictive class')
plt.ylabel('true class')
plt.title('Class Distribution')
plt.show()
# sincee accuracy was not that good the data in the heat mapp is miss predicted
# so we get a one in  another box  rather than the diagonal 

