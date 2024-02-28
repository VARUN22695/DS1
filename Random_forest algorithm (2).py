#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("airfoil_self_noise.dat",sep="\t")


# In[3]:


df


# In[4]:


df.header=None


# In[5]:


df


# In[6]:


df=pd.read_csv("airfoil_self_noise.dat",sep="\t",header=None)


# In[7]:


df


# In[8]:


df.columns=["Freq","Angle","Chord Length","FS vel","suction","pressure level"]


# In[9]:


df


# In[10]:


df


# In[20]:


df["Freq"].isnull()


# In[26]:


df["Freq"].isnull().sum()


# In[27]:


df


# In[11]:


x=df.iloc[:,:-1]


# In[12]:


x


# In[13]:


y=df.iloc[:,-1]


# In[14]:


y


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.33, random_state=42)


# In[19]:


from sklearn.ensemble import RandomForestRegressor


# In[20]:


from sklearn.metrics import accuracy_score


# In[48]:


rf_model = RandomForestRegressor(n_estimators=100,random_state=42)


# In[49]:


rf_model.fit(x_train,y_train)


# In[50]:


y_pred= rf_model.predict(x_test)


# In[51]:


acc= accuracy_score(y_test,y_pred)


# In[52]:


from sklearn.metrics import mean_absolute_error


# In[53]:


acc=mean_absolute_error(y_pred,y_test)


# In[54]:


acc


# In[55]:


from sklearn.metrics import r2_score


# In[56]:


acc1= r2_score(y_test,y_pred)


# In[57]:


acc1


# In[58]:


import numpy as np


# In[59]:


rmse= np.sqrt(acc)


# In[60]:


rmse


# In[61]:


from sklearn.model_selection import GridSearchCV, KFold


# In[63]:


param_grid = {
    'n_estimators': [100, 200, 300],            # Number of trees in the forest
    'max_depth': [None, 10, 20],                 # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],             # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]                # Minimum number of samples required to be at a leaf node
}


# In[64]:


grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


# In[ ]:


grid_search.fit(x_train, y_train)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(random_state=42)

# Create the GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_rf_regressor = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_rf_regressor.predict(X_test)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE) using GridSearchCV:", rmse)


# In[21]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', C=1)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(svm_classifier, X, y, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Calculate and print the average cross-validation score
average_cv_score = cv_scores.mean()
print("Average Cross-Validation Score:", average_cv_score)


# In[22]:


from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# In[25]:


k=load_iris()


# In[26]:


k


# In[27]:


type(k)


# In[30]:


x=k.data


# In[31]:


y=k.target


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


regression= LinearRegression()


# In[36]:


regression.fit(x_train,y_train)


# In[37]:


y_test_pred= regression.predict(x_test)


# In[45]:


y_train_pred= regression.predict(x_test)


# In[46]:


from sklearn.metrics import mean_squared_error


# In[47]:


mean_squared_error(y_train_pred,y_test_pred)


# In[48]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# In[53]:


model= SVC(kernel='linear',C=1)


# In[57]:


cv_scores=cross_val_score(model,x,y,cv=10)


# In[58]:


cv_scores


# In[59]:


acc= cv_scores.mean()


# In[60]:


acc


# In[61]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

# Initialize the Support Vector Machine classifier
svm = SVC()

# Initialize GridSearchCV with the classifier and parameter grid
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)

# Perform GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Train the final model with the best parameters
best_estimator = grid_search.best_estimator_
best_estimator.fit(X_train, y_train)

# Evaluate the final model on the test set
test_score = best_estimator.score(X_test, y_test)
print("Test Score:", test_score)


# In[ ]:





# In[62]:


from sklearn.linear_model import Lasso


# In[63]:


from sklearn.linear_model import Ridge


# In[65]:


model=Lasso(alpha=0.1)


# In[66]:


model.fit(x_train,y_train)


# In[67]:


y_pred= model.predict(x_test)


# In[68]:


mean_squared_error(y_pred,y_test)


# In[69]:


model=Ridge(alpha=0.1)


# In[70]:


model.fit(x_train,y_train)


# In[71]:


y_pred= model.predict(x_test)


# In[72]:


mean_squared_error(y_pred,y_test)


# In[73]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

# Initialize the Support Vector Machine classifier
svm = SVC()

# Initialize GridSearchCV with the classifier and parameter grid
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)

# Perform GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Train the final model with the best parameters
best_estimator = grid_search.best_estimator_
best_estimator.fit(X_train, y_train)

# Evaluate the final model on the test set
test_score = best_estimator.score(X_test, y_test)
print("Test Score:", test_score)


# In[75]:


model=SVC()


# from sklearn.model_

# In[77]:


from sklearn.model_selection import train_test_split , GridSearchCV


# In[82]:


grid={"C":[0.1,1,10,100],"gamma":[0.001,0.01,0.1,1],"kernel":["linear","poly","rbf"]}


# In[83]:


grid= GridSearchCV(estimator=model,param_grid=grid,cv=5)


# In[84]:


grid.fit(x_train,y_train)


# In[87]:


grid.best_params_


# In[91]:


grid.best_score_


# In[93]:


k=grid.best_estimator_


# In[94]:


k.fit(x_train, y_train)


# In[95]:


k.score(x_test,y_test)


# In[ ]:




