#!/usr/bin/env python
# coding: utf-8

# In[1]:


8+90


# In[17]:


from scipy.integrate import quad

# Define the function to integrate
def integrand(x):
    return x**3

# Integrate the function from 0 to 1
result, error = quad(integrand, 0, 1)
print("Result of integration:", result)


# In[18]:


from scipy.optimize import minimize

# Define the function to minimize
def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Initial guess
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

# Minimize the function
res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print("Minimum value found:", res.x)


# In[2]:


from sklearn.model_selection import train_test_split
import numpy as np


# In[4]:


x=np.array([[1,2],[3,4],[5,6],[7,8]])
y=np.array([0,1,0,1])


# In[5]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np

# Generating synthetic data
np.random.seed(42)  # for reproducibility
X = np.random.rand(1000, 10)  # 1000 data points with 10 features
y = np.random.randint(2, size=1000)  # binary labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you have:
# X_train: Training features (80% of the data)
# X_test: Testing features (20% of the data)
# y_train: Training labels corresponding to X_train
# y_test: Testing labels corresponding to X_test


# In[6]:


from sklearn.model_selection import train_test_split
import numpy as np


# In[8]:


np.random.seed(34)
x=np.random.rand(1000,10)
y=np.random.randint(2,size=1000)
x_train,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)


# In[13]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Instantiate the Naive Bayes model
model = GaussianNB()

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Instantiate the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[2]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report


# In[6]:


iris=load_iris()


# In[7]:


iris


# In[8]:


x=iris.data
y=iris.target


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[10]:


logistic=LogisticRegression()


# In[11]:


logistic.fit(x_train,y_train)


# In[13]:


y_pred=logistic.predict(x_test)


# In[14]:


accuracy=accuracy_score(y_pred,y_test)


# In[15]:


cr= classification_report(y_pred,y_test)


# In[16]:


print(accuracy)
print(cr)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier

# Define the features and labels of the two points
X = [[0, 0], [1, 1]]  # Features (coordinates)
y = ['red', 'blue']   # Labels

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier on the data
knn.fit(X, y)

# Define a new point to classify
new_point = [[0.5, 0.5]]

# Predict the class of the new point
predicted_class = knn.predict(new_point)

print("Predicted class of the new point:", predicted_class)



# In[19]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[20]:


iris


# In[21]:


# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Use only the first two features (sepal length and sepal width)
X_subset = X[:, :2]  # Selecting the first two columns (features)

# Split the subset of the dataset into training and testing sets
X_train_subset, X_test_subset, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=1
knn_subset = KNeighborsClassifier(n_neighbors=1)

# Train the classifier on the training data (subset)
knn_subset.fit(X_train_subset, y_train)

# Make predictions on the testing data (subset)
y_pred_subset = knn_subset.predict(X_test_subset)

# Calculate the accuracy of the model (subset)
accuracy_subset = accuracy_score(y_test, y_pred_subset)
print("Accuracy using only the first two features (sepal length and sepal width):", accuracy_subset)


# In[22]:


iris


# In[23]:


import pandas as pd


# In[27]:


pd.DataFrame(iris,columns=["data","target"])


# In[28]:


pd.DataFrame(iris)


# In[29]:


type(iris)


# In[30]:


import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris_data = load_iris()

# Convert the data to a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column to the DataFrame
df['target'] = iris.target

# Display the DataFrame
print(df)


# In[32]:


df1=pd.DataFrame(data=iris.data,columns=iris.feature_names)


# In[33]:


df1


# In[34]:


df1["target"]=iris.target


# In[35]:


df1


# In[39]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict([[6.7,3.0,5.2,2.3]])
print(y_pred)
print([y_pred- y_test])



# In[40]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Plot the first two features
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset - Sepal Length vs Sepal Width')
plt.show()


# In[41]:


import matplotlib.pyplot as plt

# Data
square_footage = [1500, 2000, 1800, 2200, 1700]
price = [250000, 300000, 280000, 320000, 270000]

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(square_footage, price, color='blue')
plt.xlabel('Square Footage (sq. ft.)')
plt.ylabel('Price ($)')
plt.title('House Prices vs Square Footage')
plt.grid(True)
plt.show()


# In[ ]:





# In[57]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
square_footage = [1500, 2000, 1800, 2200, 1700]
num_bedrooms = [3, 4, 3, 5, 3]
price = [250000, 300000, 280000, 320000, 270000]

# Create 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(square_footage, num_bedrooms, price, c='blue', marker='o')

# Set labels
ax.set_xlabel('Square Footage (sq. ft.)')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price ($)')

# Set title
ax.set_title('House Prices vs Square Footage vs Number of Bedrooms')

# Show plot
plt.show()


# In[64]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
square_footage =  np.random.randint(1500,6000,100)
num_bedrooms =np.random.randint(3,9,100)
price =  np.random.randint(250000,960000,100)

# Create 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(square_footage, num_bedrooms, price, c=(1, 0, 0), marker='o')  # Use RGB tuple for blue color


# Set labels
ax.set_xlabel('Square Footage (sq. ft.)')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price ($)')

# Set title
ax.set_title('House Prices vs Square Footage vs Number of Bedrooms')

# Show plot
plt.show()


# In[43]:


import numpy as np

# Generate random numbers within a specific range
# For example, to generate 5 random numbers between 10 and 20
lower_bound = 10
upper_bound = 20
random_numbers = lower_bound + (upper_bound - lower_bound) * np.random.rand(5)

print(random_numbers)


# In[44]:


np.random.rand(10)


# In[56]:


np.random.randint(0,12,5)


# In[65]:


from sklearn.model_selection import train_test_split
import numpy as np

# Generating synthetic data
np.random.seed(42)  # for reproducibility
X = np.random.randint(0, 100, size=100)  # Generate 100 random integers between 0 and 100
y = np.random.randint(0, 100, size=100)  # Generate 100 random integers between 0 and 100

# Reshape X and y to be 2D arrays
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you have:
# X_train: Training features (80% of the data)
# X_test: Testing features (20% of the data)
# y_train: Training labels corresponding to X_train
# y_test: Testing labels corresponding to X_test


# In[67]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model=LinearRegression()


# In[68]:


model.fit(X_train,y_train)


# In[69]:


y_test_pred= model.predict(X_test)


# In[71]:


y_train_pred= model.predict(X_train)


# In[73]:


mse1=mean_squared_error(y_test_pred,y_test)
mse2=mean_squared_error(y_train_pred, y_train)


# In[74]:


print(mse1)
print(mse2)


# In[75]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Make predictions on the testing set
y_test_pred = model.predict(X_test)

# Calculate mean squared error on training and testing sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Print mean squared error
print("Mean Squared Error on training set:", mse_train)
print("Mean Squared Error on testing set:", mse_test)


# In[ ]:




