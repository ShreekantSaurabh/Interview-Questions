
# coding: utf-8

# # 1. Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 2. Importing dataset

# wine_dataset is divided into red_wine_dataset and white_wine_dataset for analysis

# In[2]:


dataset_wine = pd.read_csv("wine_dataset.csv")
dataset_red = pd.read_csv("red_wine_dataset.csv")
dataset_white = pd.read_csv("white_wine_dataset.csv")


# # 3. Assess Data Quality & Missing Values 

# In[3]:


dataset_wine.head()


# In[4]:


dataset_wine.describe()


# In[5]:


dataset_wine.isnull().sum()


# # 4. Visualize the data

# In[6]:


sns.set()
dataset_red.hist(figsize=(10,10), color='red')
plt.show()


# Observations:
# 1. Alcohol, residual sugar, free & total sulfur dioxide, chlorides, citric acid level distribution looks skewed.
# 

# In[7]:


sns.set()
dataset_white.hist(figsize=(10,10), color='blue')
plt.show()


# Observations:
# 1. Alcohol, residual sugar, free sulfur dioxide, chlorides, citric acid level distribution looks skewed.

# # Correlation Heat Maps for Red & White wine

# In[8]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Red Wine Correlation of features Heatmap")
corr_red = dataset_red.corr()
sns.heatmap(corr_red, 
            xticklabels=corr_red.columns.values,
            yticklabels=corr_red.columns.values,
           linecolor='white', cmap="Reds", annot=True)
plt.show()


# Observations regarding red wine:
# 1. Fixed acidity and the pH levels are highly negatively correlated. (-0.68) ie. lesser pH means more acidic.
# 2. Fixed acidity and Density are highly positively correlated (0.68) ie. when fixed acidity increases density of red wine increases as well.
# 3. Fixed acidity and Citric acid are highly positively correlated. (0.67) ie. More citric acid means more acidic.
# 4. Total sulfur dioxide and free sulfur dioxide are highly positively correlated (0.67)
# 5. Volatile acidity and Citric acid are strongly negatively correlated (-0.55).
# 6. Citric acid and pH are strongly negatively correlated (-0.54).
# 7. Alcohol and Density are negatively correlated (-0.5) ie. when alcohol percentage decreases, density grows.
# 8. Quality and Alcohol are positively correlated (0.48) ie. wines that contain a higher amount of alcohol are better in quality.
# 9. Quality and Volatile acidity are negatively correlated (-0.39) ie. volatile acidity is an indicator of spoilage and could give rise to unpleasant smell.
# 10. Chlorides and sulphates are positively correlated (0.37)
# 11. Density is positively correlated with citric acid and residual sugar (0.36)
# 12. Density and pH are negatively correlated (-0.34)

# In[9]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("White Wine Correlation of features Heatmap")
corr_white = dataset_white.corr()
sns.heatmap(corr_white, 
            xticklabels=corr_white.columns.values,
            yticklabels=corr_white.columns.values,
           linecolor='white', cmap="Purples", annot=True)
plt.show()


# Observations regarding white wine:
# 1. Density and residual sugar are strongly positively correlated (0.84)
# 2. Alcohol and Density are strongly negatively correlated (-0.78)
# 3. Total sulfur dioxide and free sulfur dioxide are highly positively correlated (0.62)
# 4. Density and total Sulfur dioxide are strongly positively correlated (0.53)
# 5. Alcohol and residual sugar are negatively correlated (-0.45)
# 6. Alcohol and total sulfur dioxide are negatively correlated (-0.45)
# 7. Quality and Alcohol are positively correlated (0.44)
# 8. Fixed acidity and the pH levels are highly negatively correlated. (-0.43) ie. lesser pH means more acidic.
# 9. Total sulfur dioxide and residual sugar positively correlated (0.4)
# 10. Alcohol and Chloride are negatively correlated (-0.36)
# 11. Free sulfur dioxide and residual sugar positively correlated (0.3)

# In[10]:


corr_diff = corr_red - corr_white
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Differences between Red and White Wines")
corr = corr_diff
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           linecolor='white', cmap="coolwarm", annot=True)
plt.show()


# Observations regarding Correlation Differences between Red and White Wines:
# 1. Alcohol and Residual Sugar are Highly positively correlated (0.49)
# 2. Density and Residual sugar are negatively Correlated (-0.48)
# 3. Density and Total sulfur dioxide are negatively Correlated (-0.46)
# 4. Density and fixed acidity are positively Correlated (0.4)
# 5. Citric acid and Volatile acidity are negatively Correlated (-0.40)
# 6. Fixed acidity and Citric acid are positively Correlated (0.38)
# 7. pH and Citric acid are negatively Correlated (-0.38)
# 8. Sulphates and Chlorides are positively Correlated (0.35)
# 9. Sulphates and pH are negatively Correlated (-0.35)
# 10. Density and free sulfur dioxide are negatively Correlated (-0.32)

#                                                        RED      |     WHITE
#     * alcohol vs. residual.sugar         :    no corr.          :   strong -ve corr.
#     * residual.sugar vs. density         :    weak +ve corr.    :   strong +ve corr. 
#     * density vs. Total sulfur dioxide   :    no corr           :   strong +ve corr.
#     * fixed.acidity vs. density          :    strong +ve corr.  :   weak +ve corr.
#     * Citric acid vs Volatile acidity    :    weak -ve corr.    :   no corr.
#     * Fixed acidity and Citric acid      :    Strong +Ve corr.  :   weak +ve corr.
#     * pH and Citric acid                 :    Strong -ve corr.  :   weak -ve corr.
#     * chlorides vs. sulphates            :    weak +ve corr.    :   no corr.
#     * Sulphates and pH                   :    weak -ve          :   weak +ve           
#     * Density and free sulfur dioxide    :    no corr           :   weak +ve
#                 

# # 5. Model Building for Red Wine using Random Forest Classifier

#  Defining Target and features

# In[11]:


y = dataset_red.quality                  # set 'quality' as target
X = dataset_red.drop('quality', axis=1)  # rest other variables are features
print(y.shape, X.shape)


# Group the wine into two groups:  'quality' > 5 as "good wine" and 'quality' < 5 as "not so good wine"

# In[12]:


y1 = (y > 5).astype(int)
y1.head()


# In[13]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix


# Split data into training and test datasets

# In[14]:


seed = 8 # set seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2,
                                                    random_state=seed)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Train and evaluate the Random Forest Classifier with Cross Validation

# In[15]:


RF_clf = RandomForestClassifier(random_state=seed)
RF_clf


# Compute k-fold cross validation on training dataset and see mean accuracy score

# In[16]:


cv_scores = cross_val_score(RF_clf,X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy scores for the iterations are {}'.format(cv_scores))
print('The mean accuracy score is {}'.format(cv_scores.mean()))


# Fitting the Random Forest Classifier on training set

# In[18]:


RF_clf.fit(X_train, y_train)


# Predicting the Random Forest Classifier on test set

# In[19]:


pred_RF = RF_clf.predict(X_test)


# # Accuracy, log loss and confusion matrix

# In[20]:


print(accuracy_score(y_test, pred_RF))
print(log_loss(y_test, pred_RF))
print(confusion_matrix(y_test, pred_RF))


# # Hyperparameters Tuning of the Random Forest classifier using GridSearchCV

# In[21]:


from sklearn.model_selection import GridSearchCV
grid_values = {'n_estimators':[50,100,200],'max_depth':[None,30,15,5],
               'max_features':['auto','sqrt','log2'],'min_samples_leaf':[1,20,50,100]}
grid_RF = GridSearchCV(RF_clf,param_grid=grid_values,scoring='accuracy')
grid_RF.fit(X_train, y_train)


# Finding Best Parameters

# In[22]:


grid_RF.best_params_


# Other than number of estimators, the other recommended values are the defaults.
# Fitting the Random Forest Classifier on training set with Best Parameters.

# In[23]:


RF_clf = RandomForestClassifier(n_estimators=100,random_state=seed)
RF_clf.fit(X_train,y_train)
pred_RF = RF_clf.predict(X_test)


# # Accuracy, log loss and confusion matrix after Hyperparameter Tuning

# In[25]:


print(accuracy_score(y_test,pred_RF))
print(log_loss(y_test,pred_RF))
print(confusion_matrix(y_test,pred_RF))


# # 6. Model Building for White Wine using Random Forest Classifier

# Defining Target and features

# In[26]:


y = dataset_white.quality                  # set 'quality' as target
X = dataset_white.drop('quality', axis=1)  # rest other variables are features
print(y.shape, X.shape)


# Group the wine into two groups: 'quality' > 5 as "good wine" and 'quality' < 5 as "not so good wine"

# In[27]:


y1 = (y > 5).astype(int)
y1.head()


# In[28]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix


# Split data into training and test datasets

# In[29]:


seed = 8 # set seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2,
                                                    random_state=seed)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Train and evaluate the Random Forest Classifier with Cross Validation

# In[30]:


RF_clf = RandomForestClassifier(random_state=seed)
RF_clf


# Compute k-fold cross validation on training dataset and see mean accuracy score

# In[32]:


cv_scores = cross_val_score(RF_clf,X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy scores for the iterations are {}'.format(cv_scores))
print('The mean accuracy score is {}'.format(cv_scores.mean()))


# Fitting the Random Forest Classifier on training set

# In[33]:


RF_clf.fit(X_train, y_train)


# Predicting the Random Forest Classifier on test set

# In[34]:


pred_RF = RF_clf.predict(X_test)


# # Accuracy, log loss and confusion matrix¶

# In[35]:


print(accuracy_score(y_test, pred_RF))
print(log_loss(y_test, pred_RF))
print(confusion_matrix(y_test, pred_RF))


# # Hyperparameters Tuning of the Random Forest classifier using GridSearchCV

# In[36]:


from sklearn.model_selection import GridSearchCV
grid_values = {'n_estimators':[50,100,200],'max_depth':[None,30,15,5],
               'max_features':['auto','sqrt','log2'],'min_samples_leaf':[1,20,50,100]}
grid_RF = GridSearchCV(RF_clf,param_grid=grid_values,scoring='accuracy')
grid_RF.fit(X_train, y_train)


# Finding Best Parameters

# In[37]:


grid_RF.best_params_


# Other than number of estimators, the other recommended values are the defaults. Fitting the Random Forest Classifier on training set with Best Parameters.

# In[38]:


RF_clf = RandomForestClassifier(n_estimators=200,random_state=seed)
RF_clf.fit(X_train,y_train)
pred_RF = RF_clf.predict(X_test)


# # Accuracy, log loss and confusion matrix after Hyperparameter Tuning

# In[39]:


print(accuracy_score(y_test,pred_RF))
print(log_loss(y_test,pred_RF))
print(confusion_matrix(y_test,pred_RF))

