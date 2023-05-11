#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries, I have so many of them by default so just choose the ones that works for you
import numpy as np
import pandas as pd
from scipy import stats
import pickle
# Plotting libraries
get_ipython().system('pip install gcsfs')
get_ipython().system('pip install vaex')
get_ipython().system('pip install scikit-learn numpy')
get_ipython().system('pip install matplotlib')
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time
import tensorflow as tf
import vaex
# Sklearn libraries
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
# Filter warnings
warnings.filterwarnings('ignore') #filter warnings
# Show plots inline
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install -U klib')
import klib
get_ipython().system('pip install matplotlib==3.1.3')


# In[2]:


#load the dataset
df = pd.read_csv(r"D:\DATASET\2ND_NEW_ML-EdgeIIoT-dataset.csv")


# In[3]:


df.head()


# In[4]:


#visiualize the dataset
from matplotlib import pyplot as plt

plt.title("Class Distribution")
df.groupby("Attack_type").size().plot(kind='pie', autopct='%.2f', figsize=(20,10))


# In[5]:


#shuffle rows of dataframe 
sampler=np.random.permutation(len(df))
data=df.take(sampler)
df.head()


# In[6]:


#split test and train
from sklearn.model_selection import train_test_split
X = df.drop(['Attack_type'], axis=1)
y = df.filter(['Attack_type'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time


# In[8]:


import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import time

start_time = time.time()

lgbm = lgb.LGBMClassifier(
    n_estimators=100,
    random_state=42,
    bagging_freq=5,
    bagging_fraction=0.75
)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(lgbm, X_train, y_train, cv=10, scoring='accuracy')

end_time = time.time()
computational_time = end_time - start_time

print("Cross-Validation Accuracy Scores:")
print(cv_scores)
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")

# Fit the model on the entire training data
lgbm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lgbm.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate average detection time per sample
num_samples = len(X_test)
avg_detection_time = computational_time / num_samples

print(f"Computational time: {computational_time:.2f} seconds")
print(f"Average detection time per sample: {avg_detection_time:.6f} seconds")

# Plot the confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





# In[ ]:





# In[9]:


##########################
#Explainability Codes with SHAP


# In[10]:


import seaborn as sns
get_ipython().system(' pip install shap')
import shap
shap.initjs()


# In[12]:


# Load the SHAP explainer
explainer = shap.TreeExplainer(lgbm)

# Calculate SHAP values for all samples in the test set
shap_values = explainer.shap_values(X_test)

# Use the summary plot to visualize the feature importance
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Use the force plot to visualize the contribution of each feature for a particular prediction
# Select an index for the test set and generate a force plot for that sample
idx = 0


# In[13]:


class_names = df['Attack_type'].unique().tolist()
shap.summary_plot(shap_values, X_test, class_names=class_names)


# In[15]:


import shap

# Load the SHAP explainer
explainer = shap.TreeExplainer(lgbm)

# Selecting a single test data or row from dataset
idx = 0
sample = X_test.iloc[[idx]]

# Calculate SHAP values for the selected sample
shap_values = explainer.shap_values(sample)

# Use the summary plot to visualize the feature importance
shap.summary_plot(shap_values[0], sample)

# Use the force plot to visualize the contribution of each feature for the selected prediction
shap.force_plot(explainer.expected_value[0], shap_values[0], sample)


# In[ ]:




