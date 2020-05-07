
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from scipy import stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap 

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("spam.data.txt")
df


# In[3]:


df.columns.unique()


# In[4]:


df_null_values = df.isnull().sum()
df_null_values.values


# In[5]:


df.describe()


# In[6]:


Y_num = df['class']
X_num = df.drop(['class'], axis = 1)
X_num


# In[7]:


# Normalise the Data
X_num_std = StandardScaler().fit_transform(X_num)
X_num_std = pd.DataFrame(X_num_std)
X_num_std.describe()


# In[8]:


df_norm = pd.concat((X_num_std, Y_num), axis = 1)
df_norm


# In[9]:


Y = df_norm.iloc[:,[57]].values
X = np.asarray(df_norm.drop(['class'], axis = 1))
print ("Shape of X: ", X.shape)
print ("Shape of Y: ", Y.shape)


# In[10]:


#data biased? Data Distribution? Data Availabity for each class?
labels = ['Spam', 'Not-Spam']   
sizes = df['class'].value_counts()         
explode = (0, 0.2)                               
col = ("gray", 'cyan')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%', shadow=False, startangle=90, colors = col)


ax1.axis('equal')  
fig1.set_size_inches(8, 8)                       
plt.tight_layout()
plt.title("Spam or Not (in Percentage)")
plt.show()


# In[11]:


count = df['class'].value_counts()
count


# In[12]:


df_long = df[(df['class'] == 1)]
long  = df_long.groupby(["ccapital_run_length_longest"])["class"].count()
long


# In[13]:


plt.figure(figsize = (6,4))
plt.plot(long)
plt.xlabel("Capital Run Length Longest")
plt.ylabel("Count")
plt.title("Count of Capital Run Length Longest")


# In[14]:


df_notspam = df[(df['class'] == 0)]
notspam  = df_notspam.groupby(["ccapital_run_length_longest"])["class"].count()
notspam_list = notspam.tolist()
long_list = long.tolist()


# In[15]:


Unique_Word_length = df['ccapital_run_length_longest'].unique() 
Unique_Word_length


# In[16]:


capital_letter_longest = df['ccapital_run_length_longest']
classes = df['class']

t_test = stats.ttest_ind(capital_letter_longest, classes)
print("t-statistic value is: ", t_test[0])
print("p-value is: ", t_test[1])
# there is a relationship between capital letter longest and spam emails


# In[17]:


pca = PCA(n_components=10)
pca_results = pca.fit_transform(X)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# In[18]:


# Making a new dataset with PCA data

df_pca = pd.DataFrame(data = pca_results, columns = ['pca_1', 'pca_2','pca_3', 'pca_4','pca_5', 'pca_6','pca_7', 'pca_8','pca_9', 'pca_10'])
Y_df = pd.DataFrame(data = Y, columns = ['class_labels'])
pca_df = pd.concat((df_pca, Y_df), axis = 1)
pca_df


# In[19]:


sns.scatterplot(x="pca_1", y="pca_2",data=pca_df,legend="full",alpha=0.5)


# In[21]:


tsne = TSNE(n_components=2, verbose=1, perplexity=200, n_iter=500)
tsne_results = tsne.fit_transform(X)


# In[22]:


tsne_df = pd.DataFrame()
tsne_results_df = pd.DataFrame()
tsne_results_df['tsne-2d-one'] = tsne_results[:,0]; tsne_results_df['tsne-2d-two'] = tsne_results[:,1]
tsne_df = pd.concat((tsne_results_df, Y_df), axis = 1)
tsne_df


# In[24]:


sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",data=tsne_df,legend="full",alpha=0.5)


# In[25]:


umap_results = umap.UMAP(n_neighbors=12,min_dist=.05,metric='correlation').fit_transform(X)


# In[26]:


umap_df = pd.DataFrame()
umap_results_df = pd.DataFrame()
umap_results_df['umap_1'] = umap_results[:,0]; umap_results_df['umap_2'] = umap_results[:,1]
umap_df = pd.concat((umap_results_df, Y_df), axis = 1)
umap_df


# In[28]:


sns.scatterplot(x="umap_1", y="umap_2",data=umap_df,legend="full",alpha=0.5)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[30]:


mlp = MLPClassifier(max_iter=200)

# Hyper-parameter space to optimize MLP
parameter_space = {
    'hidden_layer_sizes': [(10,200), (20,400), (30,200), (20,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
}

# Note: the max_iter=100 that you defined on the initializer is not in the grid. 
# So, that number will be constant, while the ones in the grid will be searched.

# Run Grid Search
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)


# In[31]:


# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# In[32]:


print('Best parameters found:\n', clf.best_params_)


# In[33]:


y_true, y_pred = y_test , clf.predict(X_test)

# Confustion Matrix
cm = confusion_matrix(y_true,y_pred)
print('Confusion Matrix:\n', cm, '\n')
sns.heatmap(cm,annot=True,fmt="d", annot_kws={"size": 12}) # font size

# Classification Report
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))


# In[35]:


X_df = df_norm.drop(['class'], axis = 1)
plt.subplots(figsize=(30, 30))
sns.heatmap(X_df.corr(), annot=True, linewidths=.5, fmt= '.1f')


# In[41]:


# SelectKBest Features using F-Test

# Assign a score to each feature
x_new_f = SelectKBest(f_classif, k=46)
x_train_f = x_new_f.fit_transform(X_train, y_train)
scores = x_new_f.scores_
scores /= scores.max()

# Make a table of feature and its score in descending order
x_new_f_df = pd.DataFrame()
x_new_f_df['f_Features'] = np.arange(0,57,1)
x_new_f_df['f_Scores'] = x_new_f.scores_
x_new_f_df.sort_values(['f_Scores'], ascending = False)


# In[42]:


# Using Features selected through F-test to train model and test it

# Specify Classifier
mlp = MLPClassifier(max_iter=500)

# Hyper-parameter space to optimize MLP
parameter_space = {
    'hidden_layer_sizes': [(10,200), (20,400), (30,200), (20,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
}

# Note: the max_iter=100 that you defined on the initializer is not in the grid. 
# So, that number will be constant, while the ones in the grid will be searched.

# Run Grid Search
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(x_train_f, y_train)

# Best parameter set
print('Best parameters found:\n', clf.best_params_,'\n')


# In[43]:


X_test_f = x_new_f.fit_transform(X_test, y_test)


# In[44]:


# Now you can use the clf to make new predictions. 
# For example, check the performance on your test set.    
y_true, y_pred = y_test , clf.predict(X_test_f)

# Confustion Matrix
from sklearn.metrics import f1_score,confusion_matrix
cm = confusion_matrix(y_true,y_pred)
print('Confusion Matrix:\n', cm, '\n')

# Classification Report
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))


# In[45]:


# Univariate feature Sewlectio Chi Square Test
np.amin(X_train)


# In[50]:


X_train1 = X_train + 0.94
X_test1 = X_test + 0.94
print("Min of X_train1 is :",np.amin(X_train1))
print("Min of X_test1 is :",np.amin(X_test1))


# In[52]:


# SelectKBest Features using Chi-Square-Test

# Assign a score to each feature
x_new_chi2 = SelectKBest(chi2, k=46)
x_train_chi2 = x_new_chi2.fit_transform(X_train1, y_train)
scores = x_new_chi2.scores_
scores /= scores.max()

# Make a table of feature and its score in descending order
x_new_chi2_df = pd.DataFrame()
x_new_chi2_df['chi2_Features'] = np.arange(0,57,1)
x_new_chi2_df['chi2_Scores'] = x_new_chi2.scores_
x_new_chi2_df.sort_values(['chi2_Scores'], ascending = False)


# In[53]:


# Using Features selected through Chi-Square-Test to train model and test it

# Specify Classifier
mlp = MLPClassifier(max_iter=500)

# Hyper-parameter space to optimize MLP
parameter_space = {
    'hidden_layer_sizes': [(10,200), (20,400), (30,200), (20,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
}

# Note: the max_iter=100 that you defined on the initializer is not in the grid. 
# So, that number will be constant, while the ones in the grid will be searched.

# Run Grid Search
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(x_train_chi2, y_train)

# Best parameter set
print('Best parameters found:\n', clf.best_params_,'\n')


# In[54]:


# Preparing testing data

X_test_chi2 = x_new_chi2.fit_transform(X_test1, y_test)


# In[55]:


# Now you can use the clf to make new predictions. 
# For example, check the performance on your test set.    
y_true, y_pred = y_test , clf.predict(X_test_chi2)

# Confustion Matrix
from sklearn.metrics import f1_score,confusion_matrix
cm = confusion_matrix(y_true,y_pred)
print('Confusion Matrix:\n', cm, '\n')
#sns.heatmap(cm,annot=True,fmt="d", annot_kws={"size": 12}) # font size

# Classification Report
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))


# In[56]:


# Univariate feature Seclection Mutual Information
# SelectKBest Features using Mutual Information

# Assign a score to each feature
x_new_mi = SelectKBest(mutual_info_classif, k=46)
x_train_mi = x_new_mi.fit_transform(X_train, y_train)
scores = x_new_mi.scores_
scores /= scores.max()

# Make a table of feature and its score in descending order
x_new_mi_df = pd.DataFrame()
x_new_mi_df['mi_Features'] = np.arange(0,57,1)
x_new_mi_df['mi_Scores'] = x_new_mi.scores_
x_new_mi_df.sort_values(['mi_Scores'], ascending = False)


# In[57]:


# Using Features selected through Mutual Information Test to train model and test it

# Specify Classifier
mlp = MLPClassifier(max_iter=500)

# Hyper-parameter space to optimize MLP
parameter_space = {
    'hidden_layer_sizes': [(10,200), (20,400), (30,200), (20,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
}

# Note: the max_iter=100 that you defined on the initializer is not in the grid. 
# So, that number will be constant, while the ones in the grid will be searched.

# Run Grid Search
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(x_train_mi, y_train)

# Best parameter set
print('Best parameters found:\n', clf.best_params_,'\n')


# In[58]:


X_test_mi = x_new_mi.fit_transform(X_test, y_test)


# In[59]:


# Now you can use the clf to make new predictions. 
# For example, check the performance on your test set.    
y_true, y_pred = y_test , clf.predict(X_test_mi)

# Confustion Matrix
from sklearn.metrics import f1_score,confusion_matrix
cm = confusion_matrix(y_true,y_pred)
print('Confusion Matrix:\n', cm, '\n')
#sns.heatmap(cm,annot=True,fmt="d", annot_kws={"size": 12}) # font size

# Classification Report
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))

