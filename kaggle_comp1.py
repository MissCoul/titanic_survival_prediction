#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import missingno as msno
from sklearn.metrics import roc_curve, auc


# In[2]:


# Import the Titanic training and test data
train_data = pd.read_csv('/Users/minatacoulibaly/train.csv')
test_data = pd.read_csv('/Users/minatacoulibaly/test.csv')


# In[3]:


train_data.head(5)


# In[4]:


test_data.head(5)


# In[5]:


train_data.info()


# In[6]:


test_data.info()


# In[7]:


TARGET_VARIABLE = "Survived"


# In[8]:


CATEGORICAL_VARIABLES = ["Survived", "Sex", "SibSp", "Parch", 
                         "Cabin", "Embarked"]


# In[9]:


for variable in CATEGORICAL_VARIABLES:
  print("##### %s" % variable)
  print(train_data[variable].value_counts(normalize=True).mul(100).round(2).astype(str))


# In[10]:


# Check for missing values
missing_values = train_data.isnull().sum()

# Print the number of missing values in each column
print(missing_values)


# In[11]:


# Check for missing values
missing_values2 = test_data.isnull().sum()

# Print the number of missing values in each column
print(missing_values2)


# In[12]:


#Percent rate of missing values in age for train data
print("Total percent of missing values", (train_data.isna().sum()["Age"]/len(train_data))*100)
print("Number of missing value by survival modalities\n", train_data[train_data["Age"].isna()].groupby(["Survived"]).size())


# In[13]:


#Percent rate of missing values in Cabin for train data
print("Total percent of missing values", (train_data.isna().sum()["Cabin"]/len(train_data))*100)
print("Number of missing value by survival modalities\n", train_data[train_data["Cabin"].isna()].groupby(["Survived"]).size())


# In[14]:


#Percent rate of missing values in age for test data
print("Total percent of missing values", (test_data.isna().sum()["Age"]/len(train_data))*100)


# In[15]:


#Percent rate of missing values in age for test data
print("Total percent of missing values", (test_data.isna().sum()["Cabin"]/len(train_data))*100)


# In[16]:


get_ipython().system('pip install missingno')


# In[17]:


msno.bar(train_data)


# In[18]:


msno.bar(test_data)


# In[19]:


train_data['Cabin']=train_data['Cabin'].fillna('Unknown')
train_data['Cabin'][:10]


# In[20]:


test_data['Cabin']=test_data['Cabin'].fillna('Unknown')
test_data['Cabin'][:10]


# In[21]:


train_data.dropna(subset=['Embarked'],how='any',inplace=True)


# In[22]:


test_data.shape


# In[ ]:





# In[23]:


test_data.shape


# In[24]:


train_data['Age']=train_data['Age'].replace(np.NaN,train_data['Age'].mean())


# In[25]:


test_data['Age']=test_data['Age'].replace(np.NaN,test_data['Age'].mean())


# In[26]:


# Check for missing values
missing_values = train_data.isnull().sum()

# Print the number of missing values in each column
print(missing_values)


# In[27]:


# Check for missing values
missing_values2 = test_data.isnull().sum()

# Print the number of missing values in each column
print(missing_values2)


# In[28]:


train_data['Age'] = train_data['Age'].astype(int)


# In[29]:


train_data[train_data['Name'].duplicated()]


# In[30]:


test_data['Age'] = test_data['Age'].astype(int)


# In[31]:


test_data.info()


# In[32]:


train_data.info()


# In[33]:


# Convert the name variable to a categorical variable
train_data['Name'] = train_data['Name'].astype('category')

# Calculate the correlation coefficient
correlation_coefficient = train_data['Name'].cat.codes.corr(train_data[TARGET_VARIABLE])

# Print the correlation coefficient
print(correlation_coefficient)


# In[34]:


# Convert the Ticket variable to a categorical variable
train_data['Ticket'] = train_data['Ticket'].astype('category')

# Calculate the correlation coefficient
correlation_coefficient = train_data['Ticket'].cat.codes.corr(train_data[TARGET_VARIABLE])

# Print the correlation coefficient
print(correlation_coefficient)


# In[35]:


# Calculate the correlation between passengerID and the target variable
correlation = train_data['PassengerId'].corr(train_data['Survived'])

# Print the correlation coefficient
print(correlation)


# In[36]:


# Drop the columns `name` and `ticket`
data = train_data.drop(columns=['Name', 'Ticket', 'PassengerId'])

# Print the DataFrame
print(data)


# In[37]:


# Drop the columns `name` and `ticket`
no_drop = test_data.drop(columns=['Name', 'Ticket'])
data_test = no_drop.drop(columns=['PassengerId'])

# Print the DataFrame
print(data_test)


# In[38]:


x=data.corr(method="pearson")
sns.heatmap(x, annot=True, cmap='coolwarm')


# In[39]:


fig = plt.figure(figsize =(10, 7))
plt.hist(x = [data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']],stacked=True, color = ['pink','red'],label = ['Survived','Not survived'])
plt.title('Age Histogram with Survival')
plt.xlabel('Age')
plt.ylabel('No of passengers')
plt.legend()


# In[40]:


data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
data_test['Sex'] = data_test['Sex'].map({'female': 1, 'male': 0}).astype(int)


# In[41]:


categorical_variables = ['Sex', 'Cabin', 'Embarked']


# In[42]:


encoder = OneHotEncoder()

# Fit the OneHotEncoder object to the categorical variables
encoder.fit(data[categorical_variables])

# Transform the categorical variables using the OneHotEncoder object
encoded_variables = encoder.transform(data[categorical_variables])


# In[43]:


data.head(5)


# In[44]:


# Fit the OneHotEncoder object to the categorical variables
encoder.fit(data_test[categorical_variables])

# Transform the categorical variables using the OneHotEncoder object
encoded_variables = encoder.transform(data_test[categorical_variables])


# In[45]:


data_test.head(5)


# In[46]:


data.info()


# In[47]:


data['Embarked'] = data['Embarked'].map( {'Q': 0,'S':1,'C':2}).astype(int)


# In[48]:


data_test['Embarked'] = data_test['Embarked'].map( {'Q': 0,'S':1,'C':2}).astype(int)


# In[49]:


data.info()


# In[50]:


data_test.info()


# In[51]:


# Create a dictionary to map cabin values to int values
cabin_dict = {}
i = 1
for cabin in data['Cabin']:
    if cabin not in cabin_dict:
        cabin_dict[cabin] = i
        i += 1

# Replace the cabin values with the int values
data['Cabin'] = data['Cabin'].map(cabin_dict)
data_test['Cabin'] = data_test['Cabin'].map(cabin_dict)


# In[52]:


data_test['Cabin'].fillna(0, inplace=True)

# Convert 'Cabin' column to int

data_test['Cabin'] = data_test['Cabin'].astype(int)


# In[53]:


data_test.head(5)


# In[54]:


data_test.info()


# In[55]:


data.head(5)


# In[56]:


# Create an imputer with your chosen strategy (e.g., 'mean', 'median', etc.)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the test data
data_test = imputer.fit_transform(data_test)


# In[57]:


Train = data.drop(['Survived'], axis=1)
Test = data['Survived']
x_train, x_test, y_train, y_test = train_test_split(Train, Test, test_size=0.2, random_state=1)


# In[58]:


# Train your logistic regression model
model1 = LogisticRegression()
model1.fit(Train, Test)


# In[59]:


# Initialize LogisticRegression for multiclass classification
model1 = LogisticRegression(solver='lbfgs', multi_class='multinomial')

# Train the model on your training data
model1.fit(x_train, y_train)

# Make predictions on the test data
y_pred1 = model1.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred1)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred1, average='weighted')
recall = recall_score(y_test, y_pred1, average='weighted')
f1 = f1_score(y_test, y_pred1, average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')


# In[60]:


# Initialize DecisionTreeClassifier
dt_model = DecisionTreeClassifier()

# Train the model on your training data
dt_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred_dt = dt_model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_dt)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_dt, average='weighted')
recall = recall_score(y_test, y_pred_dt, average='weighted')
f1 = f1_score(y_test, y_pred_dt, average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

# Generate a classification report
class_report = classification_report(y_test, y_pred_dt)
print("Classification Report:\n", class_report)


# In[61]:


# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)  # You can adjust hyperparameters as needed

# Train the model on your training data
rf_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_rf)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
f1 = f1_score(y_test, y_pred_rf, average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')


# In[62]:



# Initialize models
logistic_model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=1)  # Adjust hyperparameters as needed

# Initialize a list of models and their names
models = [logistic_model, decision_tree_model, random_forest_model]
model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]

# Initialize the plot
plt.figure(figsize=(8, 6))

# Plot ROC curves for each model
for model, name in zip(models, model_names):
    # Train the model on your training data
    model.fit(x_train, y_train)

    # Make probability predictions on the test data
    y_prob = model.predict_proba(x_test)

    # Calculate the TPR and FPR
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)

    # Calculate the Area Under the ROC Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Add labels and legend to the plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc='lower right')

# Show the plot
plt.grid(True)
plt.show()


# In[63]:


# Make predictions on the new, unlabeled dataset (data_test)
predictions = model.predict(data_test)


# In[64]:


# Assuming 'predictions' contains the predicted labels
for i, prediction in enumerate(predictions):
    print(f"Individual {i + 1}: {'Survived' if prediction == 1 else 'Not Survived'}")


# In[65]:


# Assuming 'predictions' is a NumPy array containing your predictions
predictions = np.array(predictions)  # Replace with your actual predictions

# Assuming 'passenger_ids' is a list or array of corresponding PassengerId values
passenger_ids = no_drop['PassengerId']  # Replace with actual PassengerId values

# Create a DataFrame from the predictions and 'PassengerId' values
prediction_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})

# Save the DataFrame to a CSV file with the header included
prediction_df.to_csv('prediction_with_header2.csv', index=False)

# Display the DataFrame with the header
print(prediction_df)


# In[ ]:




