#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import plotly.express as px
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Dataset_Predicting Hospital\diabetic_data.csv")


# In[3]:


df2= pd.read_csv(r"C:\Users\LENOVO\Downloads\Dataset_Predicting Hospital\mimic_iii_data.csv")


# In[4]:


df = df[df['race'].notna() & (df['race'] != '?')]


# In[5]:


df


# In[6]:


df2


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df


# In[9]:


df2.drop_duplicates(inplace=True)


# In[10]:


df2


# In[11]:


###Check for missing values
print("Summary of missing values in each column:")
missing_values = df.isnull().sum()
print(missing_values)


# In[12]:


###Check for missing values
print("Summary of missing values in each column:")
missing_values = df2.isnull().sum()
print(missing_values)


# In[13]:


print(df['patient_nbr'])


# In[14]:


print(df2['Patient_ID'])


# In[15]:


# Rename a column (old name to new name)
df.rename(columns={'patient_nbr': 'Patient_ID'}, inplace=True)


# In[16]:


df


# In[17]:


df = df[~df['gender'].isin(['Unknown/Invalid'])]


# In[18]:


# Merge Datasets on Patient ID
merged_df = pd.merge(df, df2, on='Patient_ID', how='outer')


# In[19]:


merged_df


# In[20]:


# Data Cleaning
merged_df.dropna(inplace=True)


# In[21]:


# Encode Categorical Variables
label_encoders = {}
for col in merged_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col])
    label_encoders[col] = le


# In[22]:


print(df['readmitted'])


# In[23]:


print(df2['Readmission_Flag'])


# In[24]:


# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()


# In[25]:


sns.countplot(x='gender', data = df)
plt.title('Gender Distribution')
plt.show()


# In[26]:


sns.countplot(x='race', data=df)
plt.title('Race Distribution')
plt.show()


# In[27]:


# Group the data by 'Diagnoses' and calculate the average ICU Length of Stay
diagnosis_icustay_avg = merged_df.groupby('Diagnoses')['ICU_Length_of_Stay'].mean().reset_index()

# Sort the result by the average ICU Length of Stay in descending order (optional)
diagnosis_icustay_avg = diagnosis_icustay_avg.sort_values(by='ICU_Length_of_Stay', ascending=False)

# Display the result
print(diagnosis_icustay_avg)


# In[28]:


# Count the number of occurrences for each diagnosis
diagnosis_counts = df2['Diagnoses'].value_counts()


# In[29]:


# Create a pie chart to visualize the number of occurrences of each diagnosis
plt.figure(figsize=(8,8))
diagnosis_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
plt.title('Distribution of Diagnoses')
plt.ylabel('')  # To remove the y-label, as it's not necessary for a pie chart
plt.show()


# In[30]:


# Group by 'Diagnoses' and calculate the mode for each test (if multiple modes, we pick the first one)
mode_df = df2.groupby('Diagnoses').agg(lambda x: x.mode().iloc[0]).reset_index()


# In[31]:


print(mode_df)


# In[32]:


# Melt the data for easier plotting
mode_df_melted = pd.melt(
    mode_df, 
    id_vars=['Diagnoses'], 
    value_vars=['Blood_Glucose', 'Creatinine', 'Hemoglobin', 'WBC', 'Heart_Rate', 
                'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'SpO2', 'Respiratory_Rate'],
    var_name='Lab Test', 
    value_name='Mode Value'
)

# Verify the melted DataFrame
print(mode_df_melted)


# In[33]:


# Plot the results using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Diagnoses', y='Mode Value', hue='Lab Test', data=mode_df_melted)

# Customize the plot
plt.title('Mode of Lab Test Values by Diagnosis')
plt.ylabel('Mode Value')
plt.xlabel('Diagnosis')
plt.xticks(rotation=45)
plt.legend(title='Lab Test')
plt.tight_layout()

# Show the plot
plt.show()


# In[34]:


#Distribution of ICU length of stay by Diagnosis

plt.figure(figsize=(12, 6))

sns.histplot(data=merged_df, x='ICU_Length_of_Stay', hue='Diagnoses', kde=False, multiple="stack") #kde=True for kernel density estimate

plt.title('Distribution of ICU Length of Stay by Diagnosis')

plt.xlabel('ICU Length of Stay')

plt.ylabel('Count')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if needed

plt.tight_layout() # Adjust layout to prevent labels from overlapping

plt.show()


# In[35]:


# Create a treemap using Plotly
fig = px.treemap(
    mode_df_melted, 
    path=['Diagnoses', 'Lab Test'],  # Nested path for hierarchical structure
    values='Mode Value',  # The mode value will determine the size of each block
    color='Mode Value',  # Color the blocks based on mode values
    hover_data=['Lab Test'],  # Show the test names on hover
    title='Treemap of Mode Values by Diagnosis'
)

# Show the plot
fig.show()


# In[36]:


readmission_rate=merged_df['readmitted'].value_counts()

print(readmission_rate)


# In[37]:


readmission_rate.plot.pie( autopct='%1.1f%%', startangle=90)

plt.ylabel('')  # Remove default 'Values' label

plt.title('Readmission Distribution')

plt.show()


# In[38]:


sns.boxplot(x='Readmission_Flag', y='ICU_Length_of_Stay', data=merged_df)
plt.title('ICU Stay Length vs Readmission')
plt.show()


# In[39]:


# Mode of Diagnoses by Readmission
mode_diag = merged_df.groupby('Readmission_Flag')['Diagnoses'].agg(lambda x: x.mode()[0])
print("Most Common Diagnosis for Each Readmission Status:")
print(mode_diag)


# In[40]:


# Encode Categorical Variables
age_mapping = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
    '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}
merged_df['age'] = merged_df['age'].map(age_mapping)

label_encoders = {}
for col in ['gender', 'race', 'Diagnoses']:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col])
    label_encoders[col] = le


# In[41]:


# Feature Selection
features = (['age', 'gender', 'race', 'ICU_Length_of_Stay', 'Number_of_Lab_Tests'])
X = merged_df[features]
y = merged_df['Readmission_Flag']


# In[42]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


# Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:





# In[61]:


# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[46]:


import pandas as pd

X_train_df = pd.DataFrame(X_train)  # Convert NumPy array to DataFrame
print(X_train_df.isnull().sum())    # Check for missing values


# In[47]:


from sklearn.impute import SimpleImputer

# Create an imputer object with the mean strategy
imputer = SimpleImputer(strategy='mean')

# Apply imputer to training data
X_train = imputer.fit_transform(X_train)

# Apply the same transformation to test data (if applicable)
X_test = imputer.transform(X_test)


# In[48]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[49]:


# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))


# In[50]:


y_pred = model.predict(X_train)
print("Accuracy:", accuracy_score(y_train, y_pred))
print(classification_report(y_train, y_pred))
print("confusion matrix:\n", confusion_matrix(y_train, y_pred))


# In[51]:


model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=5, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    random_state=42
)
model.fit(X_train, y_train)


# In[52]:


model.fit(X_train, y_train)


# In[53]:


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[54]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=10,      # Allow deeper trees
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)


# In[55]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[56]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model.fit(X_train_resampled, y_train_resampled)


# In[57]:


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:




