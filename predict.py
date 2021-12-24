#!/usr/bin/env python
# coding: utf-8

# # Importing all the required libraries

# In[528]:


import pandas as pd

# In[529]:


import numpy as np

# In[530]:


import matplotlib.pyplot as plt

# In[531]:


import re

# In[532]:


import string

# In[533]:


import string

# In[534]:


from IPython.core.interactiveshell import InteractiveShell

# In[535]:

# from nltk.corpus import stopwords

# In[536]:


# from nltk.stem.porter import PorterStemmer

# In[537]:


# from sklearn.feature_extraction.text import CountVectorizer

# In[538]:


# from sklearn.feature_extraction.text import TfidfVectorizer

# In[539]:
from flask import jsonify

InteractiveShell.ast_node_interactivity = "all"

# In[540]:


pd.set_option('display.latex.repr', True)

# In[541]:


pd.set_option('display.latex.longtable', True)

# In[542]:


from sklearn.model_selection import train_test_split

# In[543]:


from sklearn.datasets import load_iris

# In[544]:


from sklearn.linear_model import LogisticRegression

# In[545]:


from sklearn.ensemble import RandomForestClassifier

# In[546]:


from sklearn.svm import LinearSVC

# In[547]:


from sklearn import tree

# In[548]:


from sklearn.tree import DecisionTreeClassifier

# In[549]:


from sklearn.metrics import accuracy_score

# In[550]:


from sklearn.metrics import classification_report

# In[551]:


from sklearn.metrics import confusion_matrix

# In[552]:


from sklearn.linear_model import LinearRegression

# In[553]:


from sklearn.preprocessing import LabelEncoder

# In[554]:


from sklearn.metrics import classification_report, confusion_matrix

# In[555]:


import pickle

# # Loading the dataset

# In[556]:


df = pd.read_csv('Reports Demographics1.csv', index_col="Computer Number")

# In[557]:


df

# # Data preparation

# In[558]:


df.columns

# In[559]:


df.rename(columns={"MajorDescription": "Major",
                   "MinorDescription": "Minor",
                   "Total number of courses": "Number of courses",

                   }, inplace=True)

# In[560]:


df

# In[561]:


# Checking for null values


# In[562]:


df.isnull()

# In[563]:


df.columns

# In[564]:


# Dropping columns with null values


# In[565]:


# df.dropna(subset=["Computer Number"], inplace=True)


# In[ ]:


df.dropna(subset=["Gender"], inplace=True)

# In[ ]:


df.dropna(subset=["Academic Year"], inplace=True)

# In[ ]:


df.dropna(subset=["Year Of Study"], inplace=True)

# In[ ]:


df.dropna(subset=["School"], inplace=True)

# In[ ]:


df.dropna(subset=["Program"], inplace=True)

# In[ ]:


df.dropna(subset=["Major"], inplace=True)

# In[ ]:


df.dropna(subset=["Minor"], inplace=True)

# In[ ]:


df.dropna(subset=["Number of courses"], inplace=True)

# In[ ]:


df.dropna(subset=["Sponsor"], inplace=True)

# In[ ]:


df.dropna(subset=["Accomodated"], inplace=True)

# In[ ]:


df.dropna(subset=["CA + Exam"], inplace=True)

# In[ ]:


df

# In[ ]:


# Checking for duplicates


# In[ ]:


df.duplicated()

# In[ ]:


len(df)

# In[ ]:


# df.drop_duplicates(["Computer Number"], keep="first", inplace=True)

# In[ ]:


len(df)

# # Exploratory data analysis

# In[ ]:


df["Gender"].unique()

# In[ ]:


df["Gender"].count()

# In[ ]:


df["Gender"].value_counts().plot(kind="barh", title="The gender of ict 1110 students", color="blue")


# # Data transformation

# In[567]:


def fxn_quiz_status(var_grade):
    if (var_grade >= 45):
        return "Pass"
    else:
        return "fail"


# In[568]:


df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# In[572]:


cxz = df
print(cxz)


# # Modeling

# In[ ]:


df_gender_exam_input = df[["Gender", "Examination status"]]

# In[ ]:


df_gender_exam_input

# In[ ]:


# Apply label encodes to X


# In[ ]:


df_gender_exam_input.replace({'M': 1, 'F': 0}, inplace=True)

# In[ ]:


df_gender_exam_input

# In[ ]:


# Apply label encodes to y


# In[ ]:


df_gender_exam_input.replace({"Pass": 1, "fail": 0}, inplace=True)

# In[ ]:


df_gender_exam_input

# # Create train/test sets

# In[ ]:


# Input x


# In[ ]:


x = np.array(df["Gender"]).reshape(-1, 1)

# In[ ]:


x.shape

# In[ ]:


# Output y


# In[ ]:


y = np.array(df["Examination status"]).reshape(-1, 1)

# In[ ]:


y.shape

# In[ ]:


# Splitting


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_gender_exam_input["Gender"],
                                                    df_gender_exam_input["Examination status"], test_size=0.20)

# In[ ]:


len(x_train)
len(y_train)

# In[ ]:


len(x_test)
len(y_test)

# # Model implementation using decision tree classifier

# In[ ]:


Gender = DecisionTreeClassifier()

# In[ ]:


Gender.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# # Making predictions using training and test data

# In[ ]:


# What the model has predicted


# In[ ]:


Gender.predict(x_test.values.reshape(-1, 1))

# In[ ]:


# The actual values


# In[ ]:


y_test.values

# # Model accuracy for gender

# In[ ]:


Gender.score(x_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))

# # Confusion matrix

# In[ ]:


Gender.predict(x_test.values.reshape(-1, 1))

# In[ ]:


y_test.values

# In[ ]:



# confusion_matrix(Gender.predict(x_test.values.reshape(-1,1),y_test.values))


# # Classification report

# In[ ]:


target_names = ['B+', 'B', 'C+', 'C', 'D+']

# In[ ]:


# print(classification_report(y_test, Gender.predict(x_test.values.reshape(1,-1))),target_names=target_names)


# # Loading the dataset for minor

# In[ ]:


df

# # Data preparation

# In[ ]:


# Checking for nul values


# In[ ]:


df.isnull()

# # Exploratory data analysis

# In[ ]:


df["Minor"].unique()

# In[ ]:


df["Minor"].count()

# In[ ]:


df["Minor"].value_counts()

# In[ ]:


df["Minor"].value_counts().plot(kind="barh", title="Minor Courses")


# # Data traansformation using one hot encoding

# In[ ]:


def fxn_quiz_status(var_grade):
    if (var_grade >= 90):
        return "A+"
    elif (var_grade >= 80):
        return ("A")
    elif (var_grade >= 70):
        return ("B+")
    elif (var_grade >= 60):
        return ("B")
    elif (var_grade >= 50):
        return ("C+")
    elif (var_grade >= 45):
        return ("C")
    elif (var_grade >= 40):
        return ("D+")
    else:
        return ("D")


# In[ ]:


df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# In[ ]:



# print(cxz)
# # Modeling

# In[ ]:


df_minor_exam_input = df[["Minor", "Examination status"]]

# In[ ]:


df_minor_exam_input.head(110)

# In[ ]:


df_minor_exam_input = pd.get_dummies(df_minor_exam_input, columns=["Minor"])

# In[ ]:



# In[ ]:


# Apply label encoders to labels


# In[ ]:


minor_exam_encoder = LabelEncoder()

# In[ ]:


minor_exam_encoder.fit(df_minor_exam_input["Examination status"])

# In[ ]:


minor_exam_encoder.classes_

# In[ ]:


minor_exam_encoder.fit(df_minor_exam_input["Examination status"])

# In[ ]:


minor_exam_encoder.classes_

# In[ ]:


df_minor_exam_input.head(60)

# In[ ]:


df_minor_exam_input["ExaminationGrade"] = minor_exam_encoder.transform(df_minor_exam_input["Examination status"])

# In[ ]:


df_minor_exam_input

# # Create/train tests

# In[ ]:


df_minor_exam_input.iloc[:, 1:10]

# In[ ]:


# Splitting


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_minor_exam_input.iloc[:, 1:10],
                                                    df_minor_exam_input["ExaminationGrade"], test_size=0.20)

# In[ ]:


# x input


# In[ ]:


x = np.array(df["Minor"]).reshape(-1, 1)

# In[ ]:


x.shape

# In[ ]:


# Y output


# In[ ]:


y = np.array(df["Examination status"]).reshape(-1, 1)

# In[ ]:


y.shape

# In[ ]:


len(x_train)
len(y_train)

# In[ ]:


len(x_test)
len(y_test)

# # Model implementation using random forest

# In[ ]:


Minor = LogisticRegression()

# In[ ]:


Minor.fit(x_train.values, y_train.values.reshape(-1, 1))

# # Making predictions using training and test data

# In[ ]:


# What the model has predicted


# In[ ]:


Minor.predict(x_test.values)

# In[ ]:


# Actual values


# In[ ]:


y_test.values

# # Model accuracy for minor

# In[ ]:


Minor.score(x_test.values, y_test.values.reshape(-1, 1))

# # Confusion matrix

# # Classification report

# In[ ]:


target_names = ['B', 'B+', 'C', 'C+', 'D', 'D+']

# In[ ]:


# print(classification_report(y_test, Minor.predict(x_test.values),target_names=target_names))


# # Loading the dataset for workload

# In[ ]:


df

# # Data preparation

# In[ ]:


df.isnull()

# In[ ]:


df.duplicated()

# In[ ]:


len(df)

# In[ ]:


# df.drop_duplicates(["Computer Number"], keep="first", inplace=True)

# In[ ]:


len(df)

# # Exploratory data analysis

# In[ ]:


df["Number of courses"].unique()

# In[ ]:


df["Number of courses"].count()

# In[ ]:


df["Number of courses"].value_counts()

# In[ ]:


df["Number of courses"].value_counts().plot(kind="barh", title="Each student's number of courses", color="blue")


# # Data transformation

# In[ ]:


def fxn_quiz_status(var_grade):
    if (var_grade >= 45):
        return "Pass"
    else:
        return "fail"


# In[ ]:


df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# In[ ]:

print("here")
print(cxz)

# # Modeling

# In[ ]:


df_workload_exam_input = df[["Number of courses", "Examination status"]]

# In[ ]:


df_workload_exam_input

# In[ ]:


# Apply label encodes  to x values


# In[ ]:


df_workload_exam_input.replace({5.0: 1, 4.0: 0}, inplace=True)

# In[ ]:


df_workload_exam_input

# In[ ]:


# Apply label encodes  to y values


# In[ ]:


df_workload_exam_input.replace({"Pass": 1, "fail": 0}, inplace=True)

# In[ ]:


df_workload_exam_input

# # Create/train tests

# In[ ]:


# Input x


# In[ ]:


X = np.array(df["Number of courses"]).reshape(-1, 1)

# In[ ]:


X.shape

# In[ ]:


# Output y


# In[ ]:


Y = np.array(df["Examination status"]).reshape(-1, 1)

# In[ ]:


Y.shape

# In[ ]:


# Splitting


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_workload_exam_input["Number of courses"],
                                                    df_workload_exam_input["Examination status"], test_size=0.20)

# In[ ]:


len(x_train)
len(y_train)

# In[ ]:


len(x_test)
len(y_test)

# # Model implementation using logistic regression

# In[ ]:


workload = LogisticRegression()

# In[ ]:


workload.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# # Model implementation using training and test data

# In[ ]:


# What the model has predicted


# In[ ]:


workload.predict(x_test.values.reshape(-1, 1))

# In[ ]:


# The actual values


# In[ ]:


y_test.values

# # Model accuracy for workload

# In[ ]:


workload.score(x_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))

# # Confusion matrix

# In[ ]:


# confusion_matrix


# # Classification report

# In[ ]:


target_names = ['B', 'B+', 'C', 'C+', 'D', 'D+']

# In[ ]:


# print(classification_report(y_test, workload.predict(x_test.values),target_names=target_names))


# # Loading dataset for sponsor

# In[ ]:


df
ds = df

# # Data preparation

# In[ ]:


df.isnull()

# In[ ]:


df.duplicated()

# In[ ]:


len(df)

# In[ ]:


# df.drop_duplicates(["Computer Number"], keep="first", inplace=True)


# In[ ]:


len(df)

# # Exploratory data analysis

# In[ ]:


df["Sponsor"].unique()

# In[ ]:


df["Sponsor"].count()

# In[ ]:


df["Sponsor"].value_counts()

# In[ ]:


df["Sponsor"].value_counts().plot(kind="barh", title="Sponsorship status")


# # Data transformation using one hot encoding

# In[ ]:


def fxn_quiz_status(var_grade):
    if (var_grade >= 90):
        return "A+"
    elif (var_grade >= 80):
        return ("A")
    elif (var_grade >= 70):
        return ("B+")
    elif (var_grade >= 60):
        return ("B")
    elif (var_grade >= 50):
        return ("C+")
    elif (var_grade >= 45):
        return ("C")
    elif (var_grade >= 40):
        return ("D+")
    else:
        return ("D")


# In[ ]:


df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# In[ ]:


df

# # Modeling

# In[ ]:


df_sponsor_exam_input = df[["Sponsor", "Examination status"]]

# In[ ]:


df_sponsor_exam_input

# In[ ]:


df_sponsor_exam_input = pd.get_dummies(df_sponsor_exam_input, columns=["Sponsor"])

# In[ ]:


df_sponsor_exam_input

# In[ ]:


# Apply label encoders to labels


# In[ ]:


sponsor_exam_encoder = LabelEncoder()

# In[ ]:


sponsor_exam_encoder.fit(df_minor_exam_input["Examination status"])

# In[ ]:


sponsor_exam_encoder.classes_

# In[ ]:


sponsor_exam_encoder.fit(df_minor_exam_input["Examination status"])

# In[ ]:


sponsor_exam_encoder.classes_

# In[ ]:


df_sponsor_exam_input

# In[ ]:


df_sponsor_exam_input["ExaminationGrade"] = sponsor_exam_encoder.transform(df_minor_exam_input["Examination status"])

# In[ ]:


df_sponsor_exam_input

# # Create train/train tests

# In[ ]:


df_sponsor_exam_input

# In[ ]:


# Splitting


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_sponsor_exam_input.iloc[:, 1:10],
                                                    df_sponsor_exam_input["ExaminationGrade"], test_size=0.20)

# In[ ]:


# Input x


# In[ ]:


x = np.array(df["Sponsor"]).reshape(-1, 1)

# In[ ]:


x.shape

# In[ ]:


# output y


# In[ ]:


y = np.array(df["Examination status"]).reshape(-1, 1)

# In[ ]:


y.shape

# In[ ]:


len(x_train)
len(y_train)

# In[ ]:


len(x_test)
len(y_test)

# # Model implementation using logistic regression

# In[ ]:


sponsor_exam = LogisticRegression()

# In[ ]:


sponsor_exam.fit(x_train.values, y_train.values.reshape(-1, 1))

# # Making predictions using model and test data

# In[ ]:


# Model prediction


# In[ ]:


sponsor_exam.predict(x_test.values)

# In[ ]:


# Actual values


# In[ ]:


y_test.values

# # Model accuracy for sponsorship

# In[ ]:


sponsor_exam.score(x_test.values, y_test.values.reshape(-1, 1))

# # Classification report

# In[ ]:


target_names = ['B', 'B+', 'C', 'C+', 'D', 'D+']

# In[ ]:


# print(classification_report(y_test,sponsor_exam.predict(x_test.values),target_names=target_names))


# # Loading the dataset for accomodation

# In[ ]:


df

# # Data preparation

# In[ ]:


df.isnull()

# In[ ]:


df.duplicated()

# In[ ]:


len(df)

# In[ ]:


# df.drop_duplicates(["Computer Number"], keep="first", inplace=True)


# In[ ]:


len(df)

# # Exploratory data analysis

# In[ ]:


df["Accomodated"].unique()

# In[ ]:


df["Accomodated"].unique()

# In[ ]:


df["Accomodated"].value_counts()

# In[ ]:


df["Accomodated"].value_counts().plot(kind="barh", title="Students accomodation status", color="blue")


# # Data transformation

# In[ ]:


def fxn_quiz_status(var_grade):
    if (var_grade >= 45):
        return "Pass"
    else:
        return "fail"


# In[ ]:


df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# In[ ]:


data_set = df
data_set
print(cxz)

# # Modeling

# In[ ]:


df_accomodation_exam_input = df[["Accomodated", "Examination status"]]

# In[ ]:


df_accomodation_exam_input

# In[ ]:


# Apply label encodes  to x values


# In[ ]:


df_accomodation_exam_input.replace({"Yes": 1, "No": 0}, inplace=True)

# In[ ]:


df_accomodation_exam_input

# In[ ]:


# Apply label encodes  to y values


# In[ ]:


df_accomodation_exam_input.replace({"Pass": 1, "fail": 0}, inplace=True)

# In[ ]:


df_accomodation_exam_input

# # Create train/train tests

# In[ ]:


# Input x


# In[ ]:


X = np.array(df["Accomodated"]).reshape(-1, 1)

# In[ ]:


X.shape

# In[ ]:


# Output y


# In[ ]:


Y = np.array(df["Examination status"]).reshape(-1, 1)

# In[ ]:


Y.shape

# In[ ]:


# Splitting


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_accomodation_exam_input["Accomodated"],
                                                    df_accomodation_exam_input["Examination status"], test_size=0.20)

# In[ ]:


len(x_train)
len(y_train)

# In[ ]:


len(x_test)
len(y_test)

# # Model implementation

# In[ ]:


accomodation = DecisionTreeClassifier()

# In[ ]:


accomodation.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# # Making training and test data

# In[ ]:


# What the model has predicted


# In[ ]:


accomodation.predict(x_test.values.reshape(-1, 1))

# In[ ]:


# The actual values


# In[ ]:


y_test.values

# # Model accuracy for accomodation

# In[ ]:


accomodation.score(x_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))

# # Classification report

# In[ ]:


target_names = ['B', 'B+', 'C', 'C+', 'D+', 'D']

# In[ ]:


# (classification_report(accomodation.predict(x_test.values),y_test.values.reshape(-1,1))


# # Loading the moodle logins dataset

# In[ ]:


df

# # Data preparation

# In[ ]:


df.isnull()

# In[ ]:


df.duplicated()

# In[ ]:


len(df)

# In[ ]:


# df.drop_duplicates(["Computer Number"], keep="first", inplace=True)


# In[ ]:


len(df)

# # Exploratory data analysis

# In[ ]:


df["Moodle logins"].unique()

# In[ ]:


df["Moodle logins"].count()

# In[ ]:


df["Moodle logins"].value_counts()

# In[ ]:


df["Moodle logins"].value_counts().plot(kind="barh", title="Moodle login status")


# # Data transformation using one hot encoding

# In[ ]:


def fxn_quiz_status(var_grade):
    if (var_grade >= 90):
        return "A+"
    elif (var_grade >= 80):
        return ("A")
    elif (var_grade >= 70):
        return ("B+")
    elif (var_grade >= 60):
        return ("B")
    elif (var_grade >= 50):
        return ("C+")
    elif (var_grade >= 45):
        return ("C")
    elif (var_grade >= 40):
        return ("D+")
    else:
        return ("D")


# In[ ]:


df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# In[ ]:


df

# # Modeling

# In[ ]:


df_moodle_exam_input = df[["Moodle logins", "Examination status"]]

# In[ ]:


df_moodle_exam_input

# In[ ]:


df_moodle_exam_input = pd.get_dummies(df_moodle_exam_input, columns=["Moodle logins"])

# In[ ]:


df_moodle_exam_input

# In[ ]:


# Apply label encoders to labels


# In[ ]:


moodle_exam_encoder = LabelEncoder()

# In[ ]:


moodle_exam_encoder.fit(df_minor_exam_input["Examination status"])

# In[ ]:


LabelEncoder()

# In[ ]:


moodle_exam_encoder.classes_

# In[ ]:


moodle_exam_encoder.fit(df_minor_exam_input["Examination status"])

# In[ ]:


moodle_exam_encoder.classes_

# In[ ]:


df_moodle_exam_input.head(30)

# In[ ]:


df_moodle_exam_input["ExaminationGrade"] = moodle_exam_encoder.transform(df_moodle_exam_input["Examination status"])

# In[ ]:


df_moodle_exam_input

# # Create/train sets

# In[ ]:


df_minor_exam_input.iloc[:, 1:10]

# In[ ]:


# Splitting


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_moodle_exam_input.iloc[:, 1:10],
                                                    df_moodle_exam_input["ExaminationGrade"], test_size=0.20)

# In[ ]:


# x input


# In[ ]:


x = np.array(df["Moodle logins"]).reshape(-1, 1)

# In[ ]:


x.shape

# In[ ]:


# Y output


# In[ ]:


y = np.array(df["Examination status"]).reshape(-1, 1)

# In[ ]:


y.shape

# In[ ]:


len(x_train)
len(y_train)

# In[ ]:


len(x_test)
len(y_test)

# # Model implementation

# In[ ]:


moodle_exam = LogisticRegression()

# In[ ]:


moodle_exam.fit(x_train.values, y_train.values.reshape(-1, 1))

# # Making predictions using training and test data

# In[ ]:


# What the model has predicted


# In[ ]:


moodle_exam.predict(x_test.values)

# In[ ]:


# Actual values


# In[ ]:


y_test.values

# # Model accuracy for moodle logins

# In[ ]:


moodle_exam.score(x_test.values, y_test.values.reshape(-1, 1))

# # Classification report

# In[ ]:


target_names = ['B', 'B+', 'C', 'C+', 'D']

# In[ ]:

def fxn_quiz_status(var_grade):
    if (var_grade >= 45):
        return "Pass"
    else:
        return "fail"

# In[ ]:

df["Examination status"] = df["CA + Exam"].apply(fxn_quiz_status)

# def bodyData(d):
#     jsn = jsonify(d).json
#     print(jsn['id'])
#     data = pd.DataFrame({"Computer Number":[], "Gender":[], "MinorDescription":[], "Total number of courses":[], "Sponsor":[], "Accomodation":[], "CA + Exam":[]})
#     done = df.append((), ignore_index=True).item()
#     # print(data.())
#     if done:
#         print(df)
#         return done
#     else:
#         print("Error")




def predictML(v):
    data = cxz.loc[v]
    data = data.to_json(orient='columns')
    # print(data)
    return data

predictML(2001)






