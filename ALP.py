import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox 

df = pd.read_csv('Credit_card.csv')
df.head()

df.drop('Mobile_phone', axis=1, inplace=True)
df.drop('Work_Phone', axis=1, inplace=True)
df.drop('Phone', axis=1, inplace=True)
df.drop('EMAIL_ID', axis=1, inplace=True)
df.drop(columns='Ind_ID', inplace=True)

df['Type_Occupation'] = df['Type_Occupation'].map({'Core staff':0,'Laborers':1, 'Sales staff':2, 'Accountants':3, 'High skill tech staff':4, 'Managers':5,
                                                   'Cleaning staff':6,'Drivers':7, 'Low-skill Laborers':8, 'IT staff':9, 'Cooking staff':10, 'Waiters/barmen staff':11,
                                                   'Security staff':12,'Medicine staff':13, 'Private service staff':14, 'HR staff':15, 'Secretaries':16, 'Realty agents':17})

df['Birthday_count'] = (df['Birthday_count'])/365
df['Birthday_count'] = df['Birthday_count'].abs()

df['Employed_days']= (df['Employed_days'])/365
df['Employed_days'] = -df['Employed_days']
df['GENDER'].fillna(df['GENDER'].mode()[0], inplace=True)
income_mean = df.groupby(['Type_Income'])['Annual_income'].mean()
df['Annual_income'] = df['Annual_income'].fillna(df.groupby('Type_Income')['Annual_income'].transform('mean'))

age_mean = df.groupby(['Type_Income'])['Birthday_count'].mean()

df['Birthday_count'] = df['Birthday_count'].fillna(df.groupby('Type_Income')['Birthday_count'].transform('mean'))

df['GENDER'] = df['GENDER'].map({'M':1,'F':0})
df['Car_Owner'] = df['Car_Owner'].map({'Y':1,'N':0})
df['Propert_Owner'] = df['Propert_Owner'].map({'Y':1,'N':0})

df['EDUCATION'] = df['EDUCATION'].map({'Lower secondary':0,'Secondary / secondary special':1, 'Incomplete higher':2, 'Higher education':3, 'Academic degree':4})
df['Type_Income'] = df['Type_Income'].map({'Pensioner':0,'Commercial associate':1, 'Working':2, 'State servant':3})
df['Marital_status'] = df['Marital_status'].map({'Single / not married':0,'Married':1, 'Civil marriage':2, 'Separated':3, 'Widow':4})
df['Housing_type'] = df['Housing_type'].map({'With parents':0,'Co-op apartment':1, 'Office apartment':2, 'Municipal apartment':3, 'Rented apartment':4, 'House / apartment':5})

df_missing = df[df['Type_Occupation'].isnull()]
df_not_missing = df.dropna(subset=['Type_Occupation'])

X_train = df_not_missing[['Birthday_count', 'Employed_days', 'Annual_income', 'Family_Members', 'Marital_status', 'Type_Income', 'EDUCATION', 'Housing_type', 'CHILDREN', 'Propert_Owner', 'GENDER', 'Car_Owner']]
y_train = df_not_missing['Type_Occupation']

reg_model = RandomForestClassifier(n_estimators=200, max_depth=35)
reg_model.fit(X_train, y_train)

X_missing = df_missing[['Birthday_count', 'Employed_days', 'Annual_income', 'Family_Members', 'Marital_status', 'Type_Income', 'EDUCATION', 'Housing_type', 'CHILDREN', 'Propert_Owner', 'GENDER', 'Car_Owner']]
predicted_occupations = reg_model.predict(X_missing)

df.loc[df['Type_Occupation'].isnull(), 'Type_Occupation'] = predicted_occupations

df['Type_Occupation'].head()

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



classifier = RandomForestClassifier(n_estimators=200, max_depth=35)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)

print()

cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

print("Accuracy Score: {:0.4f}".format(accuracy_score(y_test, y_pred)))

print()

y_pred_proba = classifier.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_proba)
print(f'Log Loss: {logloss:.4f}')

def make_scrollable(container):
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return scrollable_frame

window = tk.Tk()
window.title("Credit Card Approval")

title_label = ttk.Label(window, text="Credit Card Approval", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

feature_labels = [
    "Gender", "Car Owner", "Property Owner", "Number of Children",
    "Annual Income", "Type of Income", "Education Level",
    "Marital Status", "Housing Type", "Age",
    "Employed Years", "Type Occupation", "Family Members"
]

feature_entries = {}

feature_options = {
    "Gender": ["Female", "Male"],
    "Car Owner": ["No", "Yes"],
    "Property Owner": ["No", "Yes"],
    "Number of Children": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    "Type of Income": ["Pensioner", "Commercial associate", "Working", "State servant"],
    "Education Level": ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"],
    "Marital Status": ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"],
    "Housing Type": ["With parents", "Co-op apartment", "Office apartment", "Municipal apartment", "Rented apartment", "House / apartment"],
    "Type Occupation": ['Core staff', 'Laborers', 'Sales staff', 'Accountants', 'High skill tech staff', 'Managers',
                        'Cleaning staff', 'Drivers', 'Low-skill Laborers', 'IT staff', 'Cooking staff', 'Waiters/barmen staff',
                        'Security staff', 'Medicine staff', 'Private service staff', 'HR staff', 'Secretaries', 'Realty agents']
}

scrollable_frame = make_scrollable(window)

# Create entry widgets or dropdowns for features
for label in feature_labels:
    frame = ttk.Frame(scrollable_frame)
    frame.pack(padx=10, pady=5, fill="both", expand=True)

    ttk.Label(frame, text=label).grid(row=0, column=0, padx=5, pady=5, sticky="w")

    if label in feature_options:
        # Dropdown for features with options
        feature_var = tk.StringVar()
        feature_var.set(feature_options[label][0])  # Default selection
        feature_menu = ttk.Combobox(frame, textvariable=feature_var, values=feature_options[label], state="readonly")
        feature_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        feature_entries[label] = feature_var
    else:
        # Entry for other features
        entry = ttk.Entry(frame)
        entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        feature_entries[label] = entry

# function to make predictions
def predict_approval():

    if any(entry.get() == "" for entry in feature_entries.values()):
        messagebox.showinfo("Warning", "Please fill out all of the questions.")
        return

    input_values = [
        float(entry.get()) if label not in feature_options else feature_options[label].index(entry.get())
        for label, entry in feature_entries.items()
    ]

    new_data_point = np.array(input_values).reshape(1, -1)
    new_data_point = sc.transform(new_data_point)

    new_prediction = classifier.predict(new_data_point)

    if new_prediction[0] == 0:
        messagebox.showinfo("Credit Card Application Result", "Congratulations, your credit card application is likely to be accepted.")
    else:
        messagebox.showinfo("Credit Card Application Result", "Sorry, but your credit card application is likely to be rejected because you haven't met the criteria.")

predict_button = ttk.Button(window, text="Predict Application", command=predict_approval)
predict_button.pack(pady=10)

result_label = ttk.Label(window, text="")
result_label.pack(pady=10)

window.mainloop()