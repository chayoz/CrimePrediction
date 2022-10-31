# -*- coding: utf-8 -*-
import pandas as pd
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from pandas import json_normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

states = ["AZ(Arizona)","MO(Missouri)","OH(Ohio)","WA(Washington)","TX(Texas)"]
checkstate = ["AZ","MO","OH","WA","TX"]
print("Available States:")
for s in states:
    print(s)
print()
state = input("Please select a state by typing the corresponding abbreviation: ")

# Calculate crime rate for chosen location
def crime_rate(df):
    rate = 0
    rates = []
    df["total_crimes"] = df.drop(columns=['state_id','state_abbr','year','population','burglary','larceny','motor_vehicle_theft']).sum(axis=1)
    for y in range(1985,2021):
        df_row = df.loc[df['year'] == y]
        rate = df_row.iloc[0]['total_crimes'] / df_row.iloc[0]['population'] * 100000
        rates.append(rate)
        #print("Crime rate in {0}: {1:.1f}".format(y,rate))
    df['crime_rate'] = rates
    return df

# Shows clustered data
def show_cl_kmeans(X_train):
    Sum_of_squared_distances = []
    K = range(1,10)
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X_train)
        Sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K,Sum_of_squared_distances,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title('Elbow Method For Optimal k')
    plt.show()

# K-means clusters accuracy
def show_acc_kmeans(x,label,model):
    #plotting the results
    plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=label, s=50, cmap='viridis')
    
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h).astype(np.int8), np.arange(y_min, y_max, h).astype(np.int8))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Shows SVM accuracy
def show_acc_svm(x,y,clf,labels):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC ')
    # Set-up grid for plotting.
    X0, X1 = x.iloc[:, 0], x.iloc[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('Total Crimes')
    ax.set_xlabel('Year')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()
    print()
    
def kmeans(df):
    df = df.fillna(0)
    #Scale Data
    scaler = MinMaxScaler()
    #df = df.drop(['state_id','state_abbr','year','property_crime','total_crimes'], axis=1)
    df_scaled = scaler.fit_transform(df[['year','total_crimes','crime_rate']])
    #df_scaled = pd.DataFrame(df_scaled, columns=['population','violent_crime','homicide','rape_legacy','rape_revised','robbery','aggravated_assault','burglary','larceny','motor_vehicle_theft','arson','crime_rate'])
    df_scaled = pd.DataFrame(df_scaled, columns=['year','total_crimes','crime_rate'])
    x = df_scaled[['year','total_crimes']]
    y = df_scaled[['crime_rate']]
    
    #X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = KMeans(n_clusters=3)
    label = model.fit_predict(x)
    #filter rows of original data
    #filtered_label0 = x[label == 0]

    #print(model.cluster_centers_)
    
    #elbow method for optimal K
    #show_cl_kmeans(x)
    
    #Show clusters grouping
    show_acc_kmeans(x,label,model)

    print()

def svm(df):
    from sklearn import svm
    df = df.fillna(0)
    #Scale Data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['year','total_crimes','crime_rate']])
    #df_scaled = pd.DataFrame(df_scaled, columns=['population','violent_crime','homicide','rape_legacy','rape_revised','robbery','aggravated_assault','burglary','larceny','motor_vehicle_theft','arson','crime_rate'])
    df_scaled = pd.DataFrame(df_scaled, columns=['year','total_crimes','crime_rate'])
    #x = df.drop('crime_rate', axis=1)
    x = df_scaled[['year','total_crimes']]
    y = df[['crime_rate']].round().astype(int)
    
    model = svm.SVC(kernel='linear', C = 1.0)
    clf = model.fit(x, y.values.ravel())
    labels = clf.classes_
    
    show_acc_svm(x,y,clf,labels)
    
if (state in checkstate):
    response = requests.get("https://api.usa.gov/crime/fbi/sapi/api/estimates/states/"+state+"/1985/2020?API_KEY=<enter API key here>")
    df = json_normalize(response.json()['results'])
    #df = df.sort_values(by ='year')
    df = crime_rate(df)
    checkstate.remove(state)
    print("Which algorithm would you like to use? \n1)K-means \n2)SVM")
    option = input()
    
    if(option=="1"):
        kmeans(df)
    elif(option=="2"):
        svm(df)
    else:
        print("Invalid Input")
else:
    print("Invalid Input")
    
