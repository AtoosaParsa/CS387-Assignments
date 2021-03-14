import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 

# I used the code from https://www.kaggle.com/shweta2407/regression-on-housing-data-accuracy-87 to preprocess the data
def z_score(data, column):
    #creating global variables for plotting the graph for better demonstration
    global zscore, outlier
    #creating lists to store zscore and outliers 
    zscore = []
    outlier =[]
    # for zscore generally taken thresholds are 2.5, 3 or 3.5 hence i took 3
    threshold = 3
    # calculating the mean of the passed column
    mean = np.mean(data[column])
    # calculating the standard deviation of the passed column
    std = np.std(data[column])
    for i in data[column]:
        z = (i-mean)/std
        zscore.append(z)
        #if the zscore is greater than threshold = 3 that means it is an outlier
        if np.abs(z) > threshold:
            outlier.append(i)
    print('total outliers', len(outlier))
    return outlier
    
def preprocessing(data):
    data.head()
    
    #see the datatype of each column
    data.info()
    
    #fill all the values with 0
    data.fillna(0, inplace=True)
    
    #iterate through the columns to see the frequency of different values
    for i in data.columns:
        print(data[i].value_counts())
    
    #format the date
    d =[]
    for i in data['date'].values:
        d.append(i[:4])
        
    data['date'] = d
    
    # convert everything to same datatype
    for i in data.columns:
        data[i]=data[i].astype(float)
        
    #make a new column age of the house  
    data['age'] = data['date'] - data['yr_built']
    
    #calculate the total years of renovation
    data['renov_age'] = np.abs(data['yr_renovated'] - data['yr_built'])
    data['renov_age'] = data.renov_age.apply(lambda x: x if len(str(int(x)))==2 else 0.0)
    
    #remove unwanted columns like yr_built, date, id
    data.drop(['id','date', 'yr_built', 'yr_renovated'], axis=1, inplace=True)
    data.head()
    
    #print highly correlated variables
    corr_features =[]
    
    for i , r in data.corr().iterrows():
        k=0
        for j in range(len(r)):
            if i!= r.index[k]:
                if r.values[k] >=0.5:
                    corr_features.append([i, r.index[k], r.values[k]])
            k += 1
    corr_features
    
    #let us remove highly correlated features that is above 0.8
    feat =[]
    for i in corr_features:
        if i[2] >= 0.8:
            feat.append(i[0])
            feat.append(i[1])
            
    data.drop(list(set(feat)), axis=1, inplace=True)
    data.head()
    
    #plotting outliers graph for 'price' feature 
    outlier = z_score(data, 'price')
    
    
    #remove the outliers from price using zscore
    dj=[]
    for i in data.price:
        if i in set(outlier):
            dj.append(0.0)
        else:
            dj.append(i)
            
    data['P'] = dj
    
    x = data.drop(data[data['P'] == 0.0].index) 
    x.shape
    
    #defining the independent and dependent variable
    X = x.drop(['price','P'], axis=1)
    Y = x['price']
    
    #isolation forest
    import warnings
    warnings.filterwarnings(action='ignore')
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.1)
    outlier = iso.fit_predict(data)
    
    
    
    #mask variable contains all the outliers
    mask = outlier == -1
    #task variable contains all the non-outliers data
    task = outlier != -1
    #creating dataframe containing outliers
    df_1 = data[mask]
    #creating dataframe containing non-outliers
    df_2 = data[task]
    
    
    y2 = df_2['price']
    df_2.drop(['price','P'], axis=1, inplace=True)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    pred = lr.predict(x_test)
    r2_score(y_test, pred)
    
    return X, Y