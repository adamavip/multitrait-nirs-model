import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from preprocessing import preprocess
from augmentation import perform_dataaugmentation


def load_data(fname=None, target=None, sensor=None):

    d = pd.read_csv(fname)


    # Convert Protein to float
    #d[target] = pd.to_numeric(d[target], errors='coerce', downcast='float')

    #Gather response and predictors in dataframe
    #d_corr = pd.concat([d["Species"].to_frame(), d[target].to_frame(), d.loc[:,'1350.155463':]], axis = 1)

    # Delete na
    #d_corr = d_corr.dropna(axis=0)

    #train = d[d['Set']=='Training']
    #test = d[d['Set']=='Validation']
    train, test = train_test_split(d, test_size=0.2, random_state=42, shuffle=True)
    train.to_csv('./data/train.csv',index=False)
    test.to_csv('./data/test.csv',index=False)
    #export shuffled data
    #train['Set']='Training'
    #test['Set']='Validation'
    #new=pd.concat([train,test],axis=0)
    #new.to_excel('./data/shuffled_multicereal_evt5_2021.xlsx',index=False)

    #Build predictors
    #if sensor=="hone_ag":
    x_train = train.loc[:,'400':]
    #x_train = train[important_features]
    #xcolnames = x_train.columns
    x_test = test.loc[:,'400':]
    #x_test = test[important_features]

    #Extract targets
    y_train = train[target]
    y_test = test[target]

    #Convert to numpy arrays
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    #Data augmentation
    x_train, y_train = perform_dataaugmentation(x_train, y_train)

    #Create weighted predictors based on feature importance
    # x_train = weight_predictors(x_train,y_train)
    # x_test = weight_predictors(x_test,y_test)

    #Extract sample metadata from test set
    #if sensor=="hone_ag":
    train_set_metadata = train[["Genotype"]]
    test_set_metadata = test[["Genotype"]]



    #Apply preprocessing operations
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)



    #Minmax Normalization of x and y
    xscaler = MinMaxScaler(feature_range=(0.1,0.9))
    x_train = xscaler.fit_transform(x_train)
    x_test = xscaler.transform(x_test)


    yscaler = MinMaxScaler(feature_range=(0.1,0.9))
    y_train = yscaler.fit_transform(y_train.reshape(-1,1))
    y_test = yscaler.transform(y_test.reshape(-1,1))



    # Reshaping arrays
    x_train,x_test = np.expand_dims(x_train, axis=2),np.expand_dims(x_test, axis=2)
    #y_train,y_test = y_train.reshape(-1,1),y_test.reshape(-1,1)


    return x_train, y_train, x_test, y_test,yscaler, train_set_metadata, test_set_metadata
