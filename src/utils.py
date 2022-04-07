from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from datetime import datetime
import os
import pandas as pd

def predict_and_show_metrics(trained_model, X_train, y_train, X_val, y_val,yscaler):
    """
    Use trained to make predictions and compute performance metrics (R2, RMSE and RPD)
    :param trained_model: trained model
    :param X_train: NIR train data
    :param y_train: train target
    :param X_val: NIR test data
    :param y_val: test target
    :param yscaler: scaler to convert predictions to original scale
    :return:
    """
    print('Saving model..')
    now = datetime.now()
    time = now.strftime("%Y-%m-%d")
    #savefn = target+'_DLmodel_'+sensor+time
    #training_model.save(os.path.join('./saved_models/',savefn),save_format='h5')
    print('Saving ended')
    #Use model to make predictions
    print('Making predictions')
    y_train_pred = trained_model.predict(X_train)
    y_val_pred = trained_model.predict(X_val)
    y_train_pred, y_val_pred = y_train_pred.reshape(-1,1), y_val_pred.reshape(-1,1)

    #Compute R2
    r2_train = r2_score(y_train.flatten(),y_train_pred.flatten())
    r2_val = r2_score(y_val.flatten(),y_val_pred.flatten())

    #Recompute to original scale
    y_train_orig = yscaler.inverse_transform(y_train)
    y_train_pred_orig = yscaler.inverse_transform(y_train_pred)
    y_val_orig = yscaler.inverse_transform(y_val)
    y_val_pred_orig = yscaler.inverse_transform(y_val_pred)

    #Compute RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train_orig,y_train_pred_orig))
    rmse_val = np.sqrt(mean_squared_error(y_val_orig,y_val_pred_orig))

    #Compute RPD = std(reference data) / RMSE
    rpd_train = np.std(y_train_orig)/rmse_train
    rpd_val = np.std(y_val_orig)/rmse_val

    #Print metrics
    print("\n\n\n==============================")
    print("\n\n\n Evaluation metrics")
    print("\n\nR2 train: ",r2_train)
    print("RMSE train: ",rmse_train)
    print("RPD train: ",rpd_train)
    print("R2 val: ",r2_val)
    print("RMSE val: ",rmse_val)
    print("RPD val: ",rpd_val)
    print("\n\n\n==============================")

    return y_train_orig,y_train_pred_orig,y_val_orig,y_val_pred_orig


def export_results(y_train_orig,y_train_pred_orig,train_metadata,
                   y_val_orig,y_val_pred_orig,test_meta, target, model_type='CNN',sensor='Foss',crop='peanut'):

    target = ''.join(filter(str.isalnum, target))
    save_dir = "../results/predictions"
    dtObj = datetime.utcnow()
    time = dtObj.strftime("%d-%b-%Y_%H-%M-%S")

    if model_type !='CNN':
        #Calibration
        exports_cal = {'Genotype':train_metadata["Wet lab_ID"],'ytrain':y_train_orig.flatten(),'ytrain_pred':y_train_pred_orig.flatten()}
        exports_cal = pd.DataFrame(exports_cal)
        filename_cal = crop+'_'+target+'_cal_'+model_type+'_'+sensor+'_'+time+'_.xlsx'
        exports_cal.to_excel(os.path.join(save_dir,filename_cal),index=False)

    #Validation
    exports_val = {'Wet_lab_ID':test_meta["Wet lab_ID"],'yval':y_val_orig.flatten(),'yval_pred':y_val_pred_orig.flatten()}
    exports_val = pd.DataFrame(exports_val)
    filename_val = crop+'_'+target+'_val_'+model_type+'_'+sensor+'_'+time+'_.xlsx'
    exports_val.to_excel(os.path.join(save_dir,filename_val),index=False)
