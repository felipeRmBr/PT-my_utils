# textModelSpaceScanner() *********************************************************************

# adapted from Talos solution
# (check: https://autonomio.github.io/talos/#/)
import pickle
from itertools import product
from pandas import DataFrame
from statistics import mean 

# svmSpaceScanner() ********************************************************************************
## TODO:  mean_acc     -> mean_score
##        mean_val_acc -> mean_val_score

from itertools import product
from pandas import DataFrame
import pickle

from microtc.textmodel import TextModel

def svcSpaceScanner(X_train, Y_train, task, tm_search_space, tm_params_keys, 
                    svm_search_space, svm_params_keys, n_folds, metrics, 
                    partial_CV=False, backup_file = 'testing.df', backup_freq=25, 
                    save_textmodels=False, save_classifiers=False):

  """

  """

  if n_folds<2:
    print('Invalid n_folds value. Expected n_folds >= 2.')
    return None
  
  print('SCANNING SEARCH-SPACE\nBackupFile: {}'.format(backup_file))

  results_list = []
  for tm_idx, tm_params_settings in enumerate(tm_search_space):

    print(f'\nRunning experiments with TM-{tm_idx}  ({tm_idx+1}/{len(tm_search_space)})')
  
    #print(tm_params_dict)
    tm_ID = get_random_string(6)

    print('\nPreparing transformed data-splits   [', end='')
    tm_params_dict = dict(zip(tm_params_keys, tm_params_settings))

    # instantiate a TextModel object
    text_model=TextModel(**tm_params_dict)

    #text_model = instantiateTextModel(TM_params_dict)
    data_splits = getTransformedDataSplits(X_train, Y_train, task, text_model, 
                                           n_folds, partial_CV, tm_ID, save_textmodels)
    
    print(']\n')

    print('TESTING SVM CONFIGURATIONS\n')
    svm_space_size = len(svm_search_space)
    
    for svm_idx, svm_params_settings in enumerate(svm_search_space, start=1):
      svm_params_dict = dict(zip(svm_params_keys, svm_params_settings))

      # get a ramdom conf_ID
      conf_ID = get_random_string(6)

      # print some information about the search status
      fraction_string = '{}/{} '.format(svm_idx, svm_space_size).ljust(7, ' ')
      print(fraction_string, end='' )
      print(f'conf_ID: {conf_ID}   ', end = '')
      kernel_string = '{}'.format(svm_params_dict['kernel']).ljust(9, ' ')
      c_string      = '{}'.format(svm_params_dict['C']).ljust(8, ' ')
      print('kernel = {}'.format(kernel_string), end = '')
      print('C = {}'.format(c_string), end = '')

      # call svmCrossValidation() to stimate performance
      validation_results = svmCrossValidation(data_splits, 
                                              **svm_params_dict, 
                                              config_ID=conf_ID,
                                              partial_CV=partial_CV,
                                              return_train=True,
                                              save_classifiers=save_classifiers)
      
      result_dict = formResultDictionary(validation_results,
                                         tm_params_dict,
                                         svm_params_dict,
                                         conf_ID,
                                         tm_ID)

      # display progress and results
      progress_string    = '[{}% complete]'.format(round(100*(svm_idx)/svm_space_size,2)).ljust(17, ' ')
      print('  {}'.format(progress_string), end='')

      for metric in metrics:
        mean_score = result_dict['mean_{}'.format(metric)]
        mean_val_score = result_dict['mean_val_{}'.format(metric)]
        mean_score_str     = 'mean_{} = {},'.format(metric, round(mean_score,3)).ljust(19,' ')
        mean_val_score_str = 'mean_val_{} = {}'.format(metric, round(mean_val_score,3))
        print(' -- {}  {}'.format(mean_score_str, mean_val_score_str))
      
      # update the results_list
      results_list.append(result_dict)

    # save at the end of every text_model testing
    backup_file_name = f'{backup_file}.partial'
    with open(backup_file_name, 'wb') as file_handler:
      pickle.dump(DataFrame(results_list), file_handler)
        
  results_df = DataFrame(results_list)

  backup_file_name = backup_file
  with open(backup_file_name, 'wb') as file_handler:
    pickle.dump(results_df, file_handler)

  print('\n')

  return results_df

# svmCrossValidation()  ****************************************************************************
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

def svmCrossValidation(data_splits, kernel, C, config_ID, partial_CV=False, 
                       return_train=True, save_classifiers=False):
  '''
  Performs a k-fold cross validation with the given svm_configuration 

  input:

  output:

  '''
  k = len(data_splits)

  if save_classifiers:
    # Weights are initailly saved in a temp directory
    trained_classif_dir_path  = f'./trained_models/{config_ID}' 
    if not path.exists(trained_classif_dir_path):
      makedirs(trained_classif_dir_path)
  else:
    trained_classif_dir_path = ''

  acc_results = []
  val_acc_results = []
  f1_results = []
  val_f1_results = []

  for fold_idx, data_split in enumerate(data_splits):
    if partial_CV & (fold_idx%2==1):
        continue

    x_train, y_train, x_val, y_val = data_split

    svm_model = svm.SVC(kernel=kernel, C=C)
    svm_model.fit(x_train, y_train)
    
    # results on training set
    if return_train:
      y_train_pred = svm_model.predict(x_train)
      train_acc = accuracy_score(y_train, y_train_pred)
      train_f1 = f1_score(y_train, y_train_pred, average="macro")
    else:
      train_acc = '-'
      train_f1  = '-'

    # results on validation set
    y_val_pred = svm_model.predict(x_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    acc_results.append(train_acc)
    val_acc_results.append(val_acc)
    f1_results.append(train_f1)
    val_f1_results.append(val_f1)

    if save_classifiers:
      with open(f'{trained_classif_dir_path}/F{fold_idx}.svc', 'wb') as file_handler:
        pickle.dump(svm_model, file_handler)

    print('*', end='')

  return (acc_results, val_acc_results, f1_results, val_f1_results)


# **************************   get_random_string()   **************************
from random import choice
import string

def get_random_string(length):
    """
    Returns random string of fixed length
    
    """

    # choose from all lowercase+uppercase letters
    letters = string.ascii_lowercase + string.ascii_uppercase
    result_str = ''.join(choice(letters) for i in range(length))

    return result_str 

# evaluateSVC()  *********************************************************************************

from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from statistics import mean 

def evaluateSVC(x_train, y_train, x_val, y_val, kernel='rbf', C=1, return_train=False):
  svm_model = svm.SVC(kernel=kernel, C=C)
  svm_model.fit(x_train, y_train)
  
  # results on training set
  if return_train:
    y_train_pred = svm_model.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
  else:
    train_acc = '-'
    train_f1  = '-'

  # results on validation set
  y_val_pred = svm_model.predict(x_val)
  val_acc = accuracy_score(y_val, y_val_pred)
  val_f1 = f1_score(y_val, y_val_pred, average="macro")

  return (train_acc, val_acc, train_f1, val_f1)


# formResultDictionary() ********************************

from statistics import mean
def formResultDictionary(validation_results, tm_params, svm_params, conf_ID, tm_ID):
  acc_results, val_acc_results, f1_results, val_f1_results = validation_results

  k = len(acc_results)
  mean_acc = mean(acc_results)
  mean_val_acc = mean(val_acc_results)
  mean_f1 = mean(f1_results)
  mean_val_f1 = mean(val_f1_results)

  result_dictionary = {'conf_ID':conf_ID,
                       'tm_ID':tm_ID,
                       **tm_params, **svm_params,
                      'mean_acc':mean_acc,
                      'mean_val_acc':mean_val_acc,
                      'mean_f1':mean_f1,
                      'mean_val_f1':mean_val_f1}
  
  return result_dictionary


# ********************************* DATASET UTILS ***************************************
# ***************************************************************************************


def getDataSplits(X_train, Y_train, task, n_folds):
  '''
  splits train data into k folds for cross validation.

  input
  X_train  - pd.DataFrame, X_train.columns() = ['text','kfold']
  Y_train  - pd.DataFrame, Y_train.columns() = ['HS', 'TR', 'AG', 'ATG', 'kfold']
  task     - str, valid tasks = ['HS', 'TR', 'AG', 'ATG']
  n_folds  - int, number of folds for cross validation

  output
  data_splits_list  - list with n_folds different splits of the train data

  '''
  data_splits_list = []

  for K in range (n_folds):
    train_mask  = X_train.kfold==K
    x_train = X_train.loc[train_mask,'text'].to_list()
    y_train = Y_train.loc[train_mask, task].to_list()

    val_mask = Y_train.kfold!=K
    x_val = X_train.loc[val_mask,'text'].to_list()
    y_val = Y_train.loc[val_mask, task].to_list()

    data_splits_list.append( (x_train, y_train, x_val, y_val) )

  return data_splits_list

from os import path, makedirs
# prefit the text models that will be used along the experiments
def getTransformedDataSplits(X_train, Y_train, task, text_model, n_folds=7, partial_CV=True,
                             tm_ID='', save_textmodels=False):
  '''
  '''

  if save_textmodels:
    # Weights are initailly saved in a temp directory
    fitted_tm_dir_path  = f'./text_models/{tm_ID}' 
    if not path.exists(fitted_tm_dir_path):
      makedirs(fitted_tm_dir_path)
  else:
    fitted_tm_dir_path = ''

  data_splits = getDataSplits(X_train, Y_train, task, n_folds)
  transformed_data_splits = []
  
  for split_idx, split in enumerate(data_splits):
    if partial_CV & (split_idx%2==1):
        transformed_data_splits.append(None)
        continue

    x_train, y_train, x_val, y_val = split
    text_model.fit(x_train)
                                
    x_train = text_model.transform(x_train)
    x_val   = text_model.transform(x_val)

    transformed_data_splits.append((x_train, y_train, x_val, y_val))

    print('*', end='')

    if save_textmodels:
      with open(f'{fitted_tm_dir_path}/F{split_idx}.tm', 'wb') as file_handler:
        pickle.dump(text_model, file_handler)


  return transformed_data_splits