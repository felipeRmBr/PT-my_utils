import tensorflow as tf
import tensorflow.keras.optimizers as keras_optimizers

optimizers_list = {'adam-1e-3':keras_optimizers.Adam(learning_rate=0.001),
              'adam-7e-4':keras_optimizers.Adam(learning_rate=0.0007),
              'adam-5e-4':keras_optimizers.Adam(learning_rate=0.0005),
              'adam-3e-4':keras_optimizers.Adam(learning_rate=0.0003),
              'adam-1e-4':keras_optimizers.Adam(learning_rate=0.0001),
              'rmsprop-1e-3':keras_optimizers.RMSprop(learning_rate=0.001, momentum=0.0),
              'rmsprop-7e-4':keras_optimizers.RMSprop(learning_rate=0.0007, momentum=0.0),
              'rmsprop-5e-4':keras_optimizers.RMSprop(learning_rate=0.0005, momentum=0.0),
              'rmsprop-3e-4':keras_optimizers.RMSprop(learning_rate=0.0003, momentum=0.0),
              'rmsprop-1e-4':keras_optimizers.RMSprop(learning_rate=0.0001, momentum=0.0),
              'rmsprop-7.5e-5':keras_optimizers.RMSprop(learning_rate=0.000075, momentum=0.0),
              'rmsprop-5e-5':keras_optimizers.RMSprop(learning_rate=0.00005, momentum=0.0),
              'rmsprop-1e-3-mu0.9':keras_optimizers.RMSprop(learning_rate=0.001, momentum=0.9),
              'rmsprop-7e-4-mu0.9':keras_optimizers.RMSprop(learning_rate=0.0007, momentum=0.9),
              'rmsprop-5e-4-mu0.9':keras_optimizers.RMSprop(learning_rate=0.0005, momentum=0.9),
              'rmsprop-3e-4-mu0.9':keras_optimizers.RMSprop(learning_rate=0.0003, momentum=0.9),
              'rmsprop-1e-4-mu0.9':keras_optimizers.RMSprop(learning_rate=0.0001, momentum=0.9),
              'rmsprop-7.5e-5-mu0.9':keras_optimizers.RMSprop(learning_rate=0.000075, momentum=0.9),
              'rmsprop-5e-5-mu0.9':keras_optimizers.RMSprop(learning_rate=0.00005, momentum=0.9)} 


# **************************   spaceScanner()   ************************** 

# adapted from Talos solution
# (check: https://autonomio.github.io/talos/#/)
import pickle
from itertools import product
from random import sample
from pandas import DataFrame
from numpy import random as np_random

def spaceScanner(X_train, Y_train, task, model_prototype, search_space, arch_params_keys, 
                training_params_keys, fraction2eval, fitting_attemps, n_folds, stop_threshold=0.99, 
                partial_CV=False, backup_file='testing_file.df', backup_freq=25, 
                save_history_files=False, save_models_as_json=False, save_weights=False):
  
  """
  Evaluates a sample of the configurations defined in the search space.

  inputs:
  X_train               - list[numpy_array], encoded folds with the train data
  Y_train               - DataFrame, labels (Y_train.columns = ['HS','TR','AG','HTA','kfold'])
  task                  - str, valid_task_options = ['HS','TR','AG','HTA']
  model_prototype       - function, model prototype definition (returns compiled keras model)
  search_space          - list[tuple], combinations of parameters to be evaluated 
                          (Every entry in the search space is a two elements tuple, 
                          the first of this elements is a list of architecture_params and the 
                          second one is a list of training_params)
  arch_params_keys      - list[str], arch_params_keys
  training_params_keys  - list[str], training_params_keys
  evaluation_mode       - int, eithe 1 or 2. See footnote for a description about how each one of
                          this works
  fitting_attemps       - int, try fitting the model N times and keep the best result (N=fitting_attemps)
  n_folds               - int, number of folds for cross validation
  partial_CV            - boolean, partial_CV=True means that the model should only be evaluated 
                          in half of the n_folds
  backup_file           - str, name of the file where the experiment results will ve saved
                          (The filename should be passed without extension. A .df extension will be appended)
  backup_freq           - int, how often will the experiment results be backed-up
  ---------


  outputs:
  results_df            - DataFrame, one row for every configuration evaluated

  """

  if save_weights:
      # prepare the destiny directory 
      temp_files_dir = './temp_files' 
      if not os.path.exists(temp_files_dir):
        os.makedirs(temp_files_dir)

  print(f'BackupFile: {backup_file}')
  print('SCANNING SEARCH SPACE\n')

  # take a sample of the search_space
  if fraction2eval < 1:
    sample_size = int(len(search_space)*fraction2eval)
    print(f'Aproximmately {sample_size} configurations will be evaluated.')
    # space_sample=sample(search_space, sample_size)
  else: 
    sample_size = len(search_space)
    print(f'{sample_size} configurations will be evaluated.')
    # space_sample=search_space
    

  """
  When using space samples
  transform the parameters-tuples into parameters-dictionaries
  space_sample = [[dict( zip(arch_params_keys, params_combo[0]) ),
                   dict( zip(training_params_keys, params_combo[1]) )] for params_combo in space_sample]
  """

  # when using random variable
  search_space = [[dict( zip(arch_params_keys, params_combo[0]) ),
                   dict( zip(training_params_keys, params_combo[1]) )] for params_combo in search_space]

  # validate the configurations in the search_space_sample
  results_list = []
  validations_count = 0

  for config_idx, conf_params in enumerate(search_space):
    # use a binomial random variable to decide weatther to test the 
    # current configuration or pass. The parameter p of the binomial 
    # experiment is iqual to fraction2eval
    random_v = np_random.binomial(1,fraction2eval)
    if random_v == 0:
      continue

    validations_count += 1
    # get a ramdom conf_ID
    conf_ID = get_random_string(8)

    # print a configurations count
    counter_str = f'{validations_count}'.ljust(len(f'{sample_size}')+1) 
    print(f'{counter_str}- ', end = '')
    print(f'conf_ID: {conf_ID}   ', end = '')

    # evaluate the current config using croos validation
    arch_params, training_params = conf_params

    validation_results = modelCrossValidation(X_train, Y_train,
                                             task,  
                                             model_prototype,
                                             arch_params,
                                             training_params,
                                             n_folds,
                                             fitting_attemps,
                                             partial_CV,
                                             conf_ID,
                                             save_history_files,
                                             save_models_as_json,
                                             save_weights)

    # update the results list
    results_list.append(validation_results)

    # display progress and results
    train_acc_A = validation_results['train_acc_A']
    val_acc_A   = validation_results['val_acc_A']
    train_acc_B = validation_results['train_acc_B']
    val_acc_B   = validation_results['val_acc_B']

    #progress_string    = '[aprox. {}% complete]'.format(round(100*(config_count+1)/sample_size,2)).ljust(20, ' ')

    train_acc_A_str = 'train_acc_1 = {},'.format(round(train_acc_A,3)).ljust(20,' ')
    val_acc_A_str   = 'val_acc_1 = {}'.format(round(val_acc_A,3)).ljust(20,' ')
    train_acc_B_str = 'train_acc_2 = {},'.format(round(train_acc_B,3)).ljust(20,' ')
    val_acc_B_str   = 'val_acc_2 = {}'.format(round(val_acc_B,3)).ljust(20,' ')

    print('--  {}  {} --  {}  {}'.format(train_acc_A_str, 
                                val_acc_A_str,
                                train_acc_B_str,
                                val_acc_B_str))

    # save results every X configurations
    if len(results_list)%backup_freq == 0:
      with open(backup_file, 'wb') as file_handler:
        pickle.dump(DataFrame(results_list), file_handler)

  # final backup
  results_df = DataFrame(results_list)
  with open(backup_file, 'wb') as file_handler:
    pickle.dump(results_df, file_handler)

  print(f'\nPROCCESS COMPLETE. {len(results_list)} CONFIGURATIONS WERE SUCCESFULLY EVALUATED.')

  return results_df

# **************************   modelCrossValidation()   ************************** 
import os

def modelCrossValidation(X_train,Y_train,task,model_prototype,arch_params,training_params,
                         n_folds,fitting_attemps,partial_CV,configuration_ID,save_history_files,
                         save_model_as_json, save_weights):

    """
    Evaluates a model configuration using cross validation. 

    inputs:
    X_train               - list[numpy_array], encoded folds with the train data
    Y_train               - DataFrame, labels (Y_train.columns = ['HS','TR','AG','HTA','kfold'])
    task                  - str, valid_task_options = ['HS','TR','AG','HTA']
    model_prototype       - function obj, model prototype definition (returns compiled keras model)
    arch_params           - dict, arch_params_keys
    training_params       - dict, training_params_keys
    n_folds               - int, number of folds for cross validation
    fitting_attemps       - int, the model should be fitted N=fitting_attemps times for every fold
    partial_CV            - boolean, partial_CV=True means that the model should only be evaluated 
                            in half of the folds
    configuration_ID      - str, 8 char ID
    save_history_files    - boolean, should history_files be backed up or not
    save_model_as_json    - boolean

    
    outputs:
    evaluation_results_dict   - dict, summarizes the results of the cross validations             

    """

    if save_weights:
      # Weights are initailly saved in a temp directory
      temp_dir_path  = f'./temp_files/{configuration_ID}' 
      if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)
    else:
      temp_dir_path = ''

    # unpack the training parameters
    optimizer_id = training_params['optimizer']
    max_epochs  = training_params['max_epochs']
    batch_size = training_params['batch_size']

    # instantiate the corresponding optimizer
    optimizer = optimizers_list[optimizer_id]

    # save a json representation of the model architecture
    if save_model_as_json:
      model = model_prototype(task, arch_params, optimizer)
      model_json = model.to_json()
      with open(f"./models_json_files/{configuration_ID}.json", "w") as json_file:
        json_file.write(model_json)

    # The main work ------------------------------------------------------------
    best_attemps_record = {}    # keeps track of the best attemps for every fold
    histories_list = list()   # list[dict]

    for fold_idx, data_fold in enumerate(getDataSplitsGenerator(X_train, Y_train, n_folds)):
      
      # When partial_CV==True just evaluate the model in half of the n-folds
      if (partial_CV) and (fold_idx%2==1):
        continue

      # Run N attemps to fit the model (N=fitting_attemps)
      # save the model_history of the model with the best val_acc metric
      best_attemp_idx = 0
      best_attemp_history = None
      max_acc = 0
      for attemp_idx in range(fitting_attemps):

        model = model_prototype(task, arch_params, optimizer)

        weights_dir = f'{temp_dir_path}/F{fold_idx}/A{attemp_idx}'
        acc, model_history = fitNeuralNetworkModel(model, 
                                                   task, 
                                                   (data_fold), 
                                                   batch_size, 
                                                   max_epochs,
                                                   save_weights, 
                                                   weights_dir)

        if acc > max_acc: 
          # this is the best attemp so far
          # update the max_acc, the best_attemp_idx and the best_attemp_history
          max_acc = acc
          best_attemp_idx = attemp_idx
          best_attemp_history = model_history

      # update the best_attemps_record 
      best_attemps_record[fold_idx] = best_attemp_idx

      # append the best_attemp_history to the list of histories
      histories_list.append(best_attemp_history)
      
      # print a hint at the end of every fold validation
      print('*', end='')

    # compute a summary of the metrics
    global_best_epoch, best_epochs_list, metrics_summary_dict = getMetricsSummary(histories_list)

    # form a dictionary to report the validation results
    validation_results_dict = {'conf_ID':configuration_ID}
    validation_results_dict.update(arch_params)
    validation_results_dict.update(training_params)
    validation_results_dict.update(metrics_summary_dict)

    # save the list of histories for future reference
    if save_history_files:
        with open(f'./history_files/{configuration_ID}.dict', 'wb') as file_handler:
            pickle.dump(histories_list, file_handler)

    if save_weights:
      # up to this point we have saved a lot of weights 
      # but we actually just need a couple for each fold. 
      # therefore, we need to "pick the relevant files"
      pickTheRelevantFiles(configuration_ID, 
                           best_attemps_record, 
                           global_best_epoch, 
                           best_epochs_list, 
                           max_epochs,
                           fitting_attemps)
  
    print('  ', end='')

    return validation_results_dict



# **************************   fitNeuralNetworkModel()   **************************

def fitNeuralNetworkModel(model, task, data, batch_size, max_epochs, 
                          save_weights=False, weights_dir='', verbose=0):
  """
  Fits a precompiled keras model to the given data. 

  inputs:
  model           - compiled keras model
  task            - str, ['HS','TR', 'AG', 'HTA']
  data            - tuple, (x_train, y_train, x_val, y_val)
  batch_size      - int 
  max_epochs      - int
  save_weights    - boolean
  weights_dir     - str, filepath to the weights directory
  
  --
  x_train         - numpy_array [shape = N_t, ENCODDING_DIM]
  y_train         - numpy_array [shape = N_t, 1]
  x_val           - numpy_array [shape = N_v, ENCODDING_DIM]
  y_val           - numpy_array [shape = N_v, 1]

  outputs:
  max_acc         - float, max_val_acc registered in model_history (as defined in method A)
  model_history   - dict, model.history.history

  """

  # Callbacks
  if save_weights:
    if not os.path.exists(weights_dir):
      os.makedirs(weights_dir)

    # instantiate the custome saver callback
    custom_saver_callback = CustomSaver(weights_dir=weights_dir)
    callbacks_list = [custom_saver_callback]
      
  else:
    callbacks_list = None
    
  # Main work
  x_train, y_train, x_val, y_val = data 

  # Extract the labels that correspond to the given task
  # type(y_train) = type(y_val) = pandas.DataFrame
  y_train = y_train[task]
  y_val   = y_val[task]

  # For multiclass classification we need to transform 
  # the labels to a one-hot-encoding representation
  if task == 'HTA':
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=5)     

  model.fit(x=x_train, 
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks_list)
  
  max_acc = max(model.history.history['val_acc'])

  return max_acc, model.history.history

# **************************   CustomSaver()   **************************

import keras
class CustomSaver(keras.callbacks.Callback):
  """
  Saves the model weights at every epoch

  """
  
  def __init__(self, weights_dir):
    super(CustomSaver, self).__init__()
    self.weights_dir=weights_dir

  def on_epoch_end(self, epoch, logs=None):
    filename = f'e{epoch}.hdf5'
    filepath = f'{self.weights_dir}/{filename}'
    self.model.save(filepath,overwrite=True,include_optimizer=False)


# **************************   getMetricsSummary()   ************************** 
from statistics import mean
from statistics import median

def getMetricsSummary(hist_list):
    """
    Returns a summary of the metrics registered in the histories_list.

    inputs:
    hist_list         - list[model.history.history], list of models histories 

    output:
    global_best_epoch - int
    best_epochs_list  - list[int]
    metrics_summary   - dict, keys = [train_acc_A, val_acc_A, train_acc_B, val_acc_B] 

    ** Two methosd to evaluate every configuration: **

    Method A:
        1) Compute the mean-val-acc at each epoch (the mean with 
           respect to the different data-folds)
        2) Let E be the epoch with the hihgest mean-val-acc 
        3) Report both the mean-val-acc and mean-train-acc at epoch E

    Method B:
        1) Let E(k) be the epoch with the highest val-acc recorded for fold k
        1) Let val_acc(E(k)) and train_acc(E(k)) be the metrics regitered for fold k at the epoch E(K)
        2) Report the mean values of both val_acc(k) and acc(k)

    """   

    global_best_epoch, train_acc_A, val_acc_A = method_A_Metrics(hist_list)
    best_epochs_list, min2best, max2best, median2best, train_acc_B, val_acc_B = method_B_Metrics(hist_list)

    summary_dict = {'best_epochh':global_best_epoch,
                    'train_acc_A':train_acc_A, 
                    'val_acc_A':val_acc_A,
                    'min2best':min2best,
                    'max2best':max2best,
                    'median2best':median2best,
                    'train_acc_B':train_acc_B,
                    'val_acc_B':val_acc_B
                    }

    return global_best_epoch, best_epochs_list, summary_dict


# **************************   method_A_Metrics()   **************************

def method_A_Metrics(histories_list):
    # Method A results:
    mean_val_acc  = list()     
    mean_train_acc = list() 

    # Find M (the largest mean-val-acc)
    max_mean_val_acc = 0
    acc_zip = list(zip(*[h['acc'] for h in histories_list]))
    val_acc_zip = list(zip(*[h['val_acc'] for h in histories_list]))

    n_epochs = len(acc_zip)
    best_epoch_idx = 0
    for i in range(n_epochs):
        mean_train_acc.append(mean(acc_zip[i]))
        mean_val_acc.append(mean(val_acc_zip[i]))

        if mean_val_acc[-1] > max_mean_val_acc:
            max_mean_val_acc = mean_val_acc[-1]
            best_epoch_idx = i

    """
    DELTA = 0  
    DELTA = .5/100

    if DELTA > 0:
        # Try to find an epoch with a smaller stdev
        # Use M to pick the best epoch
        best_epoch_val_stdev = 1
        best_epoch_idx = 0

        for i in range(n_epochs):
            val_acc = mean_val_acc[i]
            val_stdev = val_acc_stdev[i]

            if (val_acc >= max_val_acc - DELTA) & (val_stdev < best_epoch_val_stdev):
                best_epoch_val_stdev = val_stdev
                best_epoch_idx = i
    """

    return best_epoch_idx, mean_train_acc[best_epoch_idx], mean_val_acc[best_epoch_idx]

def method_B_Metrics(histories_list):
    # Method B results:
    val_acc_results  = []
    train_acc_results = []

    best_epochs_list = []
    for h in histories_list:
        val_acc_history = h['val_acc']
        train_acc_history = h['acc']

        fold_max_val_acc = max(val_acc_history)
        fold_best_epoch = val_acc_history.index(fold_max_val_acc)
        best_epochs_list.append(fold_best_epoch)

        val_acc_results.append(fold_max_val_acc)
        train_acc_results.append(train_acc_history[fold_best_epoch])

    min2best = min(best_epochs_list)
    max2best = max(best_epochs_list)
    median2best = median(best_epochs_list)

    return best_epochs_list, min2best, max2best, median2best, mean(train_acc_results), mean(val_acc_results)



# **************************   pickTheRelevantFiles()   **************************
from shutil import copyfile
from shutil import rmtree

def pickTheRelevantFiles(configuration_ID,best_attemps_dict,global_best_epoch, 
                         best_epochs_list,max_epochs,fitting_attemps):

  """
  Picks the relevant files out of all the ones saved during a model's
  cross validation. 

  """
  
  # prepare the permanent directory 
  final_dir_path = f'./trained_models/{configuration_ID}' 
  if not os.path.exists(final_dir_path):
    os.makedirs(final_dir_path)

  temp_dir_path = f'./temp_files/{configuration_ID}'

  for i, best_attemp_record in enumerate(best_attemps_dict.items()):

    fold_idx, best_attemp_idx = best_attemp_record
    fold_best_eopch = best_epochs_list[i]

    # We want to keep the weights at two epochs: 
    #  1) the best one according to evaluation method 1 (global_best_epoch)
    #  2) the best one according to evaluation method 2 (fold_best_eopch)

    src_filepath_1  = f'{temp_dir_path}/F{fold_idx}/A{best_attemp_idx}/e{global_best_epoch}.hdf5'
    src_filepath_2  = f'{temp_dir_path}/F{fold_idx}/A{best_attemp_idx}/e{fold_best_eopch}.hdf5'
    dest_filepath_1 = f'{final_dir_path}/F{fold_idx}_A.hdf5'
    dest_filepath_2 = f'{final_dir_path}/F{fold_idx}_B.hdf5'

    copyfile(src_filepath_1, dest_filepath_1)
    copyfile(src_filepath_2, dest_filepath_2)

    # delete all the files that we are not gonna need
    #rmtree(f'{configuration_dir_path}/F{fold_idx}')


# **************************   getDataSplitsGenerator()   **************************
from numpy import concatenate as np_concatenate

def getDataSplitsGenerator(X_train, Y_train, n_folds):
  '''
  Splits train data into K folds for cross validation. (K=n_folds)

  inputs:
  X_train   - list[numpy_array], encoded folds
  Y_train   - DataFrame, containing the labels for the training instances 
  n_folds   - int, number of folds for cross validation

  -------
  inputs preconditions:
  len(X_train) == n_folds

  outputs:
  (x_train, y_train, x_val, y_val) - generator with K different splits of the train data

  '''

  for k in range(n_folds):
  
    # DATA
    x_train = np_concatenate([X_train[i] for i in range(n_folds) if i not in [k]], axis=0)
    x_val   = X_train[k]

    # LABELS
    train_mask = Y_train['kfold'] != k
    val_mask = Y_train['kfold'] == k

    y_train = Y_train.loc[train_mask, :]
    y_val   = Y_train.loc[val_mask, :]

    yield (x_train, y_train, x_val, y_val)


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

