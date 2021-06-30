# TODO:
# 1) receive paths as arguments of the function
# 2) return just the data that is relevant for the chosen task

def importData(task):

  '''
  Inports dataset from json files. Returns labels acording to the task option

  input:
  *train_set   - string, path to train set (include in refactoring)
  *test_set    - string, path to test set (include in refactoring)  
  task         - string, task option 

  output:
  (X_train, Y_train, X_test, Y_test)

  X_train and X_test are lists of strings containing the text of the documents

  Y_train and Y_test are lists of labels (innts/tuples) representing the labels 
  for the returned docs
  
  '''
  
  # check for invalid task options
  task_options = ['HATE', 'TARGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE', 'GLOBAL_EVALUATION']

  if task not in task_options:
    print('Invalid task. No data was returned.\n')
    print('...')
    print("Valid tasks: ['HATE', 'TARGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE', 'GLOBAL_EVALUATION']")
    return ([], [], [], [])

  # if task is valid then go ahead
  import json
  train_set = []
  dev_set = []

  # get data from .json files    
  path_train = './dataset_files/JSON/train_es_A.json'
  path_dev = './dataset_files/JSON/dev_es_A.json'

  for line in open(path_train, 'r'):
      train_set.append(json.loads(line))
      
  for line in open(path_dev, 'r'):
      dev_set.append(json.loads(line))

  # extract tweets and labels
  X_train = []
  train_labels = []
  
  for entry in train_set:
      X_train.append(entry['text'])
      train_labels.append([int(entry['hate']), 
                          int(entry['target']), 
                          int(entry['aggressiveness'])
                          ])
  X_test = []
  test_labels = []

  for entry in dev_set:
      X_test.append(entry['text'])
      test_labels.append([int(entry['hate']), 
                          int(entry['target']), 
                          int(entry['aggressiveness'])
                          ])

  # get the labels needed for the chosen task
  Y_train = getLabels(task, train_labels)
  Y_test  = getLabels(task, test_labels)

  return (X_train, Y_train, X_test, Y_test)


import pickle
import numpy as np

def loadEncodedTrainData(embedding_type, encoding_format, labels_to_return):
  """ 
  Loads encoded dataset from drive.
  
  Input:
  embedding_type   - str, valid_types = ['FT1', 'FT2', 'FT3', 'W2V100', 'W2V300', 'GloVe100', 'GloVe300']
  encoding_format  - str, valid_foramts = ['SINGLE-VEC', 'EMB-SEQ']

  Output:
  (X_train, Y_train) -- tuple containing the encoded dataset(s)

  --
  X_train   - list[numpy_array], encoded train-set partitioned in K subsets
  Y_train   - DataFrame, train set labels
  X_test    - list[numpy_array], encoded test-set
  Y_tes     - DataFrame, test set labels

  """
  valid_embedding_types = ['FT1', 'FT2', 'FT3', 'W2V100', 'W2V300', 'GloVe100', 'GloVe300']
  valid_encoding_format_options = ['SINGLE-VEC', 'EMB-SEQ']

  if embedding_type not in valid_embedding_types:
    print('Invalid embedding_type option. No data was returned.\n')
    print('...')
    print("Valid embedding types: ['FT1', 'FT2', 'FT3', 'W2V100', 'W2V300', 'GloVe100', 'GloVe300']")
    return (None, None)

  if encoding_format not in valid_encoding_format_options:
    print('Invalid format option. No data was returned.\n')
    print('...')
    print("Valid formats: ['SINGLE-VEC', 'EMB-SEQ']")
    return (None, None)

  EMBEDDINGS_INFO = {'FT1':'FastText 1 - Common Crawl + Wikipedia',
                    'FT2':'FastText 2 - Esp. Wikipedia',
                    'FT3':'FastText 3 - Spanish Unannotated Corpora',
                    'W2V300':'W2V 300d - Spanish Unannotated Corpora',
                    'W2V100':'W2V 100d - Spanish CoNLL',
                    'GloVe300':'GloVe 300d - Spanish Billion Word Corpus',
                    'GloVe100':'GloVe 100d - Spanish Billion Word Corpus'}

  embedding_info = EMBEDDINGS_INFO[embedding_type]

  print(embedding_info)
  print('Encoding Format: {}'.format(encoding_format))

  import pandas as pd

  n_folds = 7
  prep_format = 2

  # count the train instances
  total_train_instances = 0
    
  # ENCODED FOLDS
  encoded_train_folds = []
  for K in range(n_folds):
    file_name = '{}_FOLD-{}_P{}.data'.format(embedding_type, K, prep_format)
    with open('./dataset_files/Encoded/{}/{}'.format(encoding_format,file_name), 'rb') as filehandle:
      encoded_train_folds.append(pickle.load(filehandle))
      
      # update the train instances count
      total_train_instances += len(encoded_train_folds[-1])

  # LABELS
  train_dataset_df = pd.read_pickle('./dataset_files/preprocessed_train_dataset.data', None)
  train_labels = train_dataset_df.loc[:,labels_to_return + ['kfold']]

  print('\nProcess complete')
  print('{} train instances retrieved'.format(total_train_instances))

  # Check encodings dimensions
  encodings_dim = encoded_train_folds[0].shape

  print('\nencodings_dim = {}'.format(encodings_dim[1:]))

  return(encoded_train_folds, train_labels)


def loadEncodedTestData(embedding_type, encoding_format):
  """ 
  Loads encoded test dataset from drive.
  
  Input:
  embedding_type   - str, valid_types = ['FT1', 'FT2', 'FT3', 'W2V100', 'W2V300', 'GloVe100', 'GloVe300']
  encoding_format  - str, valid_foramts = ['SINGLE-VEC', 'EMB-SEQ']

  Output:
  (X_test, Y_test) -- tuple containing the encoded dataset

  --
  X_train   - list[numpy_array], encoded train-set partitioned in K subsets
  Y_train   - DataFrame, train set labels
  X_test    - list[numpy_array], encoded test-set
  Y_tes     - DataFrame, test set labels

  """
  valid_embedding_types = ['FT1', 'FT2', 'FT3', 'W2V100', 'W2V300', 'GloVe100', 'GloVe300']
  valid_encoding_format_options = ['SINGLE-VEC', 'EMB-SEQ']

  if embedding_type not in valid_embedding_types:
    print('Invalid embedding_type option. No data was returned.\n')
    print('...')
    print("Valid embedding types: ['FT1', 'FT2', 'FT3', 'W2V100', 'W2V300', 'GloVe100', 'GloVe300']")
    return (None, None)

  if encoding_format not in valid_encoding_format_options:
    print('Invalid format option. No data was returned.\n')
    print('...')
    print("Valid formats: ['SINGLE-VEC', 'EMB-SEQ']")
    return (None, None)

  EMBEDDINGS_INFO = {'FT1':'FastText 1 - Common Crawl + Wikipedia',
                    'FT2':'FastText 2 - Esp. Wikipedia',
                    'FT3':'FastText 3 - Spanish Unannotated Corpora',
                    'W2V300':'W2V 300d - Spanish Unannotated Corpora',
                    'W2V100':'W2V 100d - Spanish CoNLL',
                    'GloVe300':'GloVe 300d - Spanish Billion Word Corpus',
                    'GloVe100':'GloVe 100d - Spanish Billion Word Corpus'}

  embedding_info = EMBEDDINGS_INFO[embedding_type]

  print(embedding_info)
  print('Encoding Format: {}'.format(encoding_format))

  import pandas as pd
  prep_format = 2

  # DATA
  file_name = '{}_TEST_P{}.data'.format(embedding_type, prep_format)
  with open('./dataset_files/Encoded/{}/{}'.format(encoding_format,file_name), 'rb') as filehandle:
    encoded_test_data = pickle.load(filehandle)

  # LABELS
  test_dataset_df = pd.read_pickle('./dataset_files/preprocessed_test_dataset.data', None)
  test_labels = test_dataset_df.loc[:,labels_to_return + ['kfold']]

  print('\nProcess complete')
  print('{} test instances retrieved'.format(len(encoded_test_data)))

  # Check encodings dimensions
  encodings_dim = encoded_test_data.shape

  print('\nencodings_dim = {}'.format(encodings_dim[1:]))

  return(encoded_test_data, test_labels)
   

def getLabels(task, labels_list):
  '''
  Returns the labels needed for the chosen task

  input
  task              - str, ['HATE', 'TATGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE', 'GLOBAL_EVALUATION']
  labels_list       - list of labels in 3_dims_format (HT, TR, AG)

  outpuy
  labels_to_return  - list of labels in the format needed for the chosen task


  '''

  valid_tasks = ['HATE', 'TARGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE', 'GLOBAL_EVALUATION']
  if task not in valid_tasks:
    print('Invalid task. No labels returned.')
    return None

  labels_to_return = list()
  if task == 'HATE':
    labels_to_return = [label[0] for label in labels_list]

  elif task == 'TARGET':
    labels_to_return = [label[1] for label in labels_list]

  elif task == 'AGGR':
    labels_to_return = [label[2] for label in labels_list]

  elif task == 'TARGET_AND_AGGR':
    '''
    TR = 0, AG = 0  ->  0
    TR = 0, AG = 1  ->  1
    TR = 1, AG = 0  ->  2
    TR = 1, AG = 1  ->  3
    '''

    labels_to_return = [mapToFourClassesFormat(*label[1:]) for label in labels_list]

  elif task == 'ALL_IN_ONE':
    '''
    HT = 0, TR = 0, AG = 0  ->  0
    HT = 1, TR = 0, AG = 0  ->  1
    HT = 1, TR = 0, AG = 1  ->  2
    HT = 1, TR = 1, AG = 0  ->  3
    HT = 1, TR = 1, AG = 1  ->  4
    '''
    labels_to_return = [mapToFiveClassesFormat(*label) for label in labels_list]

  elif task == 'GLOBAL_EVALUATION':
    labels_to_return = labels_list

  return labels_to_return

def mapToFourClassesFormat(target_class, aggr_class):
  '''
  Maps labels for tasks B1 and B2 into four_classes_format 

    (0,0)  [TR = 0, AG = 0]  -> 0
    (0,1)  [TR = 0, AG = 1]  -> 1
    (1,0)  [TR = 1, AG = 0]  -> 2
    (1,1)  [TR = 1, AG = 1]  -> 3

  input:
  (TR,AG)   - tuple, labels for tasks B1 an B2

  output
  label     - int, label in four_classes_format

  '''
  if target_class == 0:
    if aggr_class == 0:
      return 0
    elif aggr_class == 1:
      return 1
  elif target_class == 1:
    if aggr_class == 0:
      return 2
    elif aggr_class == 1:
      return 3
      
def mapToFiveClassesFormat(hate_class, target_class, aggr_class):
  '''
  Maps labels for tasks A, B1 and B2 into label in five_classes_format

    [HT = 0, TR = 0, AG = 0]  ->  0
    [HT = 1, TR = 0, AG = 0]  ->  1
    [HT = 1, TR = 0, AG = 1]  ->  2
    [HT = 1, TR = 1, AG = 0]  ->  3
    [HT = 1, TR = 1, AG = 1]  ->  4

  input:
  (HT,TR,AG)   - tuple, labels for tasks A, B1 an B2

  output
  label        - int, label in five_classes_format
  
  '''

  if hate_class==0:
    return 0
  elif hate_class==1:
    return mapToFourClassesFormat(target_class, aggr_class) + 1

def mapToTargetAndAggrLabels(list_of_labels):
  labels_to_return = [mapTo2DimsFormat(label) for label in list_of_labels]
  target_labels = [label[0] for label in labels_to_return]
  aggr_labels    = [label[1] for label in labels_to_return]

  return target_labels, aggr_labels

def mapTo2DimsFormat(label):
  '''
  Maps label in five_classes_format to 3 dims labeling.

    0 -> (0,0)  [HT = 1, TR = 0, AG = 0]
    1 -> (0,1)  [HT = 1, TR = 0, AG = 1]
    2 -> (1,0)  [HT = 1, TR = 1, AG = 0]
    3 -> (1,1)  [HT = 1, TR = 1, AG = 1]

  inpunt:
  label    - int, label in four_classes_format

  output:
  (TR,AG)    - ints tuple, labeling in 2 dims format for tasks B1 and B2

  '''
  if label == 0:
    return(0,0,0)

  elif label == 1:
    return(1,0,0)

  elif label == 2:
    return(1,0,1)

  elif label == 3:
    return(1,1,0)

  elif label == 4:
    return(1,1,1)

def mapTo3DimsFormat(label):
  '''
  Maps label in five_classes_format to 3 dims labeling.

    0 -> (0,0,0)  [HT = 0, TR = 0, AG = 0]
    1 -> (1,0,0)  [HT = 1, TR = 0, AG = 0]
    2 -> (1,0,1)  [HT = 1, TR = 0, AG = 1]
    3 -> (1,1,0)  [HT = 1, TR = 1, AG = 0]
    4 -> (1,1,1)  [HT = 1, TR = 1, AG = 1]

  inpunt:
  label    - int, label in five_classes_format

  output:
  (H,T,A)  - ints tuple, labeling in 3 dims format

  '''
  if label == 0:
    return(0,0,0)

  elif label == 1:
    return(1,0,0)

  elif label == 2:
    return(1,0,1)

  elif label == 3:
    return(1,1,0)

  elif label == 4:
    return(1,1,1)


def getDataSplits(X_train, Y_train, k):
  '''
  splits train data into k folds for cross validation.

  input
  k        - int, number of esplits
  X_train  - list on documents
  Y_train  - list of target labels

  output
  data_splits_list  - list with k different splits of the train data

  '''
  data_splits_list = []
  train_set_size = len(X_train)

  for fold in range(k):
    samples_per_fold = int(train_set_size/k)

    # validation set limits
    start_idx = samples_per_fold * fold
    stop_idx  = samples_per_fold * (fold+1) 

    x_train = X_train[:start_idx] + X_train[stop_idx:]
    x_val   = X_train[start_idx:stop_idx]

    y_train = Y_train[:start_idx] + Y_train[stop_idx:]
    y_val   = Y_train[start_idx:stop_idx]

    data_splits_list.append( (x_train, y_train, x_val, y_val) )

  return data_splits_list


# prefit the text models that will be used along the experiments
def getTransformedDataSplits(X_train, Y_train, task, text_model, k=5):
  '''
  '''
  data_splits = getDataSplits(X_train, Y_train, k)

  # x_train_complement is used when the classifiers for tasks B1, B2 and B-12
  # are trained just over the hate messages. Using x_train_complement we can 
  # (probably) achieve a more accurate TEXT-MODEL 
  if task in ['TARGET', 'AGGR', 'TARGET_AND_AGGR']:
      print('Retriving x_train_complement')
      x_train_complement = getNoHateDocs()
  else:
      x_train_complement = []
  
  transformed_data_splits = []
  for split in data_splits:
    x_train, y_train, x_val, y_val = split

    extended_x_train = x_train + x_train_complement
    #print(len(x_train_complement),len(extended_x_train), len(x_train))

    text_model.fit(extended_x_train)
                                
    x_train = text_model.transform(x_train)
    x_val   = text_model.transform(x_val)

    transformed_data_splits.append((x_train, y_train, x_val, y_val))

  return transformed_data_splits

def transformData(x_train, x_val, text_model):
  '''
  Fits the text_model to train data. Then transforms both sets of docs
  using the fitted text_model

  input:
  x_train   - list of str, train documents
  x_val     - list of str, val documents

  output:
  transformed_x_train    - csr_matrix (Compressed Sparse Row matrix) 
  transformed_x_val      - csr_matrix (Compressed Sparse Row matrix) 

  '''
  text_model.fit(x_train)                      
  transformed_x_train = text_model.transform(x_train)
  transformed_x_val   = text_model.transform(x_val)

  return transformed_x_train, transformed_x_val