def compute_metrics(target, predicted):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy_s = accuracy_score(target, predicted)
    precision_macro = precision_score(target, predicted, average="macro")
    precision_pos = precision_score(target, predicted, average="binary", pos_label = '1')
    precision_neg = precision_score(target, predicted, average="binary", pos_label = '0')
    recall_pos = recall_score(target, predicted, average="binary", pos_label = '1')
    recall_neg = recall_score(target, predicted, average="binary", pos_label = '0')
    f1_pos = f1_score(target, predicted, average="binary", pos_label = '1')
    f1_neg = f1_score(target, predicted, average="binary", pos_label = '0')

    results = {'acc':accuracy_s, 
              'prec_pos' : precision_pos,
              'prec_neg' : precision_neg,  
              'recall_pos' : recall_pos,
              'recall_neg' : recall_neg,
              'f1_pos': f1_pos,
              'f1_neg': f1_neg}
    
    return results

def mapToFourClassesFormat(target_class, aggr_class):
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
  if hate_class==0:
    return 0
  elif hate_class==1:
    return mapToFourClassesFormat(target_class, aggr_class) + 1

def getLabels(task, train_labels, test_labels):
  '''
  Returns the labels needed for the chosen task

  input
  task            - str, ['HATE', 'TATGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE']
  train_labels    - list containing the labels for all the documents in the train set 
  test_labels     - list containing the labels for all the documents in te test set

  outpuy
  Y_train         - list of integerr, one entry per document in the train set
  Y_test          - list of integerr, one entry per document in the test set

  '''
  if task == 'HATE':
    Y_train = [l[0] for l in train_labels]
    Y_test  = [l[0] for l in test_labels] 
  elif task == 'TARGET':
    Y_train = [l[1] for l in train_labels]
    Y_test  = [l[1] for l in test_labels] 
  elif task == 'AGGR':
    Y_train = [l[2] for l in train_labels]
    Y_test  = [l[2] for l in test_labels] 
  elif task == 'TARGET_AND_AGGR':
    '''
    TR = 0, AG = 0  ->  0
    TR = 0, AG = 1  ->  1
    TR = 1, AG = 0  ->  2
    TR = 1, AG = 1  ->  3
    '''
    Y_train = [mapToFourClassesFormat(*l[1:]) for l in train_labels]
    Y_test  = [mapToFourClassesFormat(*l[1:]) for l in test_labels] 
  elif task == 'ALL_IN_ONE':
    '''
    HT = 0, TR = 0, AG = 0  ->  0
    HT = 1, TR = 0, AG = 0  ->  1
    HT = 1, TR = 0, AG = 1  ->  2
    HT = 1, TR = 1, AG = 0  ->  3
    HT = 1, TR = 1, AG = 1  ->  4
    '''
    Y_train = [mapToFiveClassesFormat(*l) for l in train_labels]
    Y_test  = [mapToFiveClassesFormat(*l) for l in test_labels] 

  elif task == 'GLOBAL_EVALUATION':
    Y_train = train_labels
    Y_test = test_labels
    
  else:
    print('Invalid task. No labels returned.')
    return ([],[])

  return (Y_train, Y_test)

# TODO:
# 1) receive paths as arguments of the function
# 2) return just the data that is relevant for the chosen task

def importData(task):
  
  # check for invalid task_options
  task_options = ['HATE', 'TARGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE', 'GLOBAL_EVALUATION']

  if task not in task_options:
    print('Invalid task. No data was returned.\n')
    print('...')
    print("Valid tasks: ['HATE', 'TARGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE']")
    return ([], [], [], [])

  # if task is valid then go ahead
  import json
  train_set = []
  dev_set = []

  # get data from .json files    
  path_train = './dataset_files/train_es_A.json'
  path_dev = './dataset_files/dev_es_A.json'

  for line in open(path_train, 'r'):
      train_set.append(json.loads(line))
      
  for line in open(path_dev, 'r'):
      dev_set.append(json.loads(line))

  # estract tweets and labels
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
  Y_train, Y_test = getLabels(task, train_labels=train_labels, test_labels=test_labels)

  return (X_train, Y_train, X_test, Y_test)

# Split train_set into train and validation subsets
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
