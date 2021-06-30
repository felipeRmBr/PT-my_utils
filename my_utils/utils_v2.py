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
  '''
  Maps a (target-aggr) tuple into a single label between 0 and 3 (four classes)

  input
  target_class    - int, [0,1]
  aggr_class      - int, [0,1]

  outpuy
  multidim_class  - int, [0-3]

  Examples:

  (0,0) -> 0
  (0,1) -> 1

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
  Maps a (hate-target-aggr) triplets into a single label between 0 and 4

  input
  hate_class      - int, [0,1] 
  target_class    - int, [0,1]
  aggr_class      - int, [0,1]

  outpuy
  multidim_class  - int, [0-4]

  Examples:

  (0,0,0) -> 0
  (1,0,0) -> 1

  '''

  if hate_class==0:
    return 0
  elif hate_class==1:
    return mapToFourClassesFormat(target_class, aggr_class) + 1


def selectDocsAndLabels(task, train_docs, test_docs, train_labels, test_labels):
  '''
  Returns the labels needed for the chosen task

  input
  task            - str, ['HATE', 'TATGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE']
  train_docs
  test_docs
  train_labels    - list containing the labels for docs in train set (three labels per document)
  test_labels     - list containing the labels for docs in test set (three labels per document)

  outpuy
  X_train         - list of strings, train set for the given task
  X_test          - list of strings, test set for the given task
  Y_train         - list of integers, one entry per document in X_train
  Y_test          - list of integers, one entry per document in X_test

  '''

  X_train = list()
  X_test = list()
  Y_train = list()
  Y_test = list()

  if task == 'HATE':
    X_train = train_docs
    X_test = test_docs
    Y_train = [l[0] for l in train_labels]
    Y_test  = [l[0] for l in test_labels] 

  elif task == 'TARGET':

    for i, label in enumerate(train_labels):
        # if hate_speech = 1
        if label[0] == 1: 
            Y_train.append(label[1])
            X_train.append(train_docs[i])

    for i, label in enumerate(test_labels):
        # if hate_speech = 1
        if label[0] == 1: 
            Y_test.append(label[1])
            X_test.append(test_docs[i])

  elif task == 'AGGR':

    for i, label in enumerate(train_labels):
        # if hate_speech = 1
        if label[0] == 1: 
            Y_train.append(label[2])
            X_train.append(train_docs[i])

    for i, label in enumerate(test_labels):
        # if hate_speech = 1
        if label[0] == 1: 
            Y_test.append(label[2])
            X_test.append(test_docs[i])

  elif task == 'TARGET_AND_AGGR':
    '''
    TR = 0, AG = 0  ->  0
    TR = 0, AG = 1  ->  1
    TR = 1, AG = 0  ->  2
    TR = 1, AG = 1  ->  3
    '''

    for i, label in enumerate(train_labels):
        # if hate_speech = 1
        if label[0] == 1: 
            Y_train.append(mapToFourClassesFormat(*label[1:]))
            X_train.append(train_docs[i])

    for i, label in enumerate(test_labels):
        # if hate_speech = 1
        if label[0] == 1: 
            Y_test.append(mapToFourClassesFormat(*label[1:]))
            X_test.append(test_docs[i])

  elif task == 'ALL_IN_ONE':
    '''
    HT = 0, TR = 0, AG = 0  ->  0
    HT = 1, TR = 0, AG = 0  ->  1
    HT = 1, TR = 0, AG = 1  ->  2
    HT = 1, TR = 1, AG = 0  ->  3
    HT = 1, TR = 1, AG = 1  ->  4
    '''
    X_train = train_docs
    X_test = test_docs

    Y_train = [mapToFiveClassesFormat(*label) for label in train_labels]
    Y_test  = [mapToFiveClassesFormat(*label) for label in test_labels] 

  else:
    print('Invalid task. No labels returned.')
    return (Null,Null,Null,Null)

  return (X_train, X_test, Y_train, Y_test)

# TODO:
# 1) receive paths as arguments of the function
# 2) return just the data that is relevant for the chosen task

def importData(task, return_test = True):
  '''
  Import data as needed for the chosen task

  input
  task          - str, ['HATE', 'TATGET', 'AGGR', 'TARGET_AND_AGGR', 'ALL_IN_ONE']
  return_test   - boolean, weather to return Test set or not

  outpuy
  X_train       - list of training documents
  Y_train       - list of labels for training documents
  X_test        - list of test documents
  Y_test        - list of labels for testing documents

  '''
  
  # check for invalid task_options
  task_options = ['HATE', 
                  'TARGET', 'TARGET_COMPLETE_SET', 
                  'AGGR', 'AGGR_COMPLETE_SET', 
                  'TARGET_AND_AGGR', 'TARGET_AND_AGGR_COMPLETE_SET',
                  'ALL_IN_ONE', ]

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

  # extract tweets and labels
  train_docs = []
  train_labels = []
  
  for entry in train_set:
      train_docs.append(entry['text'])
      train_labels.append([int(entry['hate']), 
                          int(entry['target']), 
                          int(entry['aggressiveness'])
                          ])
  test_docs = []
  test_labels = []

  for entry in dev_set:
      test_docs.append(entry['text'])
      test_labels.append([int(entry['hate']), 
                          int(entry['target']), 
                          int(entry['aggressiveness'])
                          ])

  # take just the docs and labels needed for the given task
  X_train, X_test, Y_train, Y_test = selectDocsAndLabels(task, train_docs, test_docs, train_labels, test_labels)

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

  samples_per_fold = int(train_set_size/k)

  for fold in range(k):
    # validation set limits
    start_idx = samples_per_fold * fold
    stop_idx  = samples_per_fold * (fold+1) 

    x_train = X_train[:start_idx] + X_train[stop_idx:]
    x_val   = X_train[start_idx:stop_idx]

    y_train = Y_train[:start_idx] + Y_train[stop_idx:]
    y_val   = Y_train[start_idx:stop_idx]

    data_splits_list.append( (x_train, y_train, x_val, y_val) )

  return data_splits_list


def getNoHateDocs():
  # if task is valid then go ahead
  import json
  train_set = []
  dev_set = []

  # get data from .json files    
  path_train = './dataset_files/train_es_A.json'

  for line in open(path_train, 'r'):
      train_set.append(json.loads(line))

  # extract tweets and labels
  train_docs = []
  train_labels = []
  
  for entry in train_set:
      train_docs.append(entry['text'])
      train_labels.append([int(entry['hate']), 
                          int(entry['target']), 
                          int(entry['aggressiveness'])
                          ])
  noHateDocs = list()
  for i, label in enumerate(train_labels):
    if label[0] == 0:
        noHateDocs.append(train_docs[i])

  return noHateDocs
    