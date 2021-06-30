from sklearn.metrics import accuracy_score, f1_score

def evaluatePredictions(val_labels, pred_labels):
  val_hate_labels, val_target_labels, val_aggr_labels    = val_labels
  pred_hate_labels, pred_target_labels, pred_aggr_labels = pred_labels

  A_acc = accuracy_score(val_hate_labels, pred_hate_labels)
  B1_acc = accuracy_score(val_target_labels, pred_target_labels)
  B2_acc = accuracy_score(val_aggr_labels, pred_aggr_labels)

  A_f1 = f1_score(val_hate_labels, pred_hate_labels, average="macro")
  B1_f1 = f1_score(val_target_labels, pred_target_labels, average="macro")
  B2_f1 = f1_score(val_aggr_labels, pred_aggr_labels, average="macro")

  F1_multi = (A_f1+ B1_f1 + B2_f1)/3

  EMR = computeEMR(list(zip(val_hate_labels, val_target_labels, val_aggr_labels)),
                   list(zip(pred_hate_labels, pred_target_labels, pred_aggr_labels)))

  results_dict = {'A_acc':A_acc,
                  'B1_acc':B1_acc,
                  'B2_acc':B2_acc,
                  'A1_f1':A_f1,
                  'B1_f1':B1_f1,
                  'B2_f1':B2_f1,
                  'F1_multi':F1_multi,
                  'EMR':EMR}

  return results_dict
  

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

def computeEMR(test_labels, pred_labels):
  total_instances = len(test_labels)
  exact_match_count= 0
  for gold, pred in zip(test_labels, pred_labels):
    #print(gold, pred)
    if gold == pred:
      exact_match_count += 1

  return exact_match_count/total_instances