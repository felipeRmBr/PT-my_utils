import pickle
from pandas import DataFrame

#******************* mergeDataFrames() ******************* 

def mergeDataFrames(list_of_results_filepaths, merged_results_filepath):
  
  """
  Merges a list_of_results DataFrames. Saves the resulting data frame 
  in the given merged_results_filepath

  """
  merged_results = DataFrame()

  for result in list_of_results_filepaths:
    with open(result, 'rb') as result_file:
      result_df = pickle.load(result_file)

    merged_results=merged_results.append(result_df, ignore_index=True)
  
  with open(merged_results_filepath, 'wb') as result_file:
      pickle.dump(merged_results, result_file)

  return merged_results


#******************* plotResultsByParameter1() ******************* 

#source: https://matplotlib.org/3.3.3/gallery/statistics/boxplot_vs_violin.html#sphx-glr-gallery-statistics-boxplot-vs-violin-py
import matplotlib.pyplot as plt
import numpy as np

def plotResultsByParameter1(results_df, params_dict, plot_type = 'boxplot', 
                           params2plot = 'ALL', metric = 'mean_val_acc', y_limits='auto'):

  if y_limits == 'auto':
    y_max = results_df[metric].max()+.007
    y_min = results_df[metric].min()-.007
    y_limits = [y_min, y_max]

  if params2plot == 'ALL':
    #parameters_list = ['kernels_combos']
    parameters_list = params_dict.keys()
  else:
    parameters_list = params2plot

  for parameter in parameters_list:
    parameter_values = params_dict[parameter]
    x_labels = [str(v) for v in parameter_values]
    n_labels = len(x_labels)

    fig, axis = plt.subplots(figsize=(12, 6.75))
    acc_data = [results_df.loc[results_df[parameter] == p][metric].array for p in parameter_values]

    if plot_type == 'boxplot':
      # BoxPlots
      axis.boxplot(acc_data)
      axis.set_title('{} results by {}'.format(metric,parameter))
      axis.set_ylim(*y_limits)

      axis.yaxis.grid(True)
      axis.set_xticks([y + 1 for y in range(n_labels)])
      axis.set_xlabel(parameter)
      axis.set_ylabel(metric)
    elif plot_type == 'violin':
      # Violin plot
      axis.violinplot(acc_data,
                    positions = [ y + 1 for y in range(n_labels) ],
                    showmeans=False,
                    showmedians=True)
      axis.set_title('{} results by {}'.format(metric,parameter))
      axis.set_ylim(*y_limits)

      axis.yaxis.grid(True)
      axis.set_xticks([ y + 1 for y in range(n_labels) ])
      axis.set_xlabel(parameter)
      axis.set_ylabel(metric)

    # add x-tick labels
    plt.setp(axis, xticks=[ y + 1 for y in range(n_labels) ],xticklabels=x_labels)
    plt.show()

#******************* plotResultsByParameter2() ******************* 

#source: https://matplotlib.org/3.3.3/gallery/statistics/boxplot_vs_violin.html#sphx-glr-gallery-statistics-boxplot-vs-violin-py
import matplotlib.pyplot as plt
import numpy as np

def plotResultsByParameter2(results_df, params_dict, params2plot = 'ALL'):
  metric_1 = 'mean_acc'
  metric_2 = 'mean_val_acc'

  y_max = max(results_df[metric_1].max(), results_df[metric_2].max()) + .007
  y_min = min(results_df[metric_1].min(), results_df[metric_2].min()) - .007
  y_limits = [y_min, y_max]
  x_shift = 2.5
  box_width = 1.25
  tiks_d = 4

  if params2plot == 'ALL':
    #parameters_list = ['kernels_combos']
    parameters_list = params_dict.keys()
  else:
    parameters_list = params2plot

  for parameter in parameters_list:
    parameter_values = params_dict[parameter]
    x_labels = [str(v) for v in parameter_values]
    N_labels = len(x_labels)
    enumerate_labels = np.array( list(range(N_labels)) )

    fig, axis = plt.subplots(figsize=(12, 6.75))
    acc_data = [results_df.loc[results_df[parameter] == p][metric_1].array for p in parameter_values]
    val_acc_data = [results_df.loc[results_df[parameter] == p][metric_2].array for p in parameter_values]

    # BoxPlots
    axis.boxplot(acc_data, positions=(enumerate_labels*tiks_d) + x_shift -(.25 + box_width/2), widths=[box_width]*N_labels)
    axis.boxplot(val_acc_data, positions=(enumerate_labels*tiks_d)+ x_shift + (.25 + box_width/2), widths=[box_width]*N_labels)
    axis.set_title('{} results by {}'.format('accuracy',parameter))
    axis.set_ylim(*y_limits)

    axis.yaxis.grid(True)
    axis.set_xticks(enumerate_labels*tiks_d)
    axis.set_xlabel(parameter)
    axis.set_ylabel('accuray/val_accuracy')

    # add x-tick labels
    plt.setp(axis, xticks= (enumerate_labels*tiks_d)+x_shift , xticklabels=x_labels)
    plt.show()

#******************* sort_results() ******************* 

def sort_results(results_df, min_val_acc, trade_factor):
  """
  Sorts results_df using the customModelsSort() method.
  Only those results where val_cc >= min_val_acc are considered.
  The trade_factor parameter sets the importance given to val_acc vs 
  overfitting growth (see customModelsSort() method).

  inputs:
  results_df        - pandas.DataFrame, set of results
  min_val_acc       - float
  trade_factor      - int > 0

  outputs:
  sorted_results    - pandas.DataFrame

  """
  sorted_results = results_df.sort_values(by = 'mean_val_acc', ascending=False, inplace=False)
  sorted_results = sorted_results.loc[sorted_results['mean_val_acc']>min_val_acc]

  values_list = sorted_results.loc[:,['mean_acc', 'mean_val_acc']].values.tolist()
  index_list = sorted_results.index.to_list()

  list_of_models = list()
  for i in range(len(index_list)):
    list_of_models.append([index_list[i]] + values_list[i])

  sorted_indexes = [entry[0] for entry in sortListOfModels(list_of_models, trade_factor=trade_factor)]

  sorted_results = sorted_results.loc[sorted_indexes]

  return sorted_results

def sortListOfModels(list_of_models, trade_factor): 
  current_position = 0

  while current_position <= len(list_of_models)-2:
    new_position = current_position
    A = list_of_models[current_position]

    for position2compare in range(current_position+1,len(list_of_models)):
      B = list_of_models[position2compare]
      if customModelsSort(A, B, trade_factor=trade_factor) == 1:
        new_position = position2compare

    # print(current_position, new_position)

    if current_position != new_position:
      list_of_models[current_position], list_of_models[new_position] = list_of_models[new_position], list_of_models[current_position]
    else:
      # list stay as it is
      current_position += 1

  return list_of_models


def customModelsSort(A, B, trade_factor):
  '''
  Sorts models A and B using two metrics for comparison: val_acc and overfitting.
  Overfitting is stimated using the difference between train_acc and val_acc.
  
  The main idea of the function is to reward the models with the best trade off 
  between accuracy inprovement and overfitting growth.
  
  input:
  A       -- tuple, (index, acc, val_acc)
  B       -- tuple, (index, acc, val, acc)

  Precondition: Model A is currently above B in a descending sorted list

  output:
  boolean  -- 0 -> A and B should stay in the order they currently are
              1 -> A and B should swap positions

  
  -----

    TESTING EXAMPLE:

    B = (1, 90, 85)
    A = (2, 86, 84.95)

    test_1 = customModelsSort(A, B, 5)
    test_2 = customModelsSort(B, A, 5)

    test_1, test_2


  '''

  _, A_acc, A_val_acc = A
  _, B_acc, B_val_acc = B

  A_overfit = A_acc - A_val_acc
  B_overfit = B_acc - B_val_acc 

  if A_val_acc == B_val_acc:
    if A_overfit <= B_overfit:
      return 0  # keep A and B in the order they currently are
    else:
      return 1  # swap A and B

  elif A_val_acc > B_val_acc:
    acc_inprovement = A_val_acc - B_val_acc
    overfit_growth = A_overfit - B_overfit

    #print(acc_inprovement)
    #print(overfit_growth)

    if acc_inprovement >= overfit_growth/trade_factor:
      return 0  # keep A and B in the order they currently are
    else:
      return 1  # swap A and B

  elif B_val_acc > A_val_acc: 
    acc_inprovement = B_val_acc - A_val_acc
    overfit_growth = B_overfit - A_overfit

    #print(acc_inprovement)
    #print(overfit_growth)

    if acc_inprovement <= overfit_growth/trade_factor:
      return 0  # keep A and B in the order they currently are
    else:
      return 1  # swap A and B



