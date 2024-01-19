import tensorflow as tf
from tensorflow import keras
from input_pipeline import parse_dataset
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import date

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

sys.path.insert(1, 'path-to-cnn_directory/')
from cnn_architecture import build_model as build_model_globalview

sys.path.insert(2, 'path-to-preprocessing_directory')
from utils import save_prediction_planetpatrol, save_tic_in_file

import argparse

def getArgs(argv=None):
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "--tfrecord_dir",
    type=str,
    required=True,
    help="Base folder containing TFRecord data.")
  
  args = parser.parse_args()

  return parser.parse_args(argv)


def load_testset(name="kepler",return_filenames=False):
  """
    Load TFRecord Dataset.
    Input:
      - name: dataset name.
      - return_filenames: specify whether to return the list of tfrecord filenames or not.
  """
  # Set dataset name
  dname = name
  
  if name=='kepler' or name=='tess':
    filename='train-*'
  elif name=="mix":
    filename='*_train-*'
  else:
    exit(-1)

  argv = None
  args = getArgs()

  TFRECORD_DIR = args.tfrecord_dir
  
  TEST_FILENAMES = tf.io.gfile.glob(TFRECORD_DIR + filename)
  
  test_dataset = parse_dataset(TEST_FILENAMES, "test", dname, False)
  
  
  if return_filenames:
    return test_dataset, TEST_FILENAMES

  return test_dataset


"""
  # NOTE. Old method, to be removed (?)
def load_model(i):
  # Load model and weights
  ep = "10"
  data = "8giu2022/"
  PATH = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/" 
  MODEL_PATH = PATH + "CNN/cnn_201_61_paddsame_1fc1024/cnn"
  CKPT_PATH =  PATH + "trained_models/pretrain_kepler_40ep_paddsame_9fold_1fc_lr10-5_trainTESS10ep_finetunefc_" + data + str(i)  + "/cp-best_model_" + ep + ".ckpt"
  
  my_model = keras.models.load_model(MODEL_PATH)
  load_status = my_model.load_weights(CKPT_PATH)  #by_name=True only if model is saved in .h5 format. tf format supports by_name=False only
  
  print(load_status.assert_existing_objects_matched())
  
  return my_model
"""


def load_model_dropout(t, CKPT_PATH, p, number_of_views=2):
  """
    Load the model with dropout and batch normalization
    input views: global and local;
    conv filters size: 3x1
  """
  #NOTE: Build the model in test mode
  # Set trainingval to False to let dropout and batch normalization layers to run their forward pass in inference mode
  my_model = build_model_globalview(False, t, rateval=p)
  if CKPT_PATH == "":
    return my_model
  load_status = my_model.load_weights(CKPT_PATH)  #by_name=True only if model is saved in .h5 format. tf format supports by_name=False only
  
  print(load_status.assert_existing_objects_matched())
  
  return my_model


def extract_ypred(my_model, test_dataset):
  """
    Method to get the predictions the model computed for each TCE of the test dataset 
  """ 
  #Predictions
  results = my_model.predict(test_dataset)
  # Initialize lists
  y_pred = []
  for i in range(len(results)):
    y_pred.append(float(results[i]))
    #y_pred.append(np.round(results[i]))

  return y_pred


def my_precision_recall_curve(models, test_dataset, prec_filename):
  """
    Method to compute and display the precision-recall curve
  """ 
  # Get true labels
  y_true_list = extract_label_from_dataset(test_dataset)[0]
  # Data structures initialization
  labels = ['DART-Vetter']#, 'KT03', 'KT04', 'KT05']#, 'KFT', 'K']
  colors = ['orange']#, 'royalblue', 'seagreen', 'salmon']#, 'violet', 'purple']
  # Array of results. y_pred[i] contains the prediction of the i-th model
  y_pred = []
  # arrays in which to save precision and recall values at different thresholds
  prec  = []
  rec   = []
  threshold = []
  
  # Get models predictions
  for i in range(len(models)):
    y_pred.append( extract_ypred(models[i], test_dataset) )
 
  
  # Compute precision-recall curve
  for i in range(len(y_pred)):
    p, r, t = precision_recall_curve(y_true_list, y_pred[i])
    prec.append(p)
    rec.append(r)
    threshold.append(t)
  
  plt.figure(1)
  for i in range(len(models)):
    plt.plot(rec[i], prec[i], '--', color=colors[i], label=labels[i])  
  
  plt.plot([0, 1], [1, 0], 'k--', label='Weak classifier')
  plt.legend(loc='lower left', fontsize='small')
  plt.title("Precision-Recall Curve")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.savefig(prec_filename, dpi=1200)
  plt.close()
  

def extract_label_from_dataset(test_dataset,dname='tey2022'):
  """
    Given the test_dataset (type:TFRecordDataset), extract the label from the tuple <feature, label>  
    Output:
      -label_list: list of tce dispositions extracted from test_dataset;
      -id_list: list of tic or toi extracted from test_dataset.
  """
  label_list = []
  tic_list = []
  id_list = []
  idx = 0
  for element in list(test_dataset.as_numpy_iterator()):
    # Extract labels
    for label_i in element[1]:
      label_list.append(int(label_i))
    #for tic_i in element[2]:
    #  tic_list.append(int(tic_i))
    for toi_i in element[2]:
      id_list.append(toi_i)
    
  return label_list, id_list


def predict_apj_testset(my_model, test_dataset, filepath_apj, model_name, save_results=True, classification_threshold=0.5):
  # Initialize data structures
  label_list = []
  tic_list = []
  toi_list = []
  eval_metrics = [0, 0, 0, 0, 0, 0, 0, 0] #tp, tn, fp, fn, precision, recall, accuracy, fscore
  
  # Filename in which to store results: <TOI, Disposition, Prediction>
  if save_results:
    print("Creating folder {}\n".format(filepath_apj + model_name))
    os.mkdir(filepath_apj + model_name)
    filename_results = filepath_apj + model_name + "/" + "tce_predictions" + model_name + "_" + date.today().strftime("%m%d%Y") + ".txt"
    with open(filename_results, 'a'):
      pass        
  # The pass statement is used as a placeholder for future code. When pass is executed, nothing happens, but you avoid getting an error when empty code is not allowed
  results = my_model.predict(test_dataset)    
  # Extract real dispositions from the test set
  label_list, tic_list = extract_label_from_dataset(test_dataset)
  
  idx = 0
  for tic_i in tic_list:
    pred_i = 0 #initialize pred_i
    if classification_threshold == 0.5:
      pred_i = np.round(results[idx])
    else:
      if results[idx] >= classification_threshold:
        pred_i = 1
    
    if pred_i == 1 and label_list[idx] == 1:   #true positive 
      eval_metrics[0] += 1
      if save_results:
        save_tic_in_file(filepath_apj + model_name + "/" + "tp.txt",int(tic_i))
    elif pred_i == 0 and label_list[idx] == 0: #true negative
      eval_metrics[1] += 1
      if save_results:
        save_tic_in_file(filepath_apj + model_name + "/" + "tn.txt",int(tic_i))
    elif pred_i == 1 and label_list[idx] == 0: #false positive
      eval_metrics[2] += 1
      if save_results:
        save_tic_in_file(filepath_apj + model_name + "/" + "fp.txt",int(tic_i))
    else:                                      #false negative
      eval_metrics[3] += 1
      if save_results:
        save_tic_in_file(filepath_apj + model_name + "/" + "fn.txt",int(tic_i))
    #NOTE: Save the result: <TOI, Disposition, results[idx]>
    if save_results:
      save_prediction_planetpatrol(filename_results, tic_i, label_list[idx], results[idx])
    idx += 1
  
  # Compute precision, recall, accuracy and fscore
  eval_metrics[4] = eval_metrics[0] / (eval_metrics[0] + eval_metrics[2])
  eval_metrics[5] = eval_metrics[0] / (eval_metrics[0] + eval_metrics[3])
  eval_metrics[6] = ( eval_metrics[0] + eval_metrics[1] ) / ( eval_metrics[0] + eval_metrics[1] + eval_metrics[2] + eval_metrics[3] )
  eval_metrics[7] = 2 * (eval_metrics[4] * eval_metrics[5]) / (eval_metrics[4] + eval_metrics[5])
  
  return eval_metrics 


def plot_roc_curve(models, test_dataset, roc_filename):
  """
    Plot the roc curves for all the input models.
    Input:
      models: a vector containing the models for which roc curve has to be computed;
      test_dataset: TFRecordDataset object consisting in the test set.
    Output:
      a plot in which the roc curves of all the input models are displayed.
  """
  # Get true labels
  y_true_list = extract_label_from_dataset(test_dataset)[0]
  # Data structures initialization
  labels = ['DART-Vetter']#, 'KT03', 'KT04', 'KT05']#, 'KFT', 'K']
  colors = ['orange']#, 'royalblue', 'seagreen', 'salmon']#, 'violet', 'purple']
  # Vettore di risultati. y_pred[i] contiene le predizioni dell'i-simo modello. 
  y_pred = []
  # Vettore dei risultati delle roc curve calcolate per ogni modello
  fpr = []
  tpr = []
  thresholds = []
  auc_vec = []
  
  # Get models predictions. Then compute roc_curves and auc
  for i in range(len(models)):
    y_pred.append( extract_ypred(models[i], test_dataset) )
    fpr_i, tpr_i, th_i = roc_curve(y_true_list, y_pred[i] )
    fpr.append( fpr_i )
    tpr.append( tpr_i )
    thresholds.append( th_i )
    auc_vec.append( auc( fpr[i], tpr[i] ) )
  
  # Plot the ROC curve for the classifiers
  
  plt.figure(1)
  # Comment the following two lines if you don't want a zoom in view of the upper left corner 
  #plt.xlim(0, 0.4)
  #plt.ylim(0.6, 1)
  plt.plot([0, 1], [0, 1], 'k--', label='Weak classifier')
  for i in range(len(models)):
    plt.plot(fpr[i], tpr[i], '--', label=labels[i] + ' (area = {:.3f})'.format(auc_vec[i]), color=colors[i])
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()
  plt.savefig(roc_filename, dpi=1200)
  plt.close()
  

if __name__ == '__main__':
  """
   NOTE.  
      [In input_pipeline\_parse_function_tfrecord()] It is essential to comment the line "ret_tuple = (feature, label)" when testing the model. At the same time, the corresponding line below (ret_tuple = (feature, label, sample_id)) has to be decommented of course.
  """
  # Set input variables
  root_path = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/trained_models/"
  test_set_filenames_type = 'mix' # set to mix if you have tfrecords filenames like '*_train-*.tfrecord'. Set to 'tess' for tfrecord filenames like "train-*"
  return_test_set_filenames = False
  ep = "05"                       # number of epochs. This value is visible in the name of the checkpoint file
  suffix = "/0/cp-best_model_" + ep + ".ckpt" 
  date_model = '20230806'
  model_directory_name = 'dartvetter'
  classification_threshold = 0.5  # this is an hyperparameter. See Section 7 of our paper for more details 
  # Variables for saving the results
  flag_save_result = True
  dest_folder = 'results-directory'       # name of the main folder in which to store predictions of your model on different dataset
  model_application_results = 'some-name' # name of the folder in which to store results about the application of the model on a given dataset

  # Get test set
  test_dataset  = load_testset(test_set_filenames_type, return_test_set_filenames) 
  # Get TCEs dispositions from test set
  label_list, id_list = extract_label_from_dataset(test_dataset, test_set_filenames_type) 
  #print("Labels: ",len(label_list))
  #print("id list: {};\n".format(len(id_list)))
  
  # Load the model
  # 1. Set the model path
  ckpt_array = []
  ckpt_array.append(root_path + date_model + model_directory_name + suffix) # array of models allows you to load more than 1 model
  # 2. Load model
  model_name = [model_application_results] 
  my_models = []
  pval = [0.3]  # dropout probability rate
  for i in range(len(ckpt_array)):
    my_models.append( load_model_dropout(0.5, ckpt_array[i], pval[i], 1) )
    #print(my_models[i].summary()) # decomment to print model's architecture

  # Evaluate the model
  for i in range(len(model_name)):
    res = predict_apj_testset(my_models[i], test_dataset, dest_folder, model_name[i], flag_save_result, classification_threshold)
    print("{} - TP={}; TN={}; FP={}; FN={}; Precision={}; Recall={}; Accuracy={}; F-score={};\n".format(model_name[i],res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7]))
  
  exit(0)
  # Precision-recall curves and ROC curve
  """
  prec_filename = dest_folder + 'prcurves/precision_recall_curve_qlpspoc-tt9_bs128.pdf'
  roc_filename = dest_folder + 'roccurves/roc_curves_qlpspoc-tt9_bs128.pdf'
  
  my_precision_recall_curve(my_models, test_dataset, prec_filename)
  plot_roc_curve(my_models, test_dataset, roc_filename)
  """

  

