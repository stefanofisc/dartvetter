import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from transfer_learning import load_model
import random

import sys
sys.path.insert(1, 'path/to/TFRecord/')
from reading_with_ParseFromString import count_total_num_samples


BATCH_SIZE = 128
BATCH_SHUFFLE = 2000

def getArgs(argv=None):
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name (e.g. standard or mix). This value determines the tfrecords filename's structure in split into training validation.")
  

  parser.add_argument(
    "--tfrecord_dir",
    type=str,
    required=True,
    help="Base folder containing TFRecord data.")
  
  parser.add_argument(
    "--dest_folder",
    type=str,
    required=True,
    help="Destination folder in which to save trained models")

  parser.add_argument(
    "--k_fold",
    type=int,
    required=True,
    help="Number of training process to perform")

  parser.add_argument(
      "--epochs",
      type=int,
      required=True,
      help="Number of training epochs + 1")

  parser.add_argument(
      '--pretrain',
      choices=('True','False'),
      required=True,
      help="Specify if you want to load pre-trained model weights."
      )
  
  parser.add_argument(
      '--finetuning',
      choices=('True','False'),
      required=True,
      help="Specify if you want to perform finetuning on fully-connected layers."
      )
  
  parser.add_argument(
      '--shuffle_dataset',
      choices=('True','False'),
      required=True,
      help="Specify whether to shuffle tfrecord files list or not before to split them into training and validation."
      )
  
  parser.add_argument(
    "--learning_rate",
    type=float,
    required=True,
    help="Learning rate value")

  parser.add_argument(
    "--rateval",
    type=float,
    required=True,
    help="Dropout rate probability. Should be in [0.2, 0.5] (Srivastava et al. 2014).")
  
 
  return parser.parse_args(argv)


def getVar():
  args          = getArgs()
  
  DNAME         = args.dataset
  TFRECORD_DIR  = args.tfrecord_dir
  DEST_FOLDER   = args.dest_folder 
  K_FOLD        = args.k_fold
  EPOCHS        = args.epochs
  PRETRAIN      = args.pretrain == 'True'
  FINETUNING    = args.finetuning == 'True'
  SHUFFLE       = args.shuffle_dataset == 'True'
  LR            = args.learning_rate
  RATEVAL       = args.rateval

  
  return DNAME, TFRECORD_DIR, DEST_FOLDER, K_FOLD, EPOCHS, PRETRAIN, FINETUNING, SHUFFLE, LR, RATEVAL


def split_into_training_validation(tfrecord_dir, i, dataset='mix', shuffle_dataset=False):
  """
    Take as input the number that identifies the current training and based on it split the tfrecord chunks into training set and validation set. Return the file list of the training set and validation set.  Input:
        tfrecord_dir: tfrecord source directory;
        i: training number (k-fold cross validation);
        dataset: based on the value of this string, the tfrecord filename's structure will change accordingly. If dataset='standard' so the tfrecords of type "train-*.tfrecord" will be considered. If dataset='mix' the filename structure changes in "*_train*"
        shuffle_dataset: Specify whether to shuffle tfrecord files list or not. #NOTE Only for single training tests. NO cross-validation. type: bool.
      Output:
        list of tfrecord filenames divided into training and validation set.
      Example:
      >>> list = np.arange(10)
      >>> n = len(list)
      >>> chunk_size = n*0.1 #set validation size
      >>> for number_of_training in range(10):
        >>> a = int(number_of_training*chunk_size)
        >>> b = int(a+chunk_size)
        >>> val = list[a:b]
        >>> train = np.delete(list,np.s_[a:b])
        >>> print(val)
        >>> print(train)
  """
  # To obtain the list of all tfrecord files in the folder ${tfrecord_dir}
  # filename template: <telescope_name>_train-<proc_id>.tfrecord. Example: tess_train-0.tfrecord
  if dataset == 'mix':
    tfrecord_files = tf.io.gfile.glob(tfrecord_dir + "*_train*")    # returns a list of files that match the given pattern(s)
  elif dataset == 'standard':
    tfrecord_files = tf.io.gfile.glob(tfrecord_dir + "train-*")     # returns a list of files that match the given pattern(s)
  else:
    print("Error in split_into_training_validation: you must specify the tfrecords filename's structure\n")

  #NOTE: Do not shuffle if you are in k-fold cross validation mode.
  if shuffle_dataset:
    random.shuffle(tfrecord_files)
  # Get number of tfrecord files
  n = len(tfrecord_files)
  # Set validation set size to 10% of dataset size
  chunk_size = n*0.1

  a = int(i*chunk_size)
  if (n % 2) == 0: 
    b = int(a+chunk_size)
  else:
    b = int(a+chunk_size)+1
    if i>0:
      a+=1
      b+=1
  
  VALID_FILENAMES = tfrecord_files[a:b]
  TRAINING_FILENAMES = np.delete(tfrecord_files, np.s_[a:b])

  return TRAINING_FILENAMES, VALID_FILENAMES
  

# Decoding and read the data
"""
  The following two methods are used to map the elements stored within the .tfrecord files 
  TFRecord example:
    Global view: [201]        This is the TCE being analyzed by the model
    av_training_set: 0 or 1   This is the TCE disposition
"""
def _feature_description_tfrecord():
  feature_description = {
    'av_training_set': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'global_view': tf.io.VarLenFeature(tf.float32),
    'toi': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'tic': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
  return feature_description


def _parse_function_tfrecord(example_proto):
  GLOBAL_VIEW_LENGTH = 201
  # Parse the input 'tf.train.Example' proto using the dictionary above.
  feature_description = _feature_description_tfrecord()
  feature = tf.io.parse_single_example(example_proto, feature_description)

  label = feature['av_training_set']
  feature["global_view"] = tf.reshape(tf.sparse.to_dense(feature['global_view']),[GLOBAL_VIEW_LENGTH])

  sample_id = feature['toi'] #feature['tic'] when toi label is not present
  del feature['tic']
  del feature['toi']
  del feature['av_training_set']        # label, not feature


  ret_tuple = (feature, label)                  # Use when Training
  #ret_tuple = (feature, label, sample_id)      # Use when Testing

  return ret_tuple


def load_dataset(filenames, mode, d_name="kepler"):
  # Load dataset given TFRecords
  dataset = tf.data.TFRecordDataset(filenames)
  
  if mode == 'train':
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False   # disable order, increase speed
    dataset = dataset.with_options(ignore_order)      # use the data as they are transmitted, instead of in their original order
  
  dataset = dataset.map(_parse_function_tfrecord)
 

  return dataset


def parse_dataset(filenames, mode="train", d_name="kepler", shuffle=True, BATCH_SHUFFLE=BATCH_SHUFFLE, EPOCHS=50, BATCH_SIZE=BATCH_SIZE):
 
  dataset = load_dataset(filenames, mode, d_name)
  if shuffle == True:
    dataset = dataset.shuffle(buffer_size=BATCH_SHUFFLE)

  if mode == 'train':
    dataset = dataset.repeat(EPOCHS)

  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=1)

  return dataset


def _train_model(DNAME, TFRECORD_DIR, number_of_training, dest_folder, epochs, PRETRAIN, FINETUNING, SHUFFLE, prc_threshold_i, LR, RATEVAL):
  # Use number of training as model name
  MODEL_NAME = str(number_of_training) 
  # When using k-fold cross validation, generate different training set as follows
  TRAINING_FILENAMES, VALID_FILENAMES = split_into_training_validation( TFRECORD_DIR, number_of_training, DNAME, SHUFFLE )
  
  print("Training: {}\n Training files: {}\n Validation files: {}\n".format(number_of_training,TRAINING_FILENAMES,VALID_FILENAMES))

  # Visualize input
  dname = "mix" 
  train_dataset = parse_dataset(TRAINING_FILENAMES, d_name=dname, EPOCHS=epochs)
  valid_dataset = parse_dataset(VALID_FILENAMES, d_name=dname, EPOCHS=epochs)
  
  # Load model
  my_model = load_model('TESS', LR, RATEVAL, False, PRETRAIN, str(number_of_training), FINETUNING, prc_threshold_i) 

  #class 1: PC; class 0: NPC. Count the number of samples for each class
  n_samples_class_1, n_samples_class_0 = count_total_num_samples(TFRECORD_DIR)
  print("Dataset size: PC={}; NPC={}\n".format(n_samples_class_1, n_samples_class_0))
  DATASET_SIZE = n_samples_class_0 + n_samples_class_1
  # Defining data split
  TRAINING_SPLIT = 0.9
  VALIDATION_SPLIT = 0.1
  n_classes = 2
  cw_0 = DATASET_SIZE / (n_classes * n_samples_class_0)
  cw_1 = DATASET_SIZE / (n_classes * n_samples_class_1)
  # Define training steps for each epoch
  STEPS_PER_EPOCH = int(TRAINING_SPLIT * DATASET_SIZE / BATCH_SIZE)
  VALIDATION_STEPS = int(VALIDATION_SPLIT * DATASET_SIZE / BATCH_SIZE)
  
  # Class weighting wj = dataset_size / (n_classes * n_samples_classj)
  class_weight = {0: cw_0, 1: cw_1} 
  print("Class weighting: NPC={}; PC={}\n".format(cw_0, cw_1)) 
  # Callbacks
  checkpoint = keras.callbacks.ModelCheckpoint(
      filepath=dest_folder + MODEL_NAME + "/cp-best_model_{epoch:02d}.ckpt",
      save_weights_only=True,
      period=epochs-1,
      verbose=1,
      )
  # The Model.fit method adjusts the model parameters to minimize the loss:
  history = my_model.fit(
      x=train_dataset,
      validation_data=valid_dataset,
      validation_steps=VALIDATION_STEPS,
      epochs=epochs,
      steps_per_epoch=STEPS_PER_EPOCH,
      class_weight=class_weight,
      callbacks=[checkpoint],
      verbose=1,
      )

  return history
  

def _plot_accuracy(h, path, number_of_training):
  plt.plot(h.history['accuracy'])
  plt.plot(h.history['val_accuracy'])
  plt.title('model accuracy - Training#' + number_of_training)
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(path + "training_plot/accuracy_" + number_of_training + ".png")
  plt.close()


def _plot_loss(h, path, number_of_training):
  plt.plot(h.history['loss'])
  plt.plot(h.history['val_loss'])
  plt.title('model loss - Training#' + number_of_training)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(path + "training_plot/loss_" + number_of_training + ".png")
  plt.close()


def _plot_metrics(history, path, number_of_training):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8, 1])
    else:
      plt.ylim([0, 1])
    plt.legend()
  
  plt.savefig(path + "training_plot/training_" + number_of_training + ".png")
  plt.close()


def main():
  """
    Input parameters:
        TFRECORD_DIR:     folder in which input tfrecords are stored
      - DEST_FOLDER:      path to the folder in which to save results
      - K_FOLD:           number of training processes. 1 if Dropout. >1 when cross-validation
      - EPOCHS:           number of training epochs
      - PRETRAIN:         boolean value to specify whether or not to load pre-trained model weights with Kepler data
      - FINETUNING:       boolean value to specify whether or not to fine-tune during training or train the entire model
    NOTE.  
      [In _parse_function_tfrecord()] It is essential to comment the line "ret_tuple = (feature, label)" when testing the model. At the same time, the corresponding line below (ret_tuple = (feature, label, sample_id)) has to be decommented of course.
  """
  # Get input parameters
  DNAME, TFRECORD_DIR, DEST_FOLDER, K_FOLD, EPOCHS, PRETRAIN, FINETUNING, SHUFFLE, LR, RATEVAL = getVar()
  # Create the directory in which to save the trained model
  os.mkdir(DEST_FOLDER)
  os.mkdir(DEST_FOLDER + 'training_plot')

  # K-fold cross validation. Set to 1 if Dropout
  for i in range(K_FOLD):  
    h = _train_model(DNAME, TFRECORD_DIR, i, DEST_FOLDER, EPOCHS, PRETRAIN, FINETUNING, SHUFFLE, 0.5, LR, RATEVAL) #i+1
    try:
      _plot_metrics(h, DEST_FOLDER, str(i)) #i+1
    except:
      print("Errore nel plot del training {}\n".format(i))


if __name__ == '__main__':
  main()
