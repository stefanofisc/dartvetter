import tensorflow as tf
import numpy as np

# Define the three shortcut functions in order to convert a standard Tensorflow type to a tf.Example-compatible tf.train.Feature
# The following functions can be used to convert a value to a type compatible with tf.Example
# These functions only uses scalar inputs. The simplest way to handle non-scalar features is to use tf.io.serialize_tensor to convert tensors to binary-strings. Strings are scalars in tensorflow. Use tf.io.parse_tensor to convert the binary-string back to a tensor.

def _bytes_feature(value):
  """Returns a byte_list from a string / byte. """
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()       # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# FloatList and BytesList expect an iterable. So you need to pass it a list of floats.
# Remove extra brackets in _float_feature, ie(value=[value] --> value=value)
def _float_feature(value):
  """Returns a float_list from a float / double. """
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint. """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




def serialize_example(global_view, label, tic, toi):  
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible data type
  feature = {
      'global_view': _float_feature(global_view),
      'av_training_set': _int64_feature(label),
      'tic': _int64_feature(tic),
      'toi': _int64_feature(toi),
      #'toi': _float_feature(toi),
      }
  # Create a Features message using tf.train.Example. Per convertire più features insieme
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


"""
  La function crea un file .tfrecord utilizzando la funzione serialize_example
  per la conversione delle feature in tipi di dato compatibili con il formato
  .tfrecord.
  Input:
    filename: nome del file tfrecord da salvare
    *kargs[]: lista delle feature -> global view, local view, label, TIC
  Output:
    file tfrecord in cui ogni campione è del tipo (gv[],lv[],tic,label)
"""
def generate_tfrecord(filename, gv_list, lbl_list, nobservations, tic_list, toi_list):
# Write the 'tf.Example' observations to the file.
  # Write to the file (filename)
  LENGTH = len(gv_list)
  if nobservations != LENGTH:
    print("nobservations: {} and gv_list: {}\n".format(nobservations, LENGTH))
  with tf.io.TFRecordWriter(filename) as writer:
    for i in range(LENGTH):
      #NOTE. 20230515 Commento. toi = []
      # read data from lists
      global_view = gv_list[i]
      label = lbl_list[i]
      tic = tic_list[i]
      toi = toi_list[i]         
      example = serialize_example(global_view.ravel(), label, tic, toi)
      writer.write(example)





"""
  Stesse due funzioni di sopra, ma senza i label. Usale per generare tfrecord su cui fare predizioni.
"""
def serialize_example_predict(tic_id, global_view, local_view):  
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible data type
  feature = {
      'tic_id': _int64_feature(tic_id),
      'global_view': _float_feature(global_view),
      'local_view': _float_feature(local_view),
      }
  # Create a Features message using tf.train.Example. Per convertire più features insieme
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def generate_tfrecord_predict(filename, tic_id_list, gv_list, lv_list, nobservations):
  # Write the 'tf.Example' observations to the file.
  # Write to the file (filename)
  LENGTH = len(gv_list)
  if nobservations != LENGTH:
    print("nobservations: {} and gv_list: {}\n".format(nobservations, LENGTH))
  with tf.io.TFRecordWriter(filename) as writer:
    for i in range(LENGTH):
      # read data from lists
      tic_id = tic_id_list[i]
      global_view = gv_list[i]
      local_view = lv_list[i]
      example = serialize_example_predict(tic_id, global_view.ravel(), local_view.ravel())
      writer.write(example)









def serialize_example_test(feature0, feature1, feature2, feature4):  
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible data type
  feature = {
      'kepid': _int64_feature(feature0),
      'tce_plnt_num': _int64_feature(feature1),
      'av_training_set': _bytes_feature(feature2),
      'local_view': _float_feature(feature4),
      }
  # Create a Features message using tf.train.Example. Per convertire più features insieme
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

if __name__ == '__main__':
  print("TFRecord main")

