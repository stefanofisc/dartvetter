import tensorflow as tf
import sys
sys.path.insert(1,'path-to-cnn_directory/')
from cnn_architecture import build_model, compile_model


def get_model_path(model_name='Kepler', padding=True):
  """
    Path to the base model architecture. Use this method when you want to load an existing model.
    Input:
      - model_name: specifica il tipo di modello, tra Kepler e TESS. :str
      - padding: specifica se vuoi caricare il modello con o senza padding 'same'. :bool
  """
  # A list of different models we tested in the past. Change according to your needs.
  if model_name == 'Kepler':
    if padding == False:
      MODEL_PATH = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/CNN/cnn_kepler_2001_201_nopadding/cnn" 
    else:
      MODEL_PATH = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/CNN/cnn_kepler_2001_201/cnn"
  elif model_name == 'TESS':
    MODEL_PATH = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/CNN/cnn_201_61_paddsame_1fc1024/cnn" 
  else:
    raise Exception("model_name error: input must be 'Kepler' or 'TESS'\n")
  
  return MODEL_PATH


def set_trainable_layers(my_model):
  """
    Fine-tuning. Set fully-connected levels trainable and freeze convolutional levels.
    When you set bn_layer.trainable = False, the BatchNormalization layer will run in inference mode, and will not update
    its mean and variance statistics. This is not the case for other layers in general, as weight trainability and
    inference/training modes are two orthogonal concepts. But the two are tied in the case of the BatchNormalization layer.
  """
  # Specify the name of the trainable layers
  layer_trainable = ['Output','FC-512-a','DO_FC1']
  print("Trainable variables before: {}\n".format(len(my_model.trainable_variables)))
  
  try:
    for layer in my_model.layers:
      if layer.name not in layer_trainable:
        #print("The layer {} is not trainable\n".format(layer.name))
        # Move all the layer's weights from trainable to non-trainable
        layer.trainable = False
  except:
    print("Error in set_trainable_layers")
    return -1

  print("Trainable variables after: {}\n".format(len(my_model.trainable_variables)))

  return my_model


def load_model(MISSION, LR, RATEVAL, existing=True, pretrain=False, MODEL_NAME="1", finetuning=False, prc_threshold_i=0.5):
  """
    Method used in input_pipeline to load the model. Allows you to specify whether to load the trained weights and whether to finetune them in the new training.  Input:
      Input:
        - MISSION: 'Kepler' or 'TESS'. :str
        - existing: specify whether load an existing model or compile the model run-time :bool
        - pretrain: specify if you want to load weights. :bool
        - MODEL_NAME: the model's name, a number. To be specified only if you want load pre-trained model weights :str
        - finetuning: specify if you want to perform fine-tuning on fully-connected layers. :bool
      Output:
        - my_model: keras model
  """
  if existing:
    # Load an existing model
    my_model = tf.keras.models.load_model( get_model_path(MISSION,True) )
  else:
    # Compile the model run-time
    if finetuning == True:
      # In fine-tuning we make sure that the base model is running in inference mode
      trainingval = False
    else:
      # Let the dropout and batch normalization layers run their forward pass in training mode
      # Train from scratch
      trainingval = True
    
    my_model = build_model(trainingval, prc_threshold_i, finetuning, LR, RATEVAL)
  
  # NOTE: Load the weights of the pre-trained model
  if pretrain:
    root_path   = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/trained_models/"
    ep          = '10'   
    lr          = '1e-3'
    rateval     = 'rateval03'
    data        = '20230531'
    arch_details = 'fc512_conv3layer_batchnorm'
    CKPT_PATH = root_path + data + "_training_kepler_" + ep + "ep_horizreflpc_" + lr + "_" + rateval + "_" + arch_details + "/" + MODEL_NAME + "/cp-best_model_" + ep + ".ckpt"
    load_status = my_model.load_weights(CKPT_PATH) 
    print(load_status.assert_existing_objects_matched())
  
  # Train some layers while freezing the weights of layers you don't want to train
  if finetuning:
    my_model = set_trainable_layers(my_model)
    # Calling compile() on a model is meant to freeze the behavior of that model. This implies that the trainable attribute
    # values at the time the model is compiled should be preserved throughout the lifetime of that model, until compile
    # is called again. Hence, if you change any trainable value, make sure to call compile() again on your model for your
    # changes to be taken into account.
    my_model = compile_model(my_model, prc_threshold_i, LR)

  # NOTE: experimental print
  print(my_model.summary())

  return my_model



if __name__ == '__main__':
  print("main transfer learning")