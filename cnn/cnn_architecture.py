import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

CONV_FILTER_SIZE = 3 

def initializer_keras():
  """
    Select a method to initialize the weights of your network architecture
  """
  #initializer = keras.initializers.he_uniform()
  initializer = keras.initializers.lecun_uniform()
  #initializer = tf.zeros_initializer()
  #initializer = keras.initializers.GlorotNormal()
  
  return initializer

  
# Model architecture: CNN 2-inputs. Global view (201), Local view (61)
def local_view(initializer, trainingval, rateval):
  # Local view input
  LV_LENGTH = 61
  local_view = keras.Input(shape=(LV_LENGTH,), name="local_view")
  input_1 = layers.Reshape((LV_LENGTH, 1))(local_view)
  
  # conv -> activation -> spatial dropout x2. pooling -> batch normalization
  input_1 = layers.Conv1D(16, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="LV1")(input_1)
  #input_1 = layers.SpatialDropout1D(rate=rateval, name="DO_L1")(input_1, training=trainingval)
  input_1 = layers.Conv1D(16, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="LV2")(input_1)
  #input_1 = layers.SpatialDropout1D(rate=rateval, name="DO_L2")(input_1, training=trainingval)
  input_1 = layers.MaxPooling1D(pool_size=7, strides=2, name="LV3")(input_1)
  #input_1 = layers.BatchNormalization(name="BN_L1")(input_1, training=trainingval)

  input_1 = layers.Conv1D(32, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="LV4")(input_1)
  #input_1 = layers.SpatialDropout1D(rate=rateval, name="DO_L3")(input_1, training=trainingval)
  input_1 = layers.Conv1D(32, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="LV5")(input_1)
  #input_1 = layers.SpatialDropout1D(rate=rateval, name="DO_L4")(input_1, training=trainingval)
  input_1 = layers.MaxPooling1D(pool_size=7, strides=2, name="LV6")(input_1)
  #input_1 = layers.BatchNormalization(name="BN_L2")(input_1, training=trainingval)
  
  output_1 = layers.Flatten()(input_1)
  #input_1 = keras.Model(inputs=local_view, outputs=input_1)     #inputs: The input(s) of the model: a keras.Input object or list of keras.Input objects
  
  return local_view, output_1


def global_view(initializer, trainingval, rateval): 
  # Global view input
  GV_LENGTH = 201
  
  global_view = keras.Input(shape=(GV_LENGTH,), name="global_view")
  input_2 = layers.Reshape((GV_LENGTH, 1))(global_view)

  input_2 = layers.Conv1D(16, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV1")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G1")(input_2, training=trainingval) 
  input_2 = layers.Conv1D(16, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV2")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G2")(input_2, training=trainingval) 
  input_2 = layers.MaxPooling1D(pool_size=5, strides=2, name="GV3")(input_2)
  input_2 = layers.BatchNormalization(name="BN_G1")(input_2, training=trainingval)
  
  input_2 = layers.Conv1D(32, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV4")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G3")(input_2, training=trainingval) 
  input_2 = layers.Conv1D(32, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV5")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G4")(input_2, training=trainingval) 
  input_2 = layers.MaxPooling1D(pool_size=5, strides=2, name="GV6")(input_2)
  input_2 = layers.BatchNormalization(name="BN_G2")(input_2, training=trainingval)
  
  input_2 = layers.Conv1D(64, CONV_FILTER_SIZE, activation="relu", padding='same',kernel_initializer=initializer, name="GV7")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G6")(input_2, training=trainingval) 
  input_2 = layers.Conv1D(64, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV8")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G7")(input_2, training=trainingval) 
  input_2 = layers.MaxPooling1D(pool_size=5, strides=2, name="GV9")(input_2)
  input_2 = layers.BatchNormalization(name="BN_G3")(input_2, training=trainingval)
  
  input_2 = layers.Conv1D(128, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV10")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G8")(input_2, training=trainingval) 
  input_2 = layers.Conv1D(128, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV11")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G9")(input_2, training=trainingval) 
  input_2 = layers.MaxPooling1D(pool_size=5, strides=2, name="GV12")(input_2)
  input_2 = layers.BatchNormalization(name="BN_G4")(input_2, training=trainingval)
  
  input_2 = layers.Conv1D(256, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV13")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G10")(input_2, training=trainingval) 
  input_2 = layers.Conv1D(256, CONV_FILTER_SIZE, activation="relu", padding='same', kernel_initializer=initializer, name="GV14")(input_2)
  input_2 = layers.SpatialDropout1D(rate=rateval, name="DO_G11")(input_2, training=trainingval) 
  input_2 = layers.MaxPooling1D(pool_size=5, strides=2, name="GV15")(input_2)
  input_2 = layers.BatchNormalization(name="BN_G5")(input_2, training=trainingval)
  
  output_2 = layers.Flatten()(input_2)
  #input_2 = keras.Model(inputs=global_view, outputs=input_2)
  
  return global_view, output_2





def global_view_vgg(initializer, trainingval, rateval):
    """
        Define model architecture
    """
    # Global view input
    GV_LENGTH = 201
    
    # Blocks configuration
    conv_names            = ["GV1","GV2","GV3","GV4","GV5","GV6","GV7","GV8","GV_9","GV_10"]
    dropout_names         = ["DO_G1","DO_G2","DO_G3","DO_G4","DO_G5", "DO_G6","DO_G7","DO_G8","DO_G9","DO_G10"]
    batch_norm_names      = ["BN_1", "BN_2", "BN_3", "BN_4", "BN_5"]
    pool_names            = ["POOL_G1", "POOL_G2", "POOL_G3", "POOL_G4", "POOL_G5"]
    conv_blocks_num       = 5
    conv_filter_size      = 3
    conv_block_filter_factor = 2
    conv_padding          = "same"
    pooling_size          = 5
    pooling_stride        = 2

    # Reshape input
    global_view = keras.Input(shape=(GV_LENGTH,), name="global_view")
    input_2 = layers.Reshape((GV_LENGTH, 1))(global_view)
    
    # Build convolutional blocks
    i_layer = -1
    for i in range(conv_blocks_num):
        i_layer += 1
        #Convolution
        input_2 = layers.Conv1D(
            pow(conv_block_filter_factor,4+i), 
            conv_filter_size, 
            activation="relu", 
            padding=conv_padding, 
            kernel_initializer=initializer, 
            name=conv_names[i_layer]
            )(input_2)
        #Regularization with Dropout
        input_2 = layers.SpatialDropout1D(
            rate=rateval, 
            name=dropout_names[i_layer]
            )(input_2, training=trainingval) 
        i_layer += 1
        #Convolution
        input_2 = layers.Conv1D(
            pow(conv_block_filter_factor,4+i), 
            conv_filter_size, 
            activation="relu", 
            padding=conv_padding, 
            kernel_initializer=initializer, 
            name=conv_names[i_layer]
            )(input_2)
        #Regularization with Dropout
        input_2 = layers.SpatialDropout1D(
            rate=rateval, 
            name=dropout_names[i_layer]
            )(input_2, training=trainingval) 
        #Max pooling. Shrink the extracted features
        input_2 = layers.MaxPooling1D(
            pool_size=pooling_size, 
            strides=pooling_stride, 
            name=pool_names[i]
            )(input_2)
        #Batch Normalization
        input_2 = layers.BatchNormalization(
            name=batch_norm_names[i]
            )(input_2, training=trainingval)
        
        
    
    output_2 = layers.Flatten()(input_2)
    
    return global_view, output_2





def global_view_fiscale(initializer, trainingval, rateval): 
  # Global view input
  GV_LENGTH = 201
  
  # Blocks configuration
  conv_names            = ["GV1","GV2","GV3","GV4","GV5","GV6","GV7","GV8","GV_9","GV_10"]
  dropout_names         = ["DO_G1","DO_G2","DO_G3","DO_G4","DO_G5"]
  batch_norm_names      = ["BN_1", "BN_2", "BN_3", "BN_4", "BN_5"]
  conv_blocks_num       = 5
  conv_filter_size      = 3
  conv_block_filter_factor = 2
  conv_padding          = "same"
  pooling_size          = 5
  pooling_stride        = 2

  # Reshape input
  global_view = keras.Input(shape=(GV_LENGTH,), name="global_view")
  input_2 = layers.Reshape((GV_LENGTH, 1))(global_view)
  
  # Build convolutional blocks
  i_layer = -1
  for i in range(conv_blocks_num):
    i_layer += 1
    #Convolution
    input_2 = layers.Conv1D(
        pow(conv_block_filter_factor,4+i), 
        conv_filter_size, 
        activation='relu', 
        padding=conv_padding, 
        kernel_initializer=initializer, 
        name=conv_names[i_layer]
        )(input_2)
    #Regularization with Dropout
    input_2 = layers.SpatialDropout1D(
        rate=rateval, 
        name=dropout_names[i]
        )(input_2, training=trainingval) 
    i_layer += 1
    #Max pooling. Shrink the extracted features
    input_2 = layers.MaxPooling1D(
        pool_size=pooling_size, 
        strides=pooling_stride, 
        name=conv_names[i_layer]
        )(input_2)
    #Batch Normalization
    input_2 = layers.BatchNormalization(
        name=batch_norm_names[i]
        )(input_2, training=trainingval)
  
  output_2 = layers.Flatten()(input_2)
  
  return global_view, output_2





def global_view_visser(initializer, trainingval, rateval):
  # Global view input
  GV_LENGTH = 201
  
  # Blocks configuration
  conv_names            = ["GV1","GV2","GV3","GV4","GV5","GV6"]
  conv_filter_size_1    = 50
  conv_filter_size_2    = 12
  conv_filter_number    = 64
  conv_padding          = "same"
  pooling_size          = 16
  pooling_stride        = 16
  average_pooling_size  = 12

  # Reshape input
  global_view = keras.Input(shape=(GV_LENGTH,), name="global_view")
  input_2 = layers.Reshape((GV_LENGTH, 1))(global_view)
  
  for i in range(2):
    # Apply two convolutional layers
    input_2 = layers.Conv1D(
          conv_filter_number, 
          conv_filter_size_1, 
          activation="relu", 
          padding=conv_padding, 
          kernel_initializer=initializer, 
          name=conv_names[i]
          )(input_2)
  # Pooling
  input_2 = layers.MaxPooling1D(
      pool_size=pooling_size, 
      strides=pooling_stride, 
      name=conv_names[2]
      )(input_2)
  for i in range(2):
    # Apply two convolutional layers
    input_2 = layers.Conv1D(
        conv_filter_number, 
        conv_filter_size_2, 
        activation="relu", 
        padding=conv_padding, 
        kernel_initializer=initializer, 
        name=conv_names[i+3]
        )(input_2)
  # Average Pooling
  input_2 = layers.AveragePooling1D(
      pool_size=average_pooling_size,
      strides=pooling_stride,
      name=conv_names[5]
      )(input_2)

  output_2 = layers.Flatten()(input_2)
  
  return global_view, output_2





def fully_connected_visser(initializer, input_2, output_2, trainingval, finetuning, rateval):
  if finetuning:
    trainingval_dropout = True
  else:
    trainingval_dropout = trainingval

  x = output_2
  fc_units = 256
  # Define the dense block
  fully_connected = layers.Dropout(
      rate=rateval, 
      name="DO_FC1"
      )(x, training=trainingval_dropout)
  fully_connected = layers.Dense(
      fc_units, 
      activation="relu", 
      kernel_initializer=initializer, 
      name="FC-" + str(fc_units) + "-a"
      )(fully_connected) 
  fully_connected = layers.Dense(
      fc_units, 
      activation="relu", 
      kernel_initializer=initializer, 
      name="FC-" + str(fc_units) + "-b"
      )(fully_connected) 
  # Output layer
  fully_connected = layers.Dense(
      1, 
      activation="sigmoid", 
      kernel_initializer=initializer, 
      name="Output"
      )(fully_connected)

  model = Model([input_2], fully_connected) 
  return model 

 

 

def fully_connected(initializer, input_2, output_2, trainingval, finetuning, rateval):
  # Fully connected layers
  if finetuning == True:
    # Il livello dropout si comporta in modalità training
    trainingval_dropout = True
    print("Training DO_FC1: {}, Training BN_FC1: {}\n".format(trainingval_dropout, trainingval))
  else:
    trainingval_dropout = trainingval

  x = output_2
  # Set the number of fully-connected neurons.
  # Use x.shape[1] to set the number of fc neurons equal to the dimension of the feature vector
  fc_units = 512
  # Define the dense block. fc->dropout->batchnorm->sigmoid
  fully_connected = layers.Dense(
      fc_units, 
      activation='relu', 
      kernel_initializer=initializer, 
      name="FC-" + str(fc_units) + "-a"
      )(x) 
  fully_connected = layers.Dropout(
      rate=rateval, 
      name="DO_FC1"
      )(fully_connected, training=trainingval_dropout)
  """
  fully_connected = layers.BatchNormalization(
      name="BN_FC1"
      )(fully_connected, training=trainingval)
  """
  # Output layer
  fully_connected = layers.Dense(
      1, 
      activation="sigmoid", 
      kernel_initializer=initializer, 
      name="Output"
      )(fully_connected)
  
  model = Model([input_2], fully_connected)
  
  return model 





def compile_model(model, prc_threshold, lr):
  astronet_epsilon = 1e-8
  
  model.compile(
      optimizer=keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=astronet_epsilon
        ),
      loss=keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), 
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.TruePositives(thresholds=prc_threshold,name='tp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'),
      ])
  return model
  #NEW_MODEL_PATH = "/home/s.fiscale/conda/Models/cnn_multiple_inputs/CNN/cnn_201_61_paddsame_1fc1024_dropout_batchnorm/cnn"
  #model.save(NEW_MODEL_PATH)
  




def build_model(trainingval=True, prc_threshold=0.5, finetuning=False, lr=0.001, rateval=0.2):
  print("Initializing...\n learning rate={}\n dropout probability rate={}\n\n".format(lr,rateval))
  # Initialize model's parameters
  initializer = initializer_keras()
  
  # Convolutional blocks
  input_2, output_2 = global_view_fiscale(initializer, trainingval, rateval)
  # Classification block
  model = fully_connected(initializer, input_2, output_2, trainingval, finetuning, rateval)
  
  return compile_model(model, prc_threshold, lr)





if __name__ == '__main__':
  trainingval = True      # NOTE. Set this value to False when using the model in testing mode
  my_model = build_model(trainingval)
  print(my_model.summary())
  """
  keras.utils.plot_model(
      my_model, 
      "/home/s.fiscale/altro/tensorflow_test/TESS/tey2022/20230526_cnn_architecture.png", 
      show_shapes=True
      )
  """
