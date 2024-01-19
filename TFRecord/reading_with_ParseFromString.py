import tensorflow as tf
import numpy as np
import os




"""
  Funzione che utilizzo per aprire un tfrecord e scomporlo nei singoli campi. Questa procedura agisce sugli nproc tfrecord che vengono generati in parallelo. Una volta scomposti, i campioni vengono re-distribuiti in 10 tfrecord aventi numero di campioni uniforme (vedi .../TESS/utils.py).
"""
def parse_from_string(FILENAME_PATTERN):
  # Inizializzazione
  tic_list = []
  toi_list = []
  disp_list = []
  gv_list = []
  # Crea oggetto TFRecordDataset
  raw_dataset = tf.data.TFRecordDataset(FILENAME_PATTERN)
  #print(raw_dataset)
  #n_examples = 0
  # Show first example only? raw_dataset.take(1)
  for raw_record in raw_dataset:
    try:
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      
      #1. Mostra tutto l'oggetto
      #print(example)
      
      #2. Mostra solo un attributo
      # 2.1 TIC
      tic_id = example.features.feature['tic'].int64_list.value[0]                      #type(tic_id) : int
      
      # 2.2 Disposition
      disp = example.features.feature['av_training_set'].int64_list.value[0]            #type(disp) : int
      
      # 2.3 Views
      global_view = np.array(example.features.feature['global_view'].float_list.value)  #type(global_view) : numpy.ndarray
      toi_id = example.features.feature['toi'].int64_list.value[0]                      #type(toi_id) : int
      #NOTE. The following line has been used to assign an artificial id to kepler tfrecords
      #toi_id = 0
      # Salva i dati
      tic_list.append(tic_id)
      toi_list.append(toi_id)
      disp_list.append(disp)
      gv_list.append(global_view)
      
      
      
    except:
      print("Errore tfrecord {}\n".format(FILENAME_PATTERN))
  return tic_list, toi_list, disp_list, gv_list #, lv_list 



"""
  Conta il numero di campioni PC,NPC in un dato tfrecord di nome FILENAME_PATTERN.
"""
def count_num_samples(FILENAME_PATTERN):
  raw_dataset = tf.data.TFRecordDataset(FILENAME_PATTERN)
  #print(raw_dataset)

  #n_examples = 0
  pc = 0        # planets
  npc = 0       # not planets
  err = 0
  
  for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    if example.features.feature['av_training_set'].int64_list.value[0] == 1:
      pc += 1
    elif example.features.feature['av_training_set'].int64_list.value[0] == 0:
      npc += 1
    else:
      err += 1
      print("Errore: av_training_set={}\n".format(example.features.feature['av_training_set'].int64_list.value[0]))

    # Conta numero di campioni del dataset
    #n_examples += 1

  return pc, npc, err #, n_examples




"""
  Conta il numero totale di campioni di ogni classe di tutti i tfrecord files.
  Input:
    - tfrecord_path: directory nella quale hai memorizzato i tfrecord;
    - nproc: numero di tfrecord files.
  Output:
   - Numero totale di campioni (#PC,#NPC) pari alla somma dei PC ed NPC di ogni tfrecord file.
"""
def count_total_num_samples(tfrecord_path):
  # Inizializzazione
  r = [0,0,0]

  for tfrec_i in os.listdir(tfrecord_path):
    try:
      filename = tfrecord_path + tfrec_i
      print(filename)
      # Count samples 
      pc, npc, err = count_num_samples(filename) 
      r[0] += pc
      r[1] += npc
      r[2] += err
    except:
      continue

  
  
  print("Planets: {}\nNot planets: {}\nErrors: {}\n".format(r[0], r[1], r[2]))
  return r[0], r[1]













if __name__ == '__main__': 
  print("Read the content of a tfrecord file")
