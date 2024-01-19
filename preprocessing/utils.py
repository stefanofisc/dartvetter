import sys
from new_preprocessing import interpolate_nan, scale_in_a_b

sys.path.insert(1, 'path/to/TFRecord/')
from reading_with_ParseFromString import parse_from_string


def salva_errore_tic(filename, TIC, errore):
  with open(filename, 'a') as f:
    f.write(str(TIC) + ", " + errore + "\n")
    f.close()


def save_tic_in_file(filename, TIC):
  with open(filename, 'a') as f:
    f.write(str(TIC) + '\n')
    f.close()


def save_prediction_planetpatrol(filename, TIC, label, prediction):
  sep = "; "
  with open(filename, 'a') as f:
    f.write(str(TIC) + sep + str(label) + sep + str(prediction) + "\n")
    f.close()



def create_uniform_tfrecord(tfrecord_path, tfrecord_dir, nproc, n_tfrecord, dim_pc, dim_npc, clean_and_scale=False):
  """
    Re-distribuisci i campioni dei tfrecord precedentemente creati affinché si ottengano N tfrecord con un numero di campioni delle due classi uniforme.
    Input:
      - tfrecord_path: directory dove sono memorizzati i tfrecord;
      - tfrecord_dir: directory dove vuoi salvare i nuovi tfrecord;
      - nproc: numero dei tfrecord files da processare;
      - n_tfrecord: numero dei tfrecord files da creare;
      - dim_pc: numero di pc del dataset;
      - dim_npc: numero dei npc del dataset.
  """
  os.mkdir(tfrecord_dir)
  # Initialization
  pc_disp = []
  pc_tic = []
  pc_toi = []
  pc_gv = []
  npc_disp = []
  npc_tic = []
  npc_toi = []
  npc_gv = []
  # Data structures storing the i-th tfrecord data
  tic_list = []
  toi_list = []
  disp_list = []
  gv_list = []
  # Open the $nproc TFRecord
  for i in range(nproc):
    try:
      tfrecord_i = tfrecord_path + "train-" + str(i) + ".tfrecord"
      # Ottieni i dati memorizzati nell'i-simo tfrecord
      tic_list, toi_list, disp_list, gv_list = parse_from_string(tfrecord_i)
      # Scorri i campioni e mantieni indice
      j = 0
      for sample in disp_list:
        #NOTE. 2023-06-22. Remove any residual NaN from flux values and interpolate over these values. Scale data in the range [a,b]=[-1,0]
        if clean_and_scale:
          # Non servirà farlo se lo fai già nella pipeline di pre-processing
          global_view_clean = interpolate_nan(gv_list[j])
          global_view_scaled = scale_in_a_b(global_view_clean, -1, 0)
        else:
          global_view_scaled = gv_list[j]
        if sample == 1:
          # Planet
          pc_disp.append(sample)
          pc_tic.append(tic_list[j])
          pc_toi.append(toi_list[j])
          pc_gv.append(global_view_scaled)
          # Data augmentation with horizontal reflection
          """
          pc_disp.append(sample)
          pc_tic.append(tic_list[j])
          pc_toi.append(toi_list[j])
          pc_gv.append(np.flip(global_view_scaled))
          """
        else:
          # Not planet
          npc_disp.append(sample)
          npc_tic.append(tic_list[j])
          npc_toi.append(toi_list[j])
          npc_gv.append(global_view_scaled)
        j += 1 
    except:
      print("Errore file {} campione numero {}\n".format(tfrecord_i),j)
      continue
  print("Ci sono {} pianeti e {} non pianeti\n".format(len(pc_disp),len(npc_disp)))
  
  # Data augmentation only
  #dim_pc = len(pc_disp)
  # Inizializzazione indici
  pc_offset = dim_pc//n_tfrecord #parte intera della divisione
  pc_start = 0
  pc_end = pc_offset-1
  npc_offset = dim_npc//n_tfrecord
  npc_start = 0
  npc_end = npc_offset-1
  # Calcola eventuali resti
  pc_resto = dim_pc%n_tfrecord
  npc_resto = dim_npc%n_tfrecord

  for k in range(n_tfrecord):
    tfrecord_name = tfrecord_dir + 'train-' + str(k) + '.tfrecord'
    
    if k == n_tfrecord-1:
      pc_end = pc_end + pc_resto
      npc_end = npc_end + npc_resto
    
    generate_tfrecord(tfrecord_name,
        pc_gv[pc_start:pc_end+1] + npc_gv[npc_start:npc_end+1],
        pc_disp[pc_start:pc_end+1] + npc_disp[npc_start:npc_end+1],
        len( pc_gv[pc_start:pc_end+1] + npc_gv[npc_start:npc_end+1] ),
        pc_tic[pc_start:pc_end+1] + npc_tic[npc_start:npc_end+1],
        pc_toi[pc_start:pc_end+1] + npc_toi[npc_start:npc_end+1]
        )
  
    pc_start = pc_end + 1
    pc_end = pc_end + pc_offset
    npc_start = npc_end + 1
    npc_end = npc_end + npc_offset
  
  print("Procedure completed")
