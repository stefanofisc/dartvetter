import numpy as np
import pandas as pd
import lightkurve as lk
from mpi4py import MPI
import lightkurve_utils
import new_preprocessing
import time
#import os
import sys
import argparse

from process_catalog import open_tce_csv, get_column_names
from init import author_info, catalog_name_list
from utils import salva_errore_tic, create_uniform_tfrecord

sys.path.insert(1, 'path/to/TFRecord')
from create_tfrecord_file import generate_tfrecord

LOCAL_PATH = 'path/to/your/working/directory'  


def getArgs(argv=None):
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--gv_length",
    type=int,
    required=True,
    help="Number of global view bins." 
    )

  parser.add_argument(
    "--lv_length",
    type=int,
    required=True,
    help="Number of local view bins." 
    )
  
  parser.add_argument(
    "--tfrecord_dir",
    type=str,
    required=True,
    help="Destination folder to store TFRecords." 
    )
  
  parser.add_argument(
    "--tce_csv_file",
    type=str,
    required=True,
    help="Threshold Crossing Events csv file." 
    )
  
  parser.add_argument(
    "--error_folder",
    type=str,
    required=True,
    help="Name of the folder in which processors will store their error files." 
    )
  
  parser.add_argument(
    "--detection_pipeline",
    type=str,
    required=True,
    help="Name of the pipeline that generated your input fits files, SPOC or QLP." 
    )
  
  parser.add_argument(
    "--catalog",
    type=str,
    required=True,
    help="Catalog name: tess,tey2022, exofop" # Specify the catalog name in order to deal with the different attributes names (e.g. tce_period, Period, period_days) 
    )

  parser.add_argument(
    "--path_to_output_views",
    type=str,
    required=False,
    help="Set the path of the folder in which you want to save the global views figures. Set to '' if you do not want to plot any figure"
    )
  
  return parser.parse_args(argv)




def num_rows_proc(nprocs, row_count):
  """
    Definisci numero di righe del file csv per ogni nodo.
    nloc: numero di righe assegnato al generico nodo
    nloc_last: numero di righe assegnato all'ultimo nodo. Gestisce il caso out of bounds
  """
  nloc = int(row_count/nprocs)
  resto = row_count % nprocs
  if resto != 0:
    nloc = nloc + 1
  
  diff = (nloc * nprocs) - row_count
  nloc_last = nloc - diff

  return nloc, nloc_last







def process_light_curve(lc, tce, GV_NBINS, catalog='tess', author='QLP'):
  """
    Preprocess a single light curve and generate one input vector: a global view. Today, June 22, 2023, I have added two methods that clean the global view flux data from residual NaNs and scale the flux in the range [-1,0], respectively.
    Input:
      - lc: the input LightCurve object consisting in the data collected in a single TESS observation year (i.e. the first year in which TESS observed the target star.
      - tce: the i-th row of the tce csv file
      - GV_NBINS: global view length (int)
      - catalog: the value of this string allows this method to retrieve the columns names of the input catalog
      - author: SPOC or QLP. If this value is set to 'SPOC', then the width of the bins is set to 2 minute. On the other hand, the size of the bins is set to 30 minute if author='QLP'.
    Output:
      - a LightCurve object consisting in the model's input view
  """
  tce_period, tce_epoch, tce_dur = get_column_names(catalog)[2:5]
  #Preprocess TIC
  # Data cleaning
  lc_clean = lc.remove_outliers(sigma = 3.0)
  # Detrending
  # 1. Create transit mask
  transit_mask = lc_clean.create_transit_mask(period=tce[tce_period], transit_time=tce[tce_epoch], duration=tce[tce_dur])
  # 2. Flatten 
  lc_flatten = new_preprocessing.my_lc_flattening(lc_clean, transit_mask, window_length=11, polyorder=3)
  # Phase-folding
  phase_folding_range = 0.8 # this value must be in (0,1]
  lc_phased = new_preprocessing.my_phase_folding(lc_flatten, tce[tce_epoch], tce[tce_period], phase_folding_range) 
  # Binning
  if author=='SPOC':
    time_bin_size_min = 2
  else:
    time_bin_size_min = 30
  lc_phased_binned = new_preprocessing.bin_lightcurve(lc_phased, time_bin_size_min)
  # Set the global view length to a fixed value, that is 201 (Yu et al. 2019)
  lc_global = new_preprocessing.set_global_view_length(lc_phased_binned, GV_NBINS)
  # Clean the global view from NaNs and normalize to 0 median and minimum transit depth -1 as in Shallue and Vanderburg (2018).
  lc_global_flux_clean = new_preprocessing.interpolate_nan(lc_global.flux.value)
  lc_global_flux_scaled = new_preprocessing.zero_median_fixed_depth(lc_global.flux.value)
  
  return lk.LightCurve(time = lc_global.time.value, flux = lc_global_flux_scaled)







"""
  Tey2023.
  Input:
    - TIC: TESS Input Catalog id. type:int
    - period: TCE period in days. type:float
    - author: name of the pipeline you must specify to properly read the fits files. This is because of the difference existing between SPOC fits files and QLP fits files. values accepted: {SPOC, QLP, EXOFOP_SPOC, EXOFOP_QLP, TEY2022, TEY2022_SPOC, TEY2022_QLP}. See lightkurve_utils() for further details.
    - not_valid_tic: filename in which to store information about any TIC for which no fits files are found. type:str.
    - multi_search: boolean variable determining whether to search fits files in different folders. In this case, author sould be a vector of values. Since this process could be quite time consuming, author can contain a maximum of 3 different values.
  Output:
    - LightCurve() object or
    - -1 in case of errors.
"""
def init_lightcurve(TIC, period, author, not_valid_tic, normalize_flux_data=False, fill_gaps_in_flux=False, open_sap_flux=False, split_by_year=False, multi_search=False):
    # Crea 1 oggetto LightCurve concatenando i fits file del TIC in analisi
    if multi_search:
      lc1 = lightkurve_utils.init_lightkurve_tess(TIC,author[0],normalize_flux_data,fill_gaps_in_flux,open_sap_flux,split_by_year)
      if lc1 == -1:
        # Se non trovi dati a 2 minuti, cerca nella cartella dei dati a 10-30minuti
        #NOTE. 20230515 blocco aggiunto per cercare dati Tey QLP nel caso in cui quelli SPOC non ci fossero
        lc1 = lightkurve_utils.init_lightkurve_tess(TIC,author[1],normalize_flux_data,fill_gaps_in_flux,open_sap_flux,split_by_year)
        if lc1 == -1:
          # Se non hai trovato dati nel MAST, cerca nei dati Tey 
            lc1 = lightkurve_utils.init_lightkurve_tess(TIC,author[2],normalize_flux_data,fill_gaps_in_flux,open_sap_flux,split_by_year)
    else:
      lc1 = lightkurve_utils.init_lightkurve_tess(TIC,author,normalize_flux_data,fill_gaps_in_flux,open_sap_flux,split_by_year)
    
    if lc1 == -1:
      # Non ci sono dati, anche se dovrebbero esserci. Qualcosa è andato storto
      salva_errore_tic(not_valid_tic, str(TIC)+"-"+str(period), 'Nessuna curva di luce')

    return lc1



"""
  Distributed processing of TESS Threshold-Crossing-Events.
  nprocs core work in parallel.
  This pipeline is able to process data from different TESS catalogs. 
  Data augmentation is applied on PCs by applying a horizontal reflection on the global view
  Both short cadence (2-min) and long cadence (10-min and 30-min) data can be processed.
  Output:
    nprocs tfrecord files are generated.
"""
def process_tce(tce_table, comm, nprocs, rank, name, patht, GV_NBINS, LV_NBINS, TFRECORD_DIR, ERROR_FOLDER, DETECTION_PIPELINE, catalog='tess', path_to_output_views=''):
    # Initialize TFRecord data structures
    gv = []       # array of global views
    lbl = []      # array of TCEs dispositions
    tic_list = [] # array of TIC id
    toi_list = [] # array of TOI id, or row id when TOI is not available
    nobservations = 0
    # Given the name of the catalog, retrieve the column names you need
    tce_tic, tce_disposition, tce_period = get_column_names(catalog)[0:3] 
    # Read TCE CSV file
    #NOTE:2023-05-12. The csv file is opened before the domain decomposition. tce_table = open_tce_csv(tce_csv_file,catalog)
    # Count rows
    row_count = tce_table.shape[0]
    #print("Numero di righe: {}\n".format(row_count))

    # Distributed domain decomposition
    if rank == 0:
      print("master")
      nloc, nloc_last = num_rows_proc(nprocs, row_count)
      print("nloc : {}\n",format(nloc))
      for j in range(1, nprocs):
        if j < nprocs:
          req = comm.isend(nloc,dest=j,tag=99)
          req.wait()
        if j == nprocs:
          req = comm.isend(nloc_last,dest=j,tag=99)
          req.wait()
    else:
      #print("slaves")
      req  = comm.irecv(source=0, tag=99)
      nloc = req.wait()
    #print("slaves num {}, nloc = {}\n".format(rank, nloc))
    START = rank*nloc
    END = START + nloc - 1
    if END > row_count:
      END = START + nloc - 2
    #print("proc {}: START = {}\tEND = {}\n".format(rank, START, END))
    # Se un core deve analizzare 1 solo TIC incrementa END per eseguire 1 iterazione nel for.
    if START == END:
      END += 1
    
    # Path del file dove ogni core memorizza i TIC per i quali è stato generato un errore
    not_valid_tic =  patht + ERROR_FOLDER + "/" +  str(rank) + "_errori.txt"
    # NOTE: ExoFOP Test set. Load the list of overlapped TCEs between the following two catalogs: Tey2022 and ExoFOP
    # Here the code

    # Open tce csv file
    db = pd.DataFrame(tce_table)
    for i in range(START, END+1):
      try:
        tce = db.iloc[i]
        TIC = int(tce[tce_tic])
        #NOTE. catalog_name_list[2]=exofop, catalog_name_list[3]=exofop_tt9 magliano et al.
        if catalog == catalog_name_list[2] or catalog == catalog_name_list[3]:
          toi_id = int(str(tce['TOI_1']).replace('.',''))
        #NOTE. Cacciapuoti,Yu,Tey. Given the lack of a toi id in the catalogs provided by the underlined authors, 
        # I decided to set the csv file rowid as toi id
        else:
          toi_id = int(i)
       
        # Determina in quale cartella cercare. 
        #if catalog == catalog_name_list[2] or catalog == catalog_name_list[3]:  
        # exofop / exofop_tt9 / exofop_tt9_cacciapuoti
        if catalog_name_list[2] in catalog:
          # ExoFOP
          if DETECTION_PIPELINE == author_info[0][0]:
            author = author_info[0][1]  # EXOFOP_SPOC
          else:
            author = author_info[1][1]  # EXOFOP_QLP
        elif catalog == catalog_name_list[1]:
          # Tey 2023. catalog = tey2022. author = TEY2022_QLP
          author = author_info[1][2]
        else:
          # catalog = catalog_name_list[0] = tess = liang yu data
          # Yu 2019 QLP
          author = author_info[1][0]
        # For each TIC-TCE, create a single LightCurve object by stitching together data collected by TESS in the first 
        # year the TIC has been observed
        lc1 = init_lightcurve(
            TIC, 
            tce[tce_period], 
            author, 
            not_valid_tic, 
            normalize_flux_data=True, 
            split_by_year=False,
            open_sap_flux=True
            )
          #open_sap_flux=True when using SPOC data
        if lc1 == -1:
          continue
        else:
          # Ci sono i dati, costruisci singolo campione di input
          # Process the LightCurve object in order to generate a binned phase-folded representation
          lc_global = process_light_curve(lc1, tce, GV_NBINS, catalog, author=DETECTION_PIPELINE)
          
          # For each feature, store tfrecord data into different list
          tic_list.append(TIC)                  # lista di int;
          toi_list.append(toi_id)               # utilizzo il rowid del catalogo come identificativo del tce;
          gv.append(lc_global.flux.value)       # matrice dove ogni riga ha una global view;
          lbl.append(tce[tce_disposition])      # lista di int.
          nobservations += 1
          # Data augmentation. Horizontal reflection on the planet candidates global views 
          """
          if catalog != catalog_name_list[2]:
            if tce[tce_disposition] == 1:
              tic_list.append(TIC)
              toi_list.append(toi_id)
              gv.append( np.flip(lc_global.flux.value) )
              lbl.append(tce[tce_disposition])
              nobservations += 1
          """
      except Exception as e:
        print("Eccezione dal proc {} sul nodo {}:".format(rank,name))
        print(i,e)
        salva_errore_tic(not_valid_tic, str(TIC), str(e)) #NOTE Rimosso +toi_id perchè dava problemi 20230512
      
    # endfor
    # NOTE: Questa parte è distribuita ed eseguita dagli n processi
    print("Total number of samples: {}\n".format(nobservations))
    
    # Terminato il pre-processing di tutte le curve di luce, genera TFRecord
    # nome del file: train-proc_id.tfrecord
    tfrecord_name = LOCAL_PATH + TFRECORD_DIR + '/train-' + str(rank) + '.tfrecord'
    generate_tfrecord(tfrecord_name, gv, lbl, nobservations, tic_list, toi_list) 

if __name__ == '__main__':
    #Set input parameters
    args = getArgs(None)
    GV_NBINS = args.gv_length
    LV_NBINS = args.lv_length
    TFRECORD_DIR = args.tfrecord_dir
    TCE_CSV_FILE = args.tce_csv_file
    ERROR_FOLDER = args.error_folder
    DETECTION_PIPELINE = args.detection_pipeline
    CATALOG = args.catalog
    PATH_TO_OUTPUT_VIEWS = args.path_to_output_views
    #Open TCEs csv file
    tce_table = open_tce_csv(TCE_CSV_FILE, CATALOG)
    #Check for input error
    if not isinstance(GV_NBINS, int):
      print("GV_NBINS is not int, but type: {}\n".format(type(GV_NBINS)))
      GV_NBINS = int(GV_NBINS)
      LV_NBINS = int(LV_NBINS)
    # Starting distributed context
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    name = MPI.Get_processor_name()
    comm.Barrier()

    start_time = MPI.Wtime()
    process_tce(tce_table, comm, nprocs, rank, name, LOCAL_PATH, GV_NBINS, LV_NBINS, TFRECORD_DIR, ERROR_FOLDER, DETECTION_PIPELINE, CATALOG, PATH_TO_OUTPUT_VIEWS)
    end_time =  MPI.Wtime()
    
    # Il processo 0 salva il tempo di esecuzione
    if rank == 0:
      print("Exit")
      #tfrecord_to_create = LOCAL_PATH + 'tfrecord_tey2022_horizreflpc_uniform_20230525/'
      #tfrecord_to_update = TFRECORD_DIR
      #create_uniform_tfrecord(tfrecord_to_update, tfrecord_to_create, 128, 10, 5204, 16460)
      

