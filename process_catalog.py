import pandas as pd
import numpy as np
from init import catalog_name_list

def get_column_names(catalog):
  if catalog == catalog_name_list[0]:        # Catalogo di Liang Yu 2019. Astronet v2
    tce_tic     = 'tic_id'
    tce_period  = 'Period'
    tce_disposition = 'Disposition'
    tce_epoch   = 'Epoc'
    tce_dur     = 'Duration'
  elif catalog == catalog_name_list[1]:      # Catalogo Tey et al 2022. Astronet v3
    tce_tic     = 'TIC ID'
    tce_period  = 'Period'
    tce_disposition = 'Consensus Label'
    tce_epoch   = 'Epoch'
    tce_dur     = 'Duration'
  elif catalog == catalog_name_list[2]:       # ExoFOP
    tce_tic     = 'TIC ID'
    tce_period  = 'Period (days)'
    tce_disposition = 'TFOPWG Disposition'
    tce_epoch   = 'Epoch (BJD)'
    tce_dur     = 'Duration (hours)'
  elif catalog == catalog_name_list[3]:       # Catalogo TESS Triple 9 v2 (Magliano et al. 2023)
    tce_tic     = 'TIC ID_1'
    tce_period  = 'Period [Days]'
    tce_disposition = 'Paper disp'
    tce_epoch   = 'Epoch (BJD)'
    tce_dur     = 'Duration [Hours]'
  else:                                       # Catalogo TESS Triple 9 v1 (Cacciapuoti et al. 2022)
    tce_tic         = 'TIC ID'
    tce_epoch       = 'Epoch [BJD]'
    tce_period      = 'Period [Days]'
    tce_dur         = 'Duration [Hours]'
    tce_disposition = 'Paper disp (LC)'

  return tce_tic, tce_disposition, tce_period, tce_epoch, tce_dur





def process_tey_catalog(tce_table_original):
  """
    Given the TCEs catalog provided by Tey et al. 2022, apply the following operations: 
      (i) remove rows having Notes=Duplicated in Y3 because they store the transit-like features of events observed both in Y1 and Y3 of TESS mission; 
      (ii) filter out the rows with Consensus Label different from E,B or J because they are affected by a degree of uncertainty; 
      (iii) binarize the remaining labels.
    NOTE. When using this data set as training set, decomment the lines commented as "#Training". When used for testing, decomment lines commented as "#Test"
  """
  #Count the number of NaN for each column
  #print(tce_table_original.isna().sum())
  # Remove the row with TIC ID = nan
  tce_table_original.dropna(subset=['TIC ID'],inplace=True)
  #Conversion needed because Topcat altered the types of 'TIC ID,Period,Duration,Epoch'
  tce_table = tce_table_original.astype({
    'TIC ID':'int32',
    'Period':'float64',
    'Duration':'float64',
    'Epoch':'float64'
    },
    copy=True
    )
  # Remove rows with S, N or NaN value in Consensus Label column
  _LABEL_COLUMN         = 'Consensus Label'
  #_ALLOWED_LABELS       = {'E','B','J'}        #Training
  _ALLOWED_LABELS       = {'B'}                 #Test
  allowed_tces          = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table             = tce_table[allowed_tces]
  # Remove rows with 'Duplicated in Y3' value in Notes column
  _NOTES_COLUMN         = 'Notes'
  _IS_DUPLICATED        = {'Duplicated in Y3'}
  allowed_tces_1        = tce_table[_NOTES_COLUMN].apply(lambda l: l not in _IS_DUPLICATED)
  tce_table             = tce_table[allowed_tces_1]
  # Count the number of samples for each of the remaining class
  print("Labels:[{}]\n".format(tce_table[_LABEL_COLUMN].unique()))
  #tot_e = tce_table[_LABEL_COLUMN].value_counts()['E'] #Training
  tot_b = tce_table[_LABEL_COLUMN].value_counts()['B']
  #tot_j = tce_table[_LABEL_COLUMN].value_counts()['J'] #Training
  #print("E={}; B={}; J={}\n".format(tot_e,tot_b,tot_j))
  print("B={}\n".format(tot_b))
  # Binarize labels
  print("*** Labels binarization ***\n") 
  #bin_labels    = {_LABEL_COLUMN:{'E':1,'B':0,'J':0}}  #Training
  bin_labels    = {_LABEL_COLUMN:{'B':0}}               #Test
  tce_table     = tce_table.replace(bin_labels)
  #tot_pc        = tce_table[_LABEL_COLUMN].value_counts()[1] #Training
  tot_npc       = tce_table[_LABEL_COLUMN].value_counts()[0]
  #print("PC={}; NPC={}\n".format(tot_pc,tot_npc))
  print("NPC={}\n".format(tot_npc))
  
  return tce_table





def process_yu_catalog(tce_table):
  """
    Given the TCEs catalog provided by Yu et al. 2019, apply the following operations: 
      (i) remove rows with missing values in period, epoch or duration;
      (ii) filter out the rows without Disposition (iii) binarize the remaining tces labels.
  """
  #print(tce_table.describe())
  # Remove rows with S, N or NaN value in Consensus Label column
  _LABEL_COLUMN         = 'Disposition'
  _ALLOWED_LABELS       = {'PC','EB','V','IS','J'}
  allowed_tces          = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table             = tce_table[allowed_tces]
  # Remove rows with missing value in Period, Epoc and/or Duration
  #print("Rows before removing missing values from period,epoch,duration:{}\n".format(len(tce_table)))
  tce_table.dropna(subset=['Period','Epoc','Duration'],inplace=True)
  #print("Rows after removing missing values from period,epoch,duration:{}\n".format(len(tce_table)))
  # Convert Duration (hours) to Duration (days)
  tce_table['Duration'] /= 24
  # Count the number of samples for each of the remaining class
  print("Labels:[{}]\n".format(tce_table[_LABEL_COLUMN].unique()))
  tot_pc = tce_table[_LABEL_COLUMN].value_counts()['PC']
  tot_eb = tce_table[_LABEL_COLUMN].value_counts()['EB']
  tot_v = tce_table[_LABEL_COLUMN].value_counts()['V']
  tot_is = tce_table[_LABEL_COLUMN].value_counts()['IS']
  tot_j = tce_table[_LABEL_COLUMN].value_counts()['J']
  #print("PC={}; EB={}; V={}; IS={}; J={}\n".format(tot_pc,tot_eb,tot_v,tot_is,tot_j))
  # Binarize labels
  print("*** Labels binarization ***\n") 
  bin_labels    = {_LABEL_COLUMN:{'PC':1,'EB':0,'V':0,'IS':0,'J':0}}
  tce_table     = tce_table.replace(bin_labels)
  tot_pc        = tce_table[_LABEL_COLUMN].value_counts()[1]
  tot_npc       = tce_table[_LABEL_COLUMN].value_counts()[0]
  print("PC={}; NPC={}\n".format(tot_pc,tot_npc))
  
  return tce_table




def process_exofop_catalog(tce_table):
  # Remove rows with missing value in Period, Epoc and/or Duration
  # This catalog does not have NaN in the columns reported below. See model_assessment in training_tey_yu folder
  #tce_table.dropna(subset=['Period (days)','Epoch (BJD)','Duration (hours)'],inplace=True)
  # data conversion not needed. Period, Epoch and Duration are already float objects
  # Convert Epoch from BJD to BTJD. Convert Duration from hours to days
  #tce_table['Epoch (BJD)'] -= 2457000.0
  #tce_table['Duration (hours)'] /= 24
  #Preserve rows with allowed labels
  _LABEL_COLUMN         = 'TFOPWG Disposition'
  _ALLOWED_LABELS       = {'KP','CP','PC','FA','APC','FP'}
  allowed_tces          = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table             = tce_table[allowed_tces]
  # Count the number of samples for each of the remaining class
  print("Labels:[{}]\n".format(tce_table[_LABEL_COLUMN].unique()))
  tot_pc = tce_table[_LABEL_COLUMN].value_counts()['PC']
  tot_kp = tce_table[_LABEL_COLUMN].value_counts()['KP']
  tot_cp = tce_table[_LABEL_COLUMN].value_counts()['CP']
  tot_apc = tce_table[_LABEL_COLUMN].value_counts()['APC']
  tot_fa = tce_table[_LABEL_COLUMN].value_counts()['FA']
  tot_fp = tce_table[_LABEL_COLUMN].value_counts()['FP']
  print("PC:{}; KP:{}; CP:{}; APC:{}; FA:{}; FP:{}\n".format(tot_pc,tot_kp,tot_cp,tot_apc,tot_fa,tot_fp))
  # Binarize labels
  print("*** Labels binarization ***\n") 
  bin_labels    = {_LABEL_COLUMN:{'KP':1,'CP':1,'PC':1,'FA':0,'APC':0,'FP':0}}
  tce_table     = tce_table.replace(bin_labels)
  tot_pc        = tce_table[_LABEL_COLUMN].value_counts()[1]
  tot_npc       = tce_table[_LABEL_COLUMN].value_counts()[0]
  print("Planets:{}; Not-planets:{}\n".format(tot_pc,tot_npc))

  return tce_table




def process_tt9_catalog(tce_table,binarize_labels=False):
  # Convert epoch in BTJD and duration in days
  #tce_table['Epoch (BJD)'] -= 2457000.0
  tce_table['Duration [Hours]'] /= 24
  if binarize_labels:
    _LABEL_COLUMN = 'Paper disp'
    label_list = tce_table[_LABEL_COLUMN].unique()
    print("Labels:[{}]\n".format(label_list))  
    tot_pc = tce_table[_LABEL_COLUMN].value_counts()['PC']
    tot_pfp = tce_table[_LABEL_COLUMN].value_counts()['pFP']
    tot_fp = tce_table[_LABEL_COLUMN].value_counts()['FP']
    print("PC:{}; pFP:{}; FP:{}\n".format(tot_pc, tot_pfp, tot_fp))
    # Binarize labels
    print("*** Labels binarization ***\n") 
    bin_labels    = {_LABEL_COLUMN:{'PC':1,'FP':0,'pFP':0}}
    tce_table     = tce_table.replace(bin_labels)
    tot_pc        = tce_table[_LABEL_COLUMN].value_counts()[1]
    tot_npc       = tce_table[_LABEL_COLUMN].value_counts()[0]
    print("Planets:{}; Not-planets:{}\n".format(tot_pc,tot_npc))

  return tce_table




def process_tt9_catalog_cacciapuoti(tce_table,binarize_labels=False):
  # define column names
  _TIC_COLUMN         = 'TIC ID'
  epoch               = 'Epoch [BJD]'
  period              = 'Period [Days]'
  duration            = 'Duration [Hours]'
  _LABEL_COLUMN       = 'Paper disp (LC)'
  _ALLOWED_LABELS     = {'PC','pFP','FP'}
  overlapping_liangyu = 'path/to/list/overlapped/tic/cacciapuotietal2022/yuetal2019/file.txt'
  # remove inconsistent rows
  tce_table.dropna(subset=[period,epoch,duration,_LABEL_COLUMN],inplace=True)
  # remove not allowed labels (e.g. 'multi')
  allowed_tces          = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table             = tce_table[allowed_tces]
  # Convert epoch in BTJD, duration in days, TIC ID in int
  pd.options.mode.chained_assignment = None  # default='warn'
  tce_table[epoch] -= 2457000.0
  tce_table[duration] /= 24
  tce_table[_TIC_COLUMN] = tce_table[_TIC_COLUMN].astype(int)

  # Remove tic overlapped with Liang Yu
  tic_list = list(np.loadtxt(overlapping_liangyu))
  # convert tic from float to int with list comprehensions
  tic_list_int = [int(x) for x in tic_list]
  #tic_list_str = [str(x) for x in tic_list_int]
  print("Converted {} TIC ID\n".format(len(tic_list_int)))
  # define list of allowed tic
  _ALLOWED_TIC          = set(tic_list_int)
  allowed_tic           = tce_table[_TIC_COLUMN].apply(lambda l: l not in _ALLOWED_TIC)
  tce_table             = tce_table[allowed_tic]

  if binarize_labels:
    #PC=779, pFP=146, FP=144, multi=1, nan=1
    label_list = tce_table[_LABEL_COLUMN].unique()
    print("Labels:[{}]\n".format(label_list))
    tot_pc = tce_table[_LABEL_COLUMN].value_counts()['PC']
    tot_pfp = tce_table[_LABEL_COLUMN].value_counts()['pFP']
    tot_fp = tce_table[_LABEL_COLUMN].value_counts()['FP']
    print(tot_pc,tot_pfp,tot_fp)
    # Binarize labels
    print("*** Labels binarization ***\n") 
    bin_labels    = {_LABEL_COLUMN:{'PC':1,'FP':0,'pFP':0}}
    tce_table     = tce_table.replace(bin_labels)
    tot_pc        = tce_table[_LABEL_COLUMN].value_counts()[1]
    tot_npc       = tce_table[_LABEL_COLUMN].value_counts()[0]
    print("Planets:{}; Not-planets:{}\n".format(tot_pc,tot_npc))
  
  return tce_table





# Open CSV file
def open_tce_csv(TCE_CSV_FILE,catalog='tess',binarize_labels=False):
  # Read CSV file of ExoFOP TOIs.
  tce_table = pd.read_csv(TCE_CSV_FILE, comment="#", delimiter=',')    #index_col="rowid"
  if catalog != catalog_name_list[1]:
    if catalog == catalog_name_list[2]:
      #print("Apro catalogo già processato e binarizzato") #duration is expressed in days as well as the period. TFOPWG contains binary values
      tce_table = process_exofop_catalog(tce_table)
      #tce_duration = "Duration (hours)" #to be converted in days
      #tce_table["Epoch (BJD)"] -= 2457000.0   
    elif catalog == catalog_name_list[3]:
      tce_table = process_tt9_catalog(tce_table,binarize_labels)  #metti false per SPOC e True per QLP
    elif catalog == catalog_name_list[4]:
      print("catalogo già processato")
      #tce_table = process_tt9_catalog_cacciapuoti(tce_table, binarize_labels) 
    else:
      # for Yu 2019 catalog
      tce_table = process_yu_catalog(tce_table)
  else:
    #Tey et al 2023 catalog
    tce_table = process_tey_catalog(tce_table)

    print("Read {} filtered rows from {}\n".format(len(tce_table),catalog))

  return tce_table


if __name__ == '__main__':
  # Do something
  print("Main")
