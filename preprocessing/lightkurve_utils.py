pimport os
import glob
import re

import numpy as np
import lightkurve as lk

from astropy.io import fits

from init import author_info, tess_observation_sectors


path_to_fits_files = [
    { author_info[0][0] : 'path/to/yuetal2019/short-cadence/fits/files' },
    { author_info[0][1] : 'path/to/exofop/short-cadence/fits/files' },
    { author_info[0][2] : 'path/to/teyetal2023/short-cadence/fits/files' },
    { author_info[1][0] : 'path/to/yuetal2019/long-cadence/fits/files' },
    { author_info[1][1] : 'path/to/exofop/long-cadence/fits/files' },
    { author_info[1][2] : 'path/to/teyetal2023/long-cadence/fits/files'},
    { author_info[2][0] : 'path/to/teyetal2023/long-cadence-vizierarchive/fits/files'}
    ] 





"""
  Read the content of the Light Curve Binary Table Extension Header
"""
def open_fits_file_lc_data(fits_file):
  with fits.open(fits_file, mode='readonly') as hdulist:
    data = hdulist[1].data
    print(repr(data))



"""
  Read time and flux from TESS light curves (SPOC)
  Input:
    - fits_file: fits filename. type:str.
    - open_sap_flux: set this variable to true if you want to access to SPOC SAP_FLUX data. type:bool.
  Output:
    - numpy.ndarray of time and flux data
"""
def open_fits_file_lc(fits_file,open_sap_flux=False):
  with fits.open(fits_file, mode='readonly') as hdulist:
    tess_bjds = hdulist[1].data['TIME']
    if open_sap_flux:
      sap_flux = hdulist[1].data['SAP_FLUX']
      return tess_bjds, sap_flux
    else:
      pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
    
  return tess_bjds, pdcsap_fluxes



"""
  Read time and flux from TESS light curves fits files (QLP)
"""
def open_fits_file_lc_qlp(fits_file):
  with fits.open(fits_file, mode='readonly') as hdulist:
    tess_bjds = hdulist[1].data['TIME']
    #kspsap_fluxes = hdulist[1].data['KSPSAP_FLUX']     #Flux data detrended by using a Kepler spline (see Huang et al. 2020a)
    sap_fluxes = hdulist[1].data['SAP_FLUX']
  
  return tess_bjds, sap_fluxes
  #return tess_bjds, kspsap_fluxes



"""
  Read time and flux from TESS light curves fits files (Tey2022)
  Input
    - return_quality_flags: set this value to true if you want access to Data Quality Flags (DQF). For further details about DQF see TESS Science Data Products Description Document, EXP-TESS-ARC-ICD-0014 Rev D, Sec.9. 
"""
def open_fits_file_lc_tey2022(fits_file,return_quality_flags=False):
  with fits.open(fits_file, mode='readonly') as hdulist:
    tess_bjds = hdulist[1].data['TIME']
    sap_fluxes = hdulist[1].data['SAP_FLUX']
    if return_quality_flags:
      quality = hdulist[1].data['QUALITY']
      return tess_bjds, sap_fluxes, quality

  return tess_bjds, sap_fluxes





""" Remove cadences where time or flux have a NaN.
  Crea una maschera sia per time che per flux, all'interno della quale
  'True' indica la presenza di un numero valido, 'False' viceversa.
  Si effettua un and logico tra le due maschere e si usa la maschera risultante
  per rimuovere i NaN dai due vettori.
  
  Input:
   - time, 1d numpy array of time values
   - flux, 1d numpy array of flux values
  Output:
   - time, 1d numpy array without NaNs
   - flux, 1d numpy array without NaNs
"""
def remove_nan(time, flux):
  time_mask = ~np.isnan(time) #False where there is a nan
  flux_mask = ~np.isnan(flux)
  
  nan_mask = np.logical_and(time_mask, flux_mask)
  
  time = time[nan_mask]
  flux = flux[nan_mask]
  
  return time, flux





"""
  Prendi la lista di file .fits relativa al TIC 
  Genera curva di luce
  Input:
    - filename_list: lista di fits file relativi
    - author: SPOC, QLP: Indicano i dati di Liang Yu 
    - author: EXOFOP_SPOC, EXOFOP_QLP: Indicano i dati di ExoFOP
    - author: TEY2022: Indica i dati forniti da Tey et al 2022
    - normalize_flux_data: if you are working with multi-sector data, you should normalize flux data of different sectors. If this variable is set to True, the LightCurve.fill_gaps() method will be called in order to fill existing gaps between flux data of different sectors.
    - fill_gaps_in_flux: LightCurve.fill_gaps() method that fills gaps in time with white Gaussian noise.
    - open_sap_flux: SPOC data only. Boolean variable that allows you to select between pdcsap and sap flux data.
  Output:
    curva di luce concatenata dalla quale sono stati rimossi NaN ed outliers
"""
def append_light_curve(filename_list, author, normalize_flux_data=False, fill_gaps_in_flux=False, open_sap_flux=True):
  # Apri fits file e prendi vettori di tempo e flusso. 
  # Rimuovi tutti i nan
  # Inizializza l'oggetto di tipo LightCurve
  lc_list = []
  for i in range(len(filename_list)):
    if author in author_info[0]:        # open spoc fits file
      time, flux = open_fits_file_lc(filename_list[i], open_sap_flux)
    elif author in author_info[1]:      # open qlp fits file
      time, flux = open_fits_file_lc_qlp(filename_list[i])
    elif author in author_info[2]:      # open tey fits file 
      time, flux, quality = open_fits_file_lc_tey2022(filename_list[i])
    else:
      print("In lightkurve_utils, append_light_curve(), Error for {}\nauthor = {}".format(filename_list[i],author))
      return -1
    time, flux = remove_nan(time, flux)
    
    # Normalizza i dati di flusso
    if normalize_flux_data:
      flux_norm=flux/np.median(flux)
      flux = flux_norm

    lc_list.append(lk.LightCurve(time=time, flux=flux))
  # Ottieni 1 curva di luce da tutti i settori 
  lc = lc_list[0].append(lc_list[1:])
  if fill_gaps_in_flux:
    # Fill gaps in flux data
    lc = lc.fill_gaps()
  
  return lc








"""
  Find fits files for a given TIC
  Input:
    - TIC:           Tess Input Catalog star id. type:int.
    - author:        SPOC,QLP,EXOFOP_SPOC,EXOFOP_QLP. Select the folder in which you want to find the fits files relative to the input TIC.
    - normalize_flux_data: if you are working with multi-sector data, you should normalize flux data of different sectors. If this variable is set to True, the LightCurve.fill_gaps() method will be called in order to fill existing gaps between flux data of different sectors.
    - fill_gaps_in_flux: LightCurve.fill_gaps() method that fills gaps in time with white Gaussian noise.
    - open_sap_flux: SPOC data only. Boolean variable that allows you to select between pdcsap and sap flux data.
    - split_by_year: set to True if you want to create two different LightCurve object (e.g. a light curve for year 1 data and a light curve for year 3 data)
"""
def init_lightkurve_tess(TIC, author='SPOC', normalize_flux_data=False, fill_gaps_in_flux=False, open_sap_flux=False, split_by_year=False):
  mask = '-s(.+?)-'
  if author == author_info[0][0]:
    # Liang Yu 2-min data
    path = path_to_fits_files[0][author]

  elif author == author_info[0][1]:
    # ExoFOP 2-min data
    path = path_to_fits_files[1][author]

  elif author == author_info[0][2]:
    # Tey et al 2023 SPOC data downloaded from MAST
    path = path_to_fits_files[2][author]

  elif author == author_info[1][0]:
    # Liang Yu 10-min 30-min data
    mask = '_s(.+?)-'
    path = path_to_fits_files[3][author]

  elif author == author_info[1][1]:
    # ExoFOP 10/30-min data
    mask = '_s(.+?)-'
    path = path_to_fits_files[4][author]

  elif author == author_info[1][2]:
    # Tey et al 2023 QLP data downloaded from MAST
    mask = '_s(.+?)-'
    path = path_to_fits_files[5][author]
  
  else:
    print("Invalid author for the mission. Should be one in [SPOC,QLP,EXOFOP_SPOC,EXOFOP_QLP]\n")
    exit(0)
  
  # Define the lists containing the fits filenames by TESS years
  filename_list_1 = [] 
  filename_list_2 = []
  filename_list_3 = [] 
  filename_list_4 = []
  filename_list_5 = []
  filename_matrix = [ filename_list_1, filename_list_2, filename_list_3, filename_list_4, filename_list_5 ]

  found = ""
  n = 0
  s = -1
  # Convert tce['TIC'] into string
  TIC = str(TIC)
  
  # Set sectors variables
  for filename in glob.glob(os.path.join(path, '*.fits')):
    if TIC in filename:
      # Search for the substring *SPOC* or *QLP* in the variable 'author'
      # This allows you to split by year SPOC or QLP data
      if author_info[0][0] in author or author_info[1][0] in author:
        if split_by_year:
          # Extract sector substring when processing SPOC or QLP data
          m = re.search(mask,filename)
          if m:
            found = m.group(1)
            n = len(found)                  # Should be n=2
            s = found[n-2] + found[n-1]     # TESS Sector, type:str
            # Split the TIC-TCE fits files by year
            for idx in range(len(tess_observation_sectors)):
              if int(s) in tess_observation_sectors[idx]:
                filename_matrix[idx].append(filename)
          else:
            print("TIC: {} - In lightkurve_utils, line 131, error\n".format(TIC))
        else:
          #I don't need to extract the sector from Tey data
          filename_matrix[0].append(filename)
  # Tey et al. authors fits files: for those TIC having two fits files, discard one of the two because they contain the same data
  # NOTE. Decommenta se usi dati SAP_FLUX Tey 
  if author == author_info[2][0]:
    if len(filename_list_1) > 1:
      filename_list_1 = filename_list_1[0:len(filename_list_1)-1]
  
  lc_1 = -1

  # This block of code will never be executed by TEY2022 because they provided a single fits file for each TIC
  if split_by_year:
    for f in filename_matrix:
      if len(f) > 0:
        lc_1 = append_light_curve(f, author, normalize_flux_data, fill_gaps_in_flux, open_sap_flux)
        return lc_1
  else:
    if filename_matrix[0]:
      lc_1 = append_light_curve(filename_matrix[0], author, normalize_flux_data, fill_gaps_in_flux, open_sap_flux)
  
  return lc_1
