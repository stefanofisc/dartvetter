import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
from scipy.signal import savgol_filter

""" 
  New pre-processing methods
  
  # Main. For each TCE
  ### 1. Remove NaN from (TIME, SAP_FLUX) data
  ### 2. Concatenate data from the first year in which the target star has been observed
  ### 3. The Lightkurve stitch() method normalize each light curve by default. This means that you have to normalize 
        light curve data from each sector 
  ### 4. Remove outliers
  ### 5. Create the transit mask and flatten the light curve
  ### 6. Fold the light curve over a given period and epoch
  ### 7. Bin the phase folded signal
  ### 8. Fix the length of the binned phase folded signal
"""

def bin_lightcurve(lc_phased, time_bin_size_min=30, select_bin_size=False, bins=201):
  """
  Bin the phase folded data to make the transit more obvious.
  Input:
    - lc_phased: LightCurve object. The phase folded light curve.
    - time_bin_size_min: the width of the bins (in minutes). The higher the value, the smoother the binned signal will be. Standard values for QLP data
                         should be 10 or 30 min, while this value should be 2 min for SPOC data.
  Output:
    - A LightCurve object consisting in the phase folded and binned signal. The length of the light curve depends on the target star.
  """
  if select_bin_size:
    lc_phased_binned=lc_phased.bin(bins=bins)
  else:
    # convert the binning time from minutes to days
    tess_observation_cadence_days = time_bin_size_min/24/60
    lc_phased_binned = lc_phased.bin(tess_observation_cadence_days)
  return lc_phased_binned





def plot_lightcurve_phased_binned(lc_phased_binned, tic_id, method='lightkurve'):
  """
  Plot the phase folded and binned light curve by using the Lightkurve or Matplotlib methods.
  Input:
    - lc_phased_binned: A LightCurve object consisting in the phase folded and binned signal.
    - tic_id: the TIC ID of the target star. type:str.
    - method: 'lightkurve' or 'matplotlib'
  #NOTE. Remember to insert show() and savefig() methods if you run this code on Purplejeans
  """
  if method=='lightkurve':
    # Plot the global view with a contiguous line
    lc_phased_binned.plot()
  else:
    plt.plot(lc_phased_binned.time.value, lc_phased_binned.flux.value, '.')
    plt.xlabel('Bins')
    plt.ylabel('Normalized Flux')
    plt.legend([tic_id])
    plt.title('Global view')





"""
  The following two methods interpolate the phase folded binned light curve in order to fix its length to a specific value.
  Source: https://stackoverflow.com/questions/44238581/interpolate-list-to-specific-length
  Input:
    - lc_phased_binned: A LightCurve object consisting in the phase folded binned light curve.
    - n_bins: the length of the output LightCurve object (e.g. the number of the global view bins)
"""
def interpolate(inp, fi):
  i, f = int(fi // 1), fi % 1   # Split floating-point index into whole & fractional parts.
  j = i+1 if f > 0 else i       # Avoid index error.
  return (1-f) * inp[i] + f * inp[j]

def set_global_view_length(lc_phased_binned, n_bins):
  inp_flux = lc_phased_binned.flux.value
  inp_time = lc_phased_binned.time.value
  delta = (len(inp_flux)-1) / (n_bins-1)

  global_view_flux = [interpolate(inp_flux, i*delta) for i in range(n_bins)]
  global_view_time = [interpolate(inp_time, i*delta) for i in range(n_bins)]

  return lk.LightCurve(time = global_view_time, flux = global_view_flux)





def my_lc_flattening(lc_collection, transit_mask, window_length=11, polyorder=3):
  """
    Removes the low frequency trend and all the events that are not due to the event to be processed by using scipy’s Savitzky-Golay filter.
    Input:
      - lc_collection: a LightCurve object. The light curve to be flattened;
      - transit_mask: Mask that flags transits. Mask is True where there are transits of interest.
      - window_length: The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.
      - polyorder: The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    Output:
      - the flattened LightCurve object. The transits of interest have been preserved during the flattening.
  """
  # My Savitzky-Golay interpolation. The interpolated output signal has the same shape as the input
  my_savgol_filter = savgol_filter(lc_collection.flux.value, window_length=window_length, polyorder=polyorder)
  # My Flattening
  flattened_flux = []
  for tm_i in range(len(transit_mask)):
    if transit_mask[tm_i] == False:
      flattened_flux.append( lc_collection.flux.value[tm_i] / my_savgol_filter[tm_i] )
    else:
      # This is a transit point, do not flatten
      flattened_flux.append( lc_collection.flux.value[tm_i] )
  # Return a LightCurve object containing the flattened flux data
  my_lc_flatten = lk.LightCurve(time=lc_collection.time.value, flux=flattened_flux)

  return my_lc_flatten





def my_phase_folding(lc, t0, period, phase_range=1):
  """
    Fold the light curve over the period of the TCE
  """
  # Compute the phase [0,1] (in cycles) for each observation (ti,mag_i), according to the following formula:
  # phi = decimal part of [(ti-t0)/period]
  phi = []
  for i in range(len(lc.time.value)):
    phase = ( lc.time.value[i]-t0 ) / period
    # Remove the integer part because we are not interested in what cycle we are. We retain the decimal part only because we want to know in which part of the generic cycle we are.
    #phi.append( phase-int(phase) )
    phi.append( phase%1 )
  # Sort the pairs (phase,mag) by phase time in ascending order
  zipped = zip(lc.flux.value, phi)
  phase_folded_sorted_in_time = sorted(zipped, key=lambda x: x[1])
  phi_sort = []
  mag_sort = []
  for i in range(len(phase_folded_sorted_in_time)):
    phi_sort.append(phase_folded_sorted_in_time[i][1])
    mag_sort.append(phase_folded_sorted_in_time[i][0])
  # Compute the phase at the previous cycle (i.e. [-1,0]) in order to have a clear picture of the shape of the entire cycle [-1,1]
  phi1 = []
  for i in range(len(phi_sort)):
    phi1.append( phi_sort[i]-1 )
  # Return a LightCurve object consisting in the phase folded light curve over an entire cycle
  full_phase = phi1 + phi_sort
  full_flux = mag_sort + mag_sort

  if phase_range > 1:
    raise Exception("Range must be in (0,1]")

  out_phase=[]
  out_flux=[]
  for flux_i, phase_i in sorted( zip(full_flux, full_phase), key=lambda x: x[1] ):
    if phase_i > -phase_range and phase_i < phase_range:
      out_phase.append( phase_i )
      out_flux.append( flux_i )

  lc_phased = lk.LightCurve(time = out_phase, flux = out_flux)
  return lc_phased





def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    Source: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]





def interpolate_nan(global_view_flux):
    """
        Detect NaNs in the global view flux data and interpolate these values
        
        Input:
            - global_view_flux: a flux vector with possible NaNs in data
        Output:
            - global_view_flux: the same flux vector without NaN. Note that the length of the input vector is preserved.
    """
    nans, x= nan_helper(global_view_flux)
    global_view_flux[nans]= np.interp(x(nans), x(~nans), global_view_flux[~nans])

    return global_view_flux





def zero_mean_one_std(global_view_flux):
    """
        Method to scale the flux data to have zero mean and unit standard deviation (0,1)

        Input:
            - global_view_flux, a flux vector without NaNs
        Output:
            - new_global_view_flux, a flux vector scaled to (0,1)
    """
    return (global_view_flux - np.mean(global_view_flux))/np.std(global_view_flux)





def scale_in_a_b(global_view_flux, a, b):
    """
        Method to scale your global view flux data in the interval [a,b].

        Input:
            - global_view_flux: a flux vector;
            - a, b: the left and right extremes of the range, respectively.
        Output:
            - scaled_global_view_flux: a flux vector with flux data scaled in [a,b]
        
        Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    """
    return (b-a)*( global_view_flux - np.min(global_view_flux) )/(np.max(global_view_flux) - np.min(global_view_flux)) + a





def zero_median_fixed_depth(global_view_flux):
  """
    To normalize the resulting views of each TCE, this method subtracts the median (np.median(global_view_flux)) from each view
    and then divide it by the absolute value of the minimum (np.min(global_view_flux)).

    Input:
        - global_view_flux: a flux vector;
    Output:
        - norm_global_view_flux: the normalized flux vector with median 0 and maximum transit depth -1
  """
  global_view_flux -= np.median(global_view_flux, axis=0)
  global_view_flux /= np.abs(np.min(global_view_flux, axis=0))
  
  return global_view_flux