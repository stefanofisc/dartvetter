import numpy as np

author_info  = [
    ['SPOC', 'EXOFOP_SPOC', 'TEY2022_SPOC'],
    ['QLP', 'EXOFOP_QLP', 'TEY2022_QLP'],
    ['TEY2022_SINGLE']
    ]

catalog_name_list = ['tess', 'tey2022', 'exofop', 'exofop_tt9', 'exofop_tt9_cacciapuoti']

# Define TESS observation sectors for Years 1-5. Source: https://tess.mit.edu/observations/
tess_year_1 = np.arange(1,14)
tess_year_2 = np.arange(14,27)
tess_year_3 = np.arange(27,40)
tess_year_4 = np.arange(40,56)
tess_year_5 = np.arange(56,70)

tess_observation_sectors = [ tess_year_1, tess_year_2, tess_year_3, tess_year_4, tess_year_5 ] 
