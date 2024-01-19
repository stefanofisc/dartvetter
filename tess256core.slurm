#!/bin/bash

#SBATCH -J job_name
#SBATCH -o path/to/output/file/-%J.out
#SBATCH -N 4
#SBATCH -w, --nodelist=wnode01,wnode02,wnode03,wnode04

source /home/s.fiscale/anaconda3/etc/profile.d/conda.sh

conda activate name_of_your_conda_environment

cd path-to-generate_input_records.py-folder

echo "starting domain decomposition... "

TODAY="20240101"
FILENAME="name-of-tfrecord-directory-to-keep-track-of-the-data-used"

PATH_TO_TCE_CSV_FILES="/home/s.fiscale/conda/Dataset/tess_catalogs_tce/"

TCE_CSV_FILE_EXOFOP=${PATH_TO_TCE_CSV_FILES}"20230112_exofop_tess_tois.csv"
TCE_CSV_FILE_EXOFOP_QLP=${PATH_TO_TCE_CSV_FILES}"20230112_exofop_tces_not_in_yu_qlp_only.csv"
TCE_CSV_FILE_EXOFOP_SPOC=${PATH_TO_TCE_CSV_FILES}"20230112_exofop_tces_not_in_yu_spoc_only.csv"

TCE_CSV_FILE_EXOFOP_TT9_SPOC=${PATH_TO_TCE_CSV_FILES}"20230626_TT9_2.0_yes_spoc_data_with_epoch.csv"
TCE_CSV_FILE_EXOFOP_TT9_QLP=${PATH_TO_TCE_CSV_FILES}"20230626_TT9_2.0_yes_qlp_data_with_epoch.csv"
TCE_CSV_FILE_EXOFOP_TT9_CACCIAPUOTI_SPOC=${PATH_TO_TCE_CSV_FILES}"20230626_TT9_cacciapuoti_yes_spoc_data.csv"
TCE_CSV_FILE_EXOFOP_TT9_CACCIAPUOTI_QLP=${PATH_TO_TCE_CSV_FILES}"20230626_TT9_cacciapuoti_yes_qlp_data.csv"

TCE_CSV_FILE_LY=${PATH_TO_TCE_CSV_FILES}"2019_liangyu_tces.csv"
TCE_CSV_FILE_TEY=${PATH_TO_TCE_CSV_FILES}"20230502_tey2022_tces_with_labels_v3.csv"
TCE_CSV_FILE_TEY_NOT_YU=${PATH_TO_TCE_CSV_FILES}"20230623_tey_tic_not_in_yu.csv"

TFRECORD_DIR="tfrecord_"${FILENAME}"_furtherdetails_"${TODAY}
ERROR_FOLDER="errori_"${TODAY}"_tfrecord"${FILENAME}"_furtherdetails"
PATH_TO_OUTPUT_VIEWS="/home/s.fiscale/altro/tensorflow_test/TESS/output_views/views_"${FILENAME}"_furtherdetails_"${TODAY}"/"

mkdir "${TFRECORD_DIR}"
mkdir "${ERROR_FOLDER}"

mpirun -v --oversubscribe  -hostfile machinefile_tess -n 128 python generate_input_records.py --gv_length="201" --lv_length="61" --tfrecord_dir="${TFRECORD_DIR}" --tce_csv_file="${TCE_CSV_FILE_EXOFOP_TT9_CACCIAPUOTI_QLP}" --error_folder="${ERROR_FOLDER}" --detection_pipeline="QLP" --catalog="exofop_tt9_cacciapuoti" --path_to_output_views="${PATH_TO_OUTPUT_VIEWS}"
