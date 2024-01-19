#!/bin/bash
#SBATCH -o path-to-your-output-file-%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --job-name="PREDICT"


source /home/s.fiscale/anaconda3/etc/profile.d/conda.sh

conda activate name_of_your_conda_environment

TESS_PATH="/home/s.fiscale/altro/tensorflow_test/TESS/"
KEPLER_PATH="/home/s.fiscale/altro/tensorflow_test/kepler/"


TFRECORD_EXOFOP_QLP_NOTINYU=${TESS_PATH}"tfrecord_exofop_not-in-yu2019/qlp/"
TFRECORD_EXOFOP_SPOC_NOTINYU=${TESS_PATH}"tfrecord_exofop_not-in-yu2019/spoc/"
TFRECORD_EXOFOP_TT9_SPOC_SAP=${TESS_PATH}"tfrecord_tt9_spoc_allyear_normalized_nouniform_20230708/"
TFRECORD_EXOFOP_TT9_V1V2=${TESS_PATH}"tfrecord_tt9/"




python /home/s.fiscale/conda/Models/cnn_multiple_inputs/evaluate.py --tfrecord_dir="${TFRECORD_EXOFOP_TT9_V1V2}"
