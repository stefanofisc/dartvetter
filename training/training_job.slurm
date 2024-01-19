#!/bin/bash
#SBATCH -o path-to-your-output-file-%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --job-name="TRAIN-DARTVETTER"


source /home/s.fiscale/anaconda3/etc/profile.d/conda.sh

conda activate name_of_your_conda_environment

TODAY="20240101"
EPOCHS="10"
DAUG="data-augmentation-details-if-needed"
LR="1e-3"
RV="03"
BS="128"
DATASET="dataset-name"
MODEL='model-name'

TESS_PATH="/home/s.fiscale/altro/tensorflow_test/TESS/"
KEPLER_PATH="/home/s.fiscale/altro/tensorflow_test/kepler/"

TFRECORD_KEPLER=${KEPLER_PATH}"tfrecord_kepler_dr24/"
TFRECORD_YU2019=${TESS_PATH}"tfrecord_yu2019/"

DEST_FOLDER="path-to-save-your-model/"${TODAY}"_training_"${DATASET}"_"${EPOCHS}"ep_"${DAUG}"_"${LR}"_rateval"${RV}"_batchsize"${BS}"_"${MODEL}"/"

python /home/s.fiscale/conda/Models/cnn_multiple_inputs/input_pipeline.py --dataset="standard" --tfrecord_dir="${TFRECORD_KEPLER}" --dest_folder="${DEST_FOLDER}" --k_fold="1" --epochs="11" --pretrain="False" --finetuning="False" --shuffle_dataset="True" --learning_rate="0.001" --rateval="0.3" --is_astronet="False"
