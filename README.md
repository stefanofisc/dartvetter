# DART-Vetter: A Deep LeARning Tool for automatic vetting of TESS candidates

### Contact
Stefano Fiscale: stefano.fiscale001@studenti.uniparthenope.it

### Background
DART-Vetter is a Convolutional Neural Network trained on Kepler and TESS Threshold Crossing Events (TCEs). The model is designed to distinguish planetary candidates from false positives detected in any transiting survey. 
For further details, readers may find useful to read <a href="https://link.springer.com/chapter/10.1007/978-981-99-3592-5_12">Fiscale et al. (2023)</a> in <i>Applications of Artificial Intelligence and Neural Systems to Data Science</i> pp 127–135.

### Citation
The paper detailing the model architecture and most recent applications on TESS candidates is in preparation. We planned to submit the manuscript to The Astrophysical Journal.

### Code
This section provides an overview on the content of each directory
- <a href="https://github.com/stefanofisc/dartvetter/tree/main/TFRecord">TFRecord</a>: methods for creating and visualizing the samples of TFRecord files;
     - <a href="https://github.com/stefanofisc/dartvetter/tree/main/TFRecord/tfrecord_data">tfrecord_data</a>: TFRecord files used to train the model;
- <a href="https://github.com/stefanofisc/dartvetter/tree/main/cnn">cnn</a>: several methods to build different network architectures;
- <a href="https://github.com/stefanofisc/dartvetter/tree/main/preprocessing">preprocessing</a>: a set of files developed to pre-process light curves. The main file is generate_input_records.py and it is used to produce the global view for each TCE. Since this pre-processing pipeline deal with a huge volumen of data (i.e. TCEs), the workload has been distributed on different nodes. We provide the tess256core.slurm file to allow the user to run the generate_input_records.py file in parallel;
- <a href="https://github.com/stefanofisc/dartvetter/tree/main/tce_csv_catalogs">tce_csv_catalogs</a>: we provide all the TCE csv catalogs used in this work;
- <a href="https://github.com/stefanofisc/dartvetter/tree/main/training">training</a>: methods for model training and assessment;
    - <a href="https://github.com/stefanofisc/dartvetter/tree/main/training/trained_models/dartvetter">trained_models/dartvetter</a>: checkpoint files to load our best model. Build the model and load the optimized weights contained in this file if you do not want to run the training procedure.

### Installation
1. Training set generation:\n
   You might generate the training samples through the generatate_input_records.py file. In this case you need to download the Kepler and TESS light curves from <a href="https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html">Mikulski Archive for Space Telescopes</a>. Otherwise, we provided in the <a href="https://github.com/stefanofisc/dartvetter/tree/main/TFRecord/tfrecord_data">tfrecord_data</a> directory the Kepler and TESS global views we used to train our model.
2. Training the model: \n
   To train the model, you can use the <a href="https://github.com/stefanofisc/dartvetter/blob/main/training/training_job.slurm">training_job.slurm</a> file. Otherwise, you might build a CNN architecture with the <a href="https://github.com/stefanofisc/dartvetter/blob/main/cnn/cnn_architecture.py">cnn_architecture.py</a> file, save the model and load the optimized weights we made available in <a href="https://github.com/stefanofisc/dartvetter/tree/main/training/trained_models/dartvetter">this</a> directory.
