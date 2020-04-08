# Citation
This repository is base on *Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet”* [Arxiv](http://arxiv.org/abs/1808.00158) and his repository (https://github.com/mravanelli/SincNet).

# SincNet
SincNet is a neural architecture for processing **raw audio samples**. It is a novel Convolutional Neural Network (CNN) that encourages the first convolutional layer to discover more **meaningful filters**. SincNet is based on parametrized sinc functions, which implement band-pass filters.

In contrast to standard CNNs, that learn all elements of each filter, only low and high cutoff frequencies are directly learned from data with the proposed method. This offers a very compact and efficient way to derive a **customized filter bank** specifically tuned for the desired application. 

This project releases a collection of codes and utilities to perform speaker identification with SincNet.
An example of speaker identification with the TIMIT database is provided. If you are interested in **SincNet applied to speech recognition you can take a look into the PyTorch-Kaldi github repository (https://github.com/mravanelli/pytorch-kaldi).** 

<img src="https://github.com/mravanelli/SincNet/blob/master/SincNet.png" width="400" img align="right">

[Take a look into our video introduction to SincNet](https://www.youtube.com/watch?v=mXQBObRGUgk&feature=youtu.be)


## Prerequisites
- Linux
- Python 3.6/2.7
- pytorch 1.0
- pysoundfile (``` conda install -c conda-forge pysoundfile```)
- We also suggest using the anaconda environment.


## How to run a TIMIT experiment
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

TIMIT dataset can be found [here](https://github.com/philipperemy/timit)

**1. Run TIMIT data preparation.**

This step is necessary to store a version of TIMIT in which start and end silences are removed and the amplitude of each speech utterance is normalized. To do it, run the following code:

``
python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored
- *data_lists/TIMIT_all.scp* is the list of the TIMIT files used for training/test the speaker id system.

**note**
The name of directories in TIMIT dataset above are uppercase while the original script of TIMIT_preparation.py expects filenames in lowercase (e.g, train/dr1/fcjf0/si1027.wav" rather than "TRAIN/DR1/FCJF0/SI1027.WAV). 

**2. Run the speaker id experiment.**

- Modify the *[data]* section of *cfg/SincNet_TIMIT.cfg* file according to your paths. In particular, modify the *data_folder* with the *$OUTPUT_FOLDER* specified during the TIMIT preparation. The other parameters of the config file belong to the following sections:
 1. *[windowing]*, that defines how each sentence is split into smaller chunks.
 2. *[cnn]*,  that specifies the characteristics of the CNN architecture.
 3. *[dnn]*,  that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
 4. *[class]*, that specify the softmax classification part.
 5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

- Once setup the cfg file, you can run the speaker id experiments using the following command:

``
python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
``

The network might take several hours to converge (depending on the speed of your GPU card). In our case, using an *nvidia TITAN X*, the full training took about 24 hours. If you use the code within a cluster is crucial to copy the normalized dataset into the local node, since the current version of the code requires frequent accesses to the stored wav files. Note that several possible optimizations to improve the code speed are not implemented in this version since are out of the scope of this work.

**3. New features.**

- CNN-based: We can switch the first convolution layer between sinc convolution and standard convolution. To do that, modify the *[cnn]* section of *cfg/SincNet_TIMIT.cfg* file. Set "use_SinConv" = True if we want to use sinc convolution and Set "use_SinConv" = True if we want to use standard convolution.
We have to change "cnn_N_filt" and "cnn_len_filt" according to.

- Linearly spaced: We can swtich the bank filters between mel-scale and linearly spaced. To do that, modify the *[cnn]* section of *cfg/SincNet_TIMIT.cfg* file. Set "use_mel_scale" = False if we want to use linearly spaced bank filters and vice versa. 

- Evaluation step: we add the evaluation step in which, we load a pre-trained model and evaluate on test set. To do that, modify the *[data]* section of *cfg/SincNet_TIMIT.cfg* file. Set "isTraing" = False and set the path of pre-trained model (pt_file=exp/SincNet_TIMIT/model_raw.pkl)

**3. Results.**

Compare Classification Error (measured at frame level) and Classification Error (measured at sentence level) among CNN-based, SincNet-mel-scale, SincNet-linearly-spaced:

<img src="https://github.com/tuananh0305/End2End_SpeakRecognition_SincNet/blob/master/FrameErrorRate.png" width="400" img align="centre">

<img src="https://github.com/tuananh0305/End2End_SpeakRecognition_SincNet/blob/master/SentenceErrorRate.png" width="400" img align="centre">


## References

[1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)
