# SincNet

In this project, we will replicate the SincNet model [1] for the speaker identification. Throughout this project, we can evaluate the performance of the SincNet architecture
integrated Machine Learning algorithms in speech identification. In addition, we try to change configurations in model architecture, parameters, optimization algorithms, to get
various performances in speech identification tasks, then compare them with those of the original model in the paper. Throughout these implementation tasks, we can get
familiar with the interesting concepts of speech processing, such as window and filters, Discrete Fourier transform, cepstral analysis, and how to combine them in a Deep
Learning architecture to solve problems in speech identification and verification as well.

# Citation

This repository is base on *Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet”* [Arxiv](http://arxiv.org/abs/1808.00158) and his repository (https://github.com/mravanelli/SincNet).


# Data set

Our training dataset is TIMIT Acoustic-Phonetic Continuous Speech Corpus, influenced by John S. Garofolo, Lori F. Lamel, William M. Fisher, Jonathan G. Fiscus, David S. Pallett, Nancy L. Dahlgren, Victor Zue [2]. The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. It contains ​a total of 6300 sentences, 10 sentences spoken by each of 630 speakers from 8 major dialect regions of the United States. A sentence data consist of a speech waveform file (.wav) and
three associated transcription files (.txt, .wrd, .phn).


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

**3. New features.**

- CNN-based: We can switch the first convolution layer between sinc convolution and standard convolution. To do that, modify the **[cnn]** section of **cfg/SincNet_TIMIT.cfg** file. Set "use_SinConv" = True if we want to use sinc convolution and Set "use_SinConv" = True if we want to use standard convolution.
We have to change "cnn_N_filt" and "cnn_len_filt" according to.

- Linearly spaced: We can swtich the bank filters between mel-scale and linearly spaced. To do that, modify the **[cnn]** section of **cfg/SincNet_TIMIT.cfg** file. Set "use_mel_scale" = False if we want to use linearly spaced bank filters and vice versa. 

- Evaluation step: we add the evaluation step in which, we load a pre-trained model and evaluate on test set. To do that, modify the **[data]** section of **cfg/SincNet_TIMIT.cfg** file. Set "isTraing" = False and set the path of pre-trained model (pt_file=exp/SincNet_TIMIT/model_raw.pkl)

- Draw Cumulative frequency response of learned filter: We calculate the cumulative frequency response of a set of learned filters with initialized
weights being mel-scaled. After getting learned filters from training, we apply DFT to these filters in time-domain to transfer it to frequency domain, we will get the magnitude of frequency response. Then we sum them to get their cumulative frequency responses.



**3. Results.**

Compare Classification Error (measured at frame level) and Classification Error (measured at sentence level) among CNN-based, SincNet-mel-scale, SincNet-linearly-spaced:

<img src="https://github.com/tuananh0305/End2End_SpeakRecognition_SincNet/blob/master/FrameErrorRate.png" width="400" img align="centre">

<img src="https://github.com/tuananh0305/End2End_SpeakRecognition_SincNet/blob/master/SentenceErrorRate.png" width="400" img align="centre">

We train the SincNet model with Adam optimizer and different learning rate and choose the best one (lr=0.001), then we train this configuration 10 times and get the average result.

The table below compares the ​Frame Error Rate and ​Classification ​Error Rate of our modified model and the given author’s model (Github) and the result of the SincNet paper.

![](https://github.com/tuananh0305/End2End_SpeakRecognition_SincNet/blob/master/result.png)



## References

[1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)
