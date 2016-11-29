import numpy as np
import librosa
import os
import _pickle as pickle
import gzip
import sys
# ignore deprecation warnings caused by librosa 0.4.3 cqt with numpy
import warnings
warnings.filterwarnings("ignore")

import scipy.io
import matplotlib.pyplot as plt

NCHANNELS = 16

'''
functions to preprocess audio i.e. extract features for classification
uses a log-power CQT encoded into a uint16 numpy matrix to save space
contains both pyspark functions for running on clusters/multiple threads
and vanilla python functions
'''

def feature_extract(songfile_name):
    '''
    takes: filename
    outputs: audio feature representation from that file (currently cqt)
    **assumes working directory contains raw song files**
    returns a tuple containing songfile name and numpy array of song features
    '''

    """
    test = scipy.io.loadmat("train_1/1_146_1.mat")
    data = test['dataStruct'][0,0][0]
    print(data.shape)
    #summed = data[:,0]
    summed = np.sum(data,axis=1)
    #plt.plot(summed)

    stft = librosa.stft(summed, n_fft=400, win_length=200, hop_length=200)
    """
    eeg_loc = os.path.abspath(songfile_name)
    rawfile = scipy.io.loadmat(eeg_loc)
    rawdata = rawfile['dataStruct'][0,0][0]

    stft = np.stack([librosa.stft(np.squeeze(layer), n_fft=400,
                            win_length=200, hop_length=200)
                            for layer in np.split(rawdata, NCHANNELS, axis=1)],
                    axis=-1)

    """desire_spect_len = 2580
    C = librosa.cqt(y=y, sr=sr, hop_length=512, fmin=None,
    n_bins=84, bins_per_octave=12, tuning=None,
    filter_scale=1, norm=1, sparsity=0.01, real=False)
    # get log-power spectrogram with noise floor of -80dB
    C = librosa.logamplitude(C**2, ref_power=np.max)
    # scale log-power spectrogram to positive integer value for smaller footpint
    noise_floor_db = 80
    scaling_factor = (2**16 - 1)/noise_floor_db
    C += noise_floor_db
    C *= scaling_factor
    C = C.astype('uint16')
    # if spectral respresentation too long, crop it, otherwise, zero-pad
    if C.shape[1] >= desire_spect_len:
        C = C[:,0:desire_spect_len]
    else:
    C = np.pad(C,((0,0),(0,desire_spect_len-C.shape[1])), 'constant')
    """
    return songfile_name, stft

'''
vanilla python functions
'''

def create_feature_matrix(song_files):
    '''
    takes: song folder filled with mp3 files as input
    outputs: key-value pairs of filenames : feature representations (spectrograms)
    list of filenames that caused exceptions
    '''
    feature_matrix = {}
    exceptions = []
    for filename in song_files: #os.listdir(song_folder):)
        if filename.endswith(".mat"):
            try:
                print("Processing: ", filename, end="\r")
                name, features = feature_extract(filename)
                feature_matrix[name] = features
            except:
                print("Exception on: ", filename)
                exceptions.append(filename)
    return feature_matrix, exceptions

def save_feature_matrix(song_folder, save_path):
    fm,excepts = create_feature_matrix(song_folder)
    print("Processing complete, saving...")
    fileHandle = gzip.open(save_path, "wb")
    pickle.dump(fm, fileHandle)
    print("Saved.")
    fileHandle.close()

'''
pyspark functions
'''

def create_feature_matrix_spark(song_files):
    # cqt wrapper
    def log_cqt(y,sr):
        C =  librosa.cqt(y=y, sr=sr, hop_length=512, fmin=None,
        n_bins=84, bins_per_octave=12, tuning=None,
        filter_scale=1, norm=1, sparsity=0.01, real=True)
        # get log-power spectrogram with noise floor of -80dB
        C = librosa.logamplitude(C**2, ref_power=np.max)
        # scale log-power spectrogram to positive integer value for smaller footpint
        noise_floor_db = 80
        scaling_factor = (2**16 - 1)/noise_floor_db
        C += noise_floor_db
        C *= scaling_factor
        C = C.astype('uint16')
        return C
    # padding wrapper
    def padding(C,desired_spect_len):
        if C.shape[1] >= desired_spect_len:
            C = C[:,0:desired_spect_len]
        else:
            C = np.pad(C,((0,0),(0,desired_spect_len-C.shape[1])), 'constant')
        return C
    # load try-catch wrapper
    def try_load(filename):
        try:
            sys.stdout.write('Processing: %s \r' % os.path.basename(filename))
            sys.stdout.flush()
            return librosa.load(filename)
        except:
            pass
    # transormations
    filesRDD = sc.parallelize(song_files)
    rawAudioRDD = filesRDD.map(lambda x: (os.path.basename(x),try_load(x))).filter(lambda x: x[1] != None)
    rawCQT = rawAudioRDD.map(lambda x: (x[int(0)], log_cqt(x[int(1)][int(0)],x[int(1)][int(1)])))
    paddedCQT = rawCQT.map(lambda x: (x[0],padding(x[1],2580)))
    return paddedCQT.collect()

def save_feature_matrix_spark(song_files,save_path,save_name):
    fm = create_feature_matrix_spark(song_files)
    fm = {i[0]:i[1] for i in fm}
    with gzip.open(os.path.join(save_path,save_name), "wb") as fileHandle:
        pickle.dump(fm, fileHandle)

def process_chunks(song_folder, save_path, overwrite = False, num_chunks=40, train_or_test="train"):
    '''
    chunk feature matrix creation to deal with memory constraints
    will split the output files in to num_chunks chunks
    '''
    files = [os.path.join(song_folder,filename) for filename in os.listdir(song_folder) if filename.endswith(".mat")]
    chunk_size = int(len(files) / num_chunks)
    j=0
    for i in range(0, len(files), chunk_size):
        j+=1
        save_name = '{}_set_cqt{}.pickle.gz'.format(train_or_test,j)
        if not os.path.isfile(os.path.join(save_path,save_name)) or overwrite:
        	save_feature_matrix(files[i:i + chunk_size],save_path + "_" + str(i))
        else:
        	print('{} already exists. Skipping file.'.format(save_name))

# cluster
# song_folder = '/scratch/mss460/shs/shs_train'
# save_path = '/home/mss460/training_set_cqt.pickle'
# hadoop song folder
#hadoop_song_folder = '/user/mss460/shs/shs_train'

# local
#song_folder = '/Users/Mike/Documents/kaggle-ecg/'
#save_path = '/Users/markostamenovic/Desktop/shs_train_pickles'

# run with command:
# spark-submit --driver-memory 8g --conf spark.driver.maxResultSize=2g preprocess_audio.py

if __name__ == "__main__":
    """
    from pyspark import SparkContext
    sc = SparkContext()
    print("Executing as main program")
    process_chunks(song_folder, save_path, overwrite = False, num_chunks=20, train_or_test="train")
    """
    process_chunks("train_1", "train_1_processed")
