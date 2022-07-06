#from ntpath import join
import torch as tr
#import torch.cuda
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
import matplotlib.pyplot as plt


class SarcasmDataset(tr.utils.data.Dataset):

    def __init__(self, annotationFile, MFCC, target_samplerate, num_samples,device):
        self.annotations = pd.read_csv(annotationFile)

        self.device = device

        self.target_samplerate = target_samplerate
        self.num_samples = num_samples

        #self.voiceExtractor = voiceExtractor
        self.MFCC = MFCC

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        #lable = self._get_audio_sample_lable(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.resample_if_nessesary(signal,sr)
        #signal = self.mixdown_if_nessesary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        #signal = self.DWT.go(signal)
        return signal #, lable

    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = tr.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def resample_if_nessesary(self, signal, sr):
        if sr != self.target_samplerate:
            resampler = torchaudio.transforms.Resample(sr,self.target_samplerate)
            resampler = resampler.to(self.device) #Hackkkkk
            signal = resampler(signal)
        return signal

    def mixdown_if_nessesary(self, signal):
        if signal.shape[0] > 1:
            signal = tr.min(signal,dim=0,keepdim=True)
        return signal
    

    def _get_audio_sample_path(self, index):

        path = self.annotations.values[index] #File Name in annotations-file.
        path = path[0]
        return path

    def _get_audio_sample_lable(self,index):
        return self.annotations.iloc(index,10) #Indx may be different!

    #def voiceExtractor(self,signal):

    def MFCC(self, signal):
        return self.MFCC(signal)
        
        
if __name__ == "__main__":
    
    ANNOTATIONS_FILE = "annotations.csv"
    AUDIO_DIR = "/"
    SAMPLE_RATE = 44100
    #NUM_SAMPLES = 44100
    NUM_SAMPLES = 22050

    if tr.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")

    
    DWTinstance = DWT()        

    sd = kickDatasetDWT(ANNOTATIONS_FILE,DWTinstance, SAMPLE_RATE, NUM_SAMPLES, device)

    print("There are " + str(len(sd)) + " samples in the dataset.")
    test = sd[10]
    print(np.shape(test))

    plt.figure()
    plt.imshow(test[:,0:200])
    plt.show()
    #signal, lable = sd[1]