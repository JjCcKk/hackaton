#from ntpath import join
from pydoc import doc
import torch as tr
#import torch.cuda
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
import matplotlib.pyplot as plt


class SarcasmDataset(tr.utils.data.Dataset):

    def __init__(self, annotationFile, MEL, target_samplerate, num_samples,device):
        self.annotations = pd.read_csv(annotationFile)
        #self.annotations = pd.read_json(annotationFile)

        self.device = device

        self.target_samplerate = target_samplerate
        self.num_samples = num_samples

        #self.voiceExtractor = voiceExtractor
        self.MEL = MEL

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.take_left_channel(signal)
        signal = self.resample_if_nessesary(signal,sr)
        #signal = self.mixdown_if_nessesary(signal)
        #signal = self._cut_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)
        signalList = self.split_and_stack(signal)

        #plt.figure()
        #plt.plot(signalList[0].numpy())
        #plt.plot(signalList[1].numpy())
        #plt.show()

        signal = self.MEL_to_tensor(signalList)

        plt.figure()
        plt.imshow(signal[1,:,:])
        plt.show()

        lable = self._get_audio_sample_lable(index)
        return signal, lable
    
    
    
    def take_left_channel(self, signal):
        return signal[0,:]


    #def _cut_if_necessary(self, signal):
    #    if signal.size(dim=0) > self.num_samples:
    #        signal = signal[:, :self.num_samples]
    #    return signal

    def right_pad_if_necessary(self, signal):
        length_signal = signal.size(dim=0)
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = tr.nn.functional.pad(signal, last_dim_padding)
        return signal
    

    def resample_if_nessesary(self, signal, sr):
        if sr != self.target_samplerate:
            resampler = torchaudio.transforms.Resample(sr,self.target_samplerate)
            resampler = resampler.to(self.device) #dirty Hack?
            signal = resampler(signal)
        return signal

    def mixdown_if_nessesary(self, signal):
        if signal.shape[0] > 1:
            signal = tr.min(signal,dim=0,keepdim=True)
        return signal
    

    def _get_audio_sample_path(self, index):

        path = self.annotations.values[index] #File Name in annotations-file in first column.
        path = path[0]
        return path

    def _get_audio_sample_lable(self,index):
        temp = self.annotations.values[index]
        return temp[1]


    #def voiceExtractor(self,signal):


    def split_and_stack(self, signal): #Returns array of tensors with length num_samples.

        out = []
        l = signal.size(dim=0)
        num_windows = int(np.floor(l / int(self.num_samples)))
        print(f"Number of windows: {num_windows}")
        if num_windows == 0:
            print("Signal has not been right-padded correctly.")
        clock = 0
        for i in range(num_windows):
            out.append(signal[clock:(clock+int(self.num_samples))])
            clock = clock + int(self.num_samples)
        print(f"Length of list of tensors: {len(out)}")
        return out


    def MEL_to_tensor(self, signalList): #Computes the MEL spectrogram for each signal in list and stacks them in a torch-tensor.
        s = len(signalList)
        out = []
        for i in range(s):
            x = signalList[i]
            m = self.MEL(x)
            out.append(m)

        out = tr.stack(out, dim=0)
        out.to(self.device)
        return out

        

    
if __name__ == "__main__":
    
            ANNOTATIONS_FILE = "annotations.csv"
            AUDIO_DIR = "/"
            SAMPLE_RATE = 44100
            #NUM_SAMPLES = 44100
            NUM_SAMPLES = 44100

            if tr.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            print(f"Using {device}")

            MELi = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=128, n_mels=64, normalized=True)

            sd = SarcasmDataset(ANNOTATIONS_FILE, MELi, SAMPLE_RATE, NUM_SAMPLES, device)

            print("There are " + str(len(sd)) + " samples in the dataset.")
            test = sd[40]
            #print(np.shape(test))

            plt.figure()
            plt.imshow(test[1,0:200])
            plt.show()



            #signal, lable = sd[1]
