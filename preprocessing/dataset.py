import torch as tr
import pandas as pd
import torchaudio
import numpy as np
import os
import json
from config import ANNOTATIONS_FILE_TRAIN, SAMPLE_RATE, NUM_SAMPLES, ANNOTATIONS_FILE_TEST


class SarcasmDataset(tr.utils.data.Dataset):

    def __init__(self, annotationFile, MEL, target_samplerate, num_samples,device):
        self.annotations = pd.read_csv(annotationFile)
        #self.annotations = pd.read_json(annotationFile)

        self.device = device

        self.target_samplerate = target_samplerate
        self.num_samples = num_samples

        #self.voiceExtractor = voiceExtractor
        self.MEL = MEL.to(device)

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.take_left_channel(signal)
        signal = self.resample_if_nessesary(signal,sr)
        signal = self.cut_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)

        #plt.figure()
        #plt.plot(signalList[0].numpy())
        #plt.plot(signalList[1].numpy())
        #plt.show()
        
        #signal = self.MEL_to_tensor(signal)
        signal = self.MEL_on_device(signal)

        #plt.figure()
        #plt.imshow(signal[1,:,:])
        #plt.show()

        #lable = self._get_audio_sample_lable(index)
        return signal
    
    def take_left_channel(self, signal):
        return signal[0,:]


    def cut_if_necessary(self, signal):
        if signal.size(dim=0) > self.num_samples:
            signal = signal[0:(self.num_samples-1)]

        return signal

    def right_pad_if_necessary(self, signal):
        length_signal = signal.size(dim=0)
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, int(num_missing_samples))
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
    

    def get_audio_sample_path(self, index):
        return self.annotations["Path"][index]

    def get_audio_sample_label(self,index):
        return self.annotations["sarcasm"][index]

    def MEL_on_device(self, signal):
        x = self.MEL(signal)
        x.to(self.device)
        return x


def create_spec(train_dir, annotation_file, mem_train):

    MELi = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=330, n_mels=32, normalized=True)
    sd = SarcasmDataset(mem_train, MELi, SAMPLE_RATE, NUM_SAMPLES, device)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    dico_mem = {}
    for i in range(len(sd)):
        np.save(train_dir + str(i), sd[i].cpu().numpy())
        dico_mem[str(i)] = int(sd.get_audio_sample_label(i))

    with open(annotation_file, 'w') as f:
        json.dump(dico_mem, f)
    


if __name__ == "__main__":

    if tr.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")

    create_spec("/outputs/data/mel_spec_train/", '/outputs/data/annotation_train.json', ANNOTATIONS_FILE_TRAIN)
    create_spec("/outputs/data/mel_spec_test/", '/outputs/data/annotation_test.json', ANNOTATIONS_FILE_TEST)
