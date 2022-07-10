import torch as tr
import torchaudio
from cnn import MelCNN
from cnn import SelfAttention

class WavLoader():

    def __init__(self, MEL, target_samplerate, num_samples,device):

        self.device = "cpu"
        self.target_samplerate = target_samplerate
        self.num_samples = num_samples
        self.MEL = MEL.to(device)

    def __len__(self):
        return(len(self.annotations))

    def getitem(self, path):
        signal, sr = torchaudio.load(path)
        signal = signal.to(self.device)
        signal = self.take_left_channel(signal)
        signal = self.resample_if_nessesary(signal,sr)
        signal = self.cut_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)
        signal = self.MEL_on_device(signal)

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
        path = self.annotations.values[index] #File Name in annotations-file in first column.
        path = path[0]
        return path


    def MEL_on_device(self, signal):
        x = self.MEL(signal)
        x.to(self.device)
        return x
        
        
        
def getMelSpectrum(path):
    MELi = torchaudio.transforms.MelSpectrogram(44100, n_fft=330, n_mels=32, normalized=True)
    loader = WavLoader(MELi,44100,44100*5, "cpu")
    return(loader.getitem(path))
    
def getPrediction(path):
    channel_factor = 4 #Can be [2,4,6]
    pool_before_attention = True #[False, True]
    reduce_channels_by = (1/4) #Can be [1/4, 1/2, 1]
    size_hidden_layer = (1/100) #Can be [(1/10),(1/20), (1/100)]
    
    SAi = SelfAttention(channel_factor**2, min(9,channel_factor**2),"cpu").to("cpu") #Parameter is dim before attention-block
    SAi = SAi.float()
    #Init instance of model
    MelCNNi = MelCNN(SAi, channel_factor, pool_before_attention, reduce_channels_by, size_hidden_layer).to("cpu")
    MelCNNi = MelCNNi.float()
    state_dict = tr.load("model/4_True_4_100.pth", map_location=tr.device('cpu'))
    MelCNNi.load_state_dict(state_dict)

    #print(MelCNNi)
    
    x = getMelSpectrum(path)
    x = x.to("cpu")
    
    with tr.no_grad():
        x = tr.unsqueeze(x,0)
        prediction = MelCNNi.forward(x)
    print(f"Prediction: {prediction}")
    
    return prediction
