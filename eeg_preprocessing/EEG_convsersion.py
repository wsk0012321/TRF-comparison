import scipy.io as sio
import pandas as pd
import numpy as np
import mne
import re
from glob import glob
from tqdm import tqdm

chanlocs = pd.read_csv(r'E:/PhD/data/Di_Liberto/raw/chanlocs.csv',encoding='utf-8')
locs = list(zip(chanlocs['Y'],chanlocs['X'],chanlocs['Z']))
ch_names = chanlocs['labels'].to_list()
ch_locs = {
    name: list(loc)
    for name, loc in zip(ch_names,locs)
}
sfreq = 128
ch_types = ['eeg'] * len(ch_names)
montage = mne.channels.make_dig_montage(ch_pos=ch_locs, coord_frame='head')
info = mne.create_info(ch_names=ch_names,sfreq=sfreq,ch_types=ch_types)

def reconstruct_set(reference,montage,info,eeg_raw,name,run):

    raw = mne.io.RawArray(eeg_raw.T,info)
    raw._data -= reference
    raw.set_montage(montage)
    output_dir = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/EEG/' + run + '/' + name + '.set'
    mne.export.export_raw(output_dir,raw,fmt='eeglab',overwrite=True)

paths = glob(r'E:/PhD/data/Di_Liberto/Di_Liberto/Natural Speech/EEG/*/*.mat')

for p in tqdm(paths):
    name = re.search('Subject\d+_Run\d+',p).group()
    run = re.search('Run\d+',p).group()
    if run not in ['Run9','Run10','Run16','Run17']:
        file = sio.loadmat(p)
        eeg_raw = file['eegData']
        reference = np.mean(file['mastoids'].T,axis=0)
        reconstruct_set(reference,montage,info,eeg_raw,name,run)
    else:
        pass    
