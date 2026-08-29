from preprocessing.load_eeg import load_recording 
from preprocessing.filtering import select_eeg_channels 

raw = load_recording("Subject00", "1") 
raw = select_eeg_channels(raw) 
print(raw.ch_names)