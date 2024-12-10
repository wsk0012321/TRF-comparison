import pickle
import pandas as pd
import numpy as np
from glob import glob
from scipy.io import savemat

def align_with_eeg(df):
    
    # create an expanded dataset matching the number of EEG sample points (64Hz)
    time_points = [row[1].to_list() for row in df.iterrows()]
    last_point = time_points[-1][0]
    sample_points = [num/64 for num in range((int(round(last_point,0))+2)*64)]
    expanded_data = [0] * len(sample_points)
    
    # insert the values to an appropriate postion
    for p in time_points:
        closest = min(sample_points, key = lambda x: abs(x-p[0]))
        index = sample_points.index(closest)
        expanded_data[index] = (p[1],p[2])
        
    # mark out the interval of valid data
    
    # first we iterate the list in reversed direction to mark out 1-second time interval after the last word 
    last_val = expanded_data[-1]
    count = 0
    last_idx = len(expanded_data)
    for item in reversed(expanded_data):
        if item == last_val:
            count += 1
        else:
            # we preserve 64 points (1 second) after the onset of the last stimulus
            rstrip_id = last_idx - (count - 64)
            break
            
    # then we iterate the list to mark out the initial of non-zero points
    truncate_idx = 0
    for idx, vals in enumerate(expanded_data):
        if vals != 0:
            truncate_idx = idx
            break
        else:
            pass
    
    return {
        'duration':(truncate_idx,last_idx),
        'sample_points':sample_points,
        'values':expanded_data
    }

# save as matlab cell array
def save_values_mat(dataset,key1,key2):
    
    rule = {
        'w2v':0,
        'bert':1
    }
    
    val_id = rule[key1]
    valid_values, duration = [],[]
    for subdata in dataset:
        valid_values.append(np.array([vals[val_id] if vals !=0 else 0 for vals in subdata['values'] ]))
        duration.append(subdata['duration'])
        
    stimuli_data = {
        'valid_values': np.array(valid_values, dtype='object'),
        'duration': np.array(duration)
    }
    
    file_path = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/value_' + key2 + '_' + key1 + '.mat'
    savemat(file_path, stimuli_data)

def save_as_pkl_mat(key):
    # key = cosine / pearsonr
    
    load_dir = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/' + key + '/*.csv'
    save_dir = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/value_' + key + '.pkl'
    paths = glob(load_dir)
    
    expanded_data = [] 
    for p in paths:
        df = pd.read_csv(p)
        data = align_with_eeg(df)
        expanded_data.append(data)
    
    with open(save_dir,'wb') as file:
        pickle.dump(expanded_data, file)
        
    save_values_mat(expanded_data,'w2v',key)
    save_values_mat(expanded_data,'bert',key)

save_as_pkl_mat('cosine')
save_as_pkl_mat('pearsonr')

