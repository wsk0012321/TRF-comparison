import pickle
import pandas as pd
from glob import glob

def align_with_eeg(df):
    
    # create an expanded dataset matching the number of EEG sample points
    time_points = [row[1].to_list() for row in df.iterrows()]
    last_point = time_points[-1][0]
    sample_points = [num/128 for num in range((int(round(last_point,0))+2)*128)]
    expanded_data = [0] * len(sample_points)
    
    
    # insert the values to an appropriate postion
    for p in time_points:
        closest = min(sample_points, key = lambda x: abs(x-p[0]))
        index = sample_points.index(closest)
        expanded_data[index] = (closest,p[1],p[2])
        
    # fill in the gaps   
    for i in range(1,len(expanded_data)):
        if expanded_data[i] == 0:
            expanded_data[i] = expanded_data[i-1]
        else:
            pass
    
    # drop out zeros and the tail
    
    # first we iterate the list in reversed direction to truncate the tail
    last_val = (expanded_data[-1][1],expanded_data[-1][2])
    count = 0
    last_idx = len(expanded_data)
    for item in reversed(expanded_data):
        if (item[1],item[2]) == last_val:
            count += 1
        else:
            # we preserve 128 points (1 second) after the onset of the last stimulus
            rstrip_id = last_idx - (count - 128)
            break
            
    # then we iterate the list to drop out all the zeros
    truncate_idx = 1
    for i in range(len(expanded_data)):
        if expanded_data[i+1] != 0:
            break
        else:
            truncate_idx += 1
            
    expanded_data = expanded_data[truncate_idx:last_idx]
    
    return expanded_data

#paths = glob(r'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/cosine/*.csv')
paths = glob(r'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/pearsons_corr/*.csv')

expanded_data = []
for p in paths:
    df = pd.read_csv(p)
    data = align_with_eeg(df)
    expanded_data.append(data)

# save as pkl file
with open(r'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/value_cosine.pkl','wb') as file:
    pickle.dump(expanded_data, file)
