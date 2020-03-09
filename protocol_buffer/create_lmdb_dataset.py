import numpy as np
import lmdb
import caffe
from   utils  import get_recursive_files, sample_audio, load_wav, manual_seed
from config import manual_seed, sampling_rate, window_length
from sklearn.model_selection import train_test_split
import wave
import librosa
from tqdm import tqdm
import datanum_pb2
# 
# protoc -I=$SRC_DIR --python_out=$DST_DIR/datanum_pb2.py $SRC_DIR/datanum.proto
# to create protoc

def get_map_size(files):
    return load_wav(files[0]).nbytes * 10 * (len(files) +2)

def get_silent_set(input_audio):
    indices = np.where(input_audio==0)[0]
    index_sets = []
    window = 16384
    counter = 0
    prev_index = -1
    first_index = -1
    for index in indices:
        if counter==0:
            first_index = index
        if index - prev_index == 1:
            counter +=1
        else:
            if counter>window:
                index_sets.append((first_index, prev_index, counter))
            counter = 0
        prev_index = index
    if counter>window:
        index_sets.append((first_index, prev_index, counter))
    return index_sets

def remove_silence(input_audio, index_sets):
    for silent_indices in index_sets:
        first_indx = silent_indices[0]
        last_idx = silent_indices[1]
        input_audio = np.delete(input_audio, [indx for indx in range(first_indx, last_idx)])
    return input_audio
    
# adapted from 
# https://github.com/francesclluis/source-separation-wavenet/blob/6d89618c77d38960c3996219f329e5806573799b/util.py#L220
# return start and ending indices first one is the start second on is the finish

def get_sequence_with_singing_indices(full_sequence, chunk_length = 800):

    signal_magnitude = np.abs(full_sequence)

    chunks_energies = []
    for i in range(0, len(signal_magnitude), chunk_length):
        chunks_energies.append(np.mean(signal_magnitude[i:i + chunk_length]))

    threshold = np.max(chunks_energies) * .1
    chunks_energies = np.asarray(chunks_energies)
    chunks_energies[np.where(chunks_energies < threshold)] = 0
    onsets = np.zeros(len(chunks_energies))
    onsets[np.nonzero(chunks_energies)] = 1
    onsets = np.diff(onsets)

    start_ind = np.squeeze(np.where(onsets == 1))
    finish_ind = np.squeeze(np.where(onsets == -1))

    if finish_ind[0] < start_ind[0]:
        finish_ind = finish_ind[1:]

    if start_ind[-1] > finish_ind[-1]:
        start_ind = start_ind[:-1]

    indices_inici_final = np.insert(finish_ind, np.arange(len(start_ind)), start_ind)

    return np.squeeze((np.asarray(indices_inici_final) + 1) * chunk_length)

def write_lmdb(out_file_name, data_list):
    lmdb_output = lmdb.open(out_file_name, map_size=get_map_size(data_list))
    with lmdb_output.begin(write=True) as txn:
        # txn is a Transaction object
        for audio_indx, audio_path in enumerate(tqdm(data_list)):
            if not('mixture' in audio_path):
                continue # looping over mixture and getting vocals from it
            mixed_data = load_wav(audio_path).astype('float32')
            vocals_data = load_wav(audio_path.replace('mixture','vocals')).astype('float32')
            '''
            # to remove zeros from mixed and vocals based on vocals
            silent_set = get_silent_set(vocals_data)
            mixed_data = remove_silence(mixed_data, silent_set)
            vocals_data = remove_silence(vocals_data, silent_set)
            '''

            vocals_indices = get_sequence_with_singing_indices(vocals_data, 800)
            

            datum = datanum_pb2.DataNum()
            datum.mixture = mixed_data.tobytes()
            datum.vocals = vocals_data.tobytes() 
            datum.vocals_indices = vocals_indices.tobytes() # used to store the indices having voice
            str_id = '{:08}'.format(audio_indx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString()) 

def create_lmdb(folder_name, out_file_name, is_train=True):
    audio_train = get_recursive_files(folder_name,'mixture.wav')
    audio_valid = None
    if is_train:
        # create validation set
        audio_train, audio_valid  = train_test_split(audio_train, test_size=0.15, random_state=manual_seed)

    write_lmdb(out_file_name, audio_train)

    if audio_valid:
        write_lmdb(out_file_name.replace('_train','')+'_valid', audio_valid)

parent_folder = '../../DataSet/musdb18/train'
create_lmdb(parent_folder, 'musdb_train')

parent_folder = '../../DataSet/musdb18/test'
create_lmdb(parent_folder, 'musdb_test', False)


# Testing Proto buffer reading
for lmdb_name in ['musdb_train', 'musdb_test', 'musdb_valid']:
    env = lmdb.open(lmdb_name, readonly=True)
    with env.begin() as txn:
        raw_datum = txn.get(b'00000000')

    datum = datanum_pb2.DataNum()
    datum.ParseFromString(raw_datum)

    mixture = np.fromstring(datum.mixture, dtype=np.float32)
    vocals = np.fromstring(datum.vocals, dtype=np.float32)
    vocals_indices = np.fromstring(datum.vocals_indices, dtype=np.int32)
    print(mixture.shape)
    print(vocals.shape)
    print(vocals_indices)









