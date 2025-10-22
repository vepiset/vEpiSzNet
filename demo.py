import os
import sys
import scipy
\
import copy
import datetime

import torch
import numpy as np

from scipy.signal import butter, lfilter
from tqdm import tqdm

from model import Net

CONFIG = {
    'example_mat': './DA454858_0.mat',
    "pretrain_weight_detection": 'fold1_epoch_2_prauc_0.310962_loss_0.026618.pth',
    "pretrain_weight_cls": 'fold0_epoch_12_prauc_0.792128_loss_0.830359.pth',
    "threshold": 0.95,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    'fp16':False
}


def get_detection_model():
    model = Net()
    state_dict = torch.load(CONFIG['pretrain_weight_detection'], map_location=CONFIG['device'])
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(CONFIG['device'])
    return model


def get_cls_model():
    model = Net()
    state_dict = torch.load(CONFIG['pretrain_weight_cls'], map_location=CONFIG['device'])
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(CONFIG['device'])
    return model


detection_model = get_detection_model()
cls_model = get_cls_model()


def get_eegs(eeg_dir):
    eeg_files = []
    for root, dirs, files in os.walk(eeg_dir):
        for file in files:
            if file.endswith('.mat'):
                eeg_files.append(os.path.join(root, file))
    return eeg_files


def prerprocess(waves):
    # prerocess the data

    def butter_bandpass(lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype="band")

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def avg_lead(waves):
        # copy一份，防止原地修改
        waves = copy.deepcopy(waves)

        meadn = np.mean(waves[:19, :], axis=0)
        data = waves[:19, :] - meadn

        return data

    waves = avg_lead(waves)

    c, l = waves.shape
    if l < 2000:
        n_ = 2000 - l
        waves = np.pad(waves, ((0, 0), (0, n_)), 'constant', constant_values=0)

    waves = np.clip(waves, -600, 600)
    waves = np.ascontiguousarray(waves)

    waves = butter_bandpass_filter(waves, 0.5, 45, 500, 2)
    waves = np.expand_dims(waves, axis=0)
    return waves


def predict(mat_file):
    # read mat
    mat_data = scipy.io.loadmat(mat_file)
    # Extract EEG components
    eeg_data = mat_data['eeg_data']  # EEG signals (typically 19 channels)
    eeg_start_time = str(mat_data['eeg_start_time'][0])  # Recording start time
    eeg_end_time = mat_data['eeg_end_time'][0]  # Recording end time
    freq = int(mat_data['eeg_freq'])  # Sampling frequency
    ans = []
    eeg_start_time = datetime.datetime.strptime(eeg_start_time, '%y%m%d%H%M%S')


    window = int(4 * freq)
    stride = int(4 * freq)
    L = eeg_data.shape[1]

    for i in tqdm(range(0, L - window, stride)):

        waves = eeg_data[:, i:i + window]

        cur_timestamp = eeg_start_time+datetime.timedelta(seconds=i / freq)
        input = prerprocess(waves)
        input = input.astype(np.float32)

        input = torch.from_numpy(input)
        input = input.to(CONFIG['device'])

        # if use video set the input there.
        images =None

        #images = torch.from_numpy(images).float()
        #images = images.to(CONFIG['device'])

        with torch.amp.autocast(enabled=CONFIG['fp16'], dtype=torch.float16, device_type=CONFIG['device']):
            with torch.no_grad():
                output = detection_model(input, images)

        prob = output['prediction1'][0, 1].detach().cpu().numpy()

        types = ["TSz", "FSz", "CPSz", "OSz", "typical absence", "atypical absence", "myoclonic", "spasm", "tonic"]

        if prob > CONFIG['threshold']:
            with torch.amp.autocast(enabled=CONFIG['fp16'], dtype=torch.float16, device_type=CONFIG['device']):
                with torch.no_grad():
                    output = cls_model(input, images)
                    cls_prediction = output['prediction3'][0, 1:].detach().cpu().numpy()
                    cls_indx = np.argmax(cls_prediction)
                    cls_prob = cls_prediction[cls_indx]
            seizure_type = types[cls_indx]

            cur_result={'prob': prob,
                        'seizure_type': seizure_type,
                        'timestamp': cur_timestamp}
            ans.append(cur_result)
            print(cur_result)

    return  ans




if __name__ == '__main__':

    predict(CONFIG['example_mat'])

