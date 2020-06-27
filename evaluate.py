from pathlib import Path
import numpy as np
import librosa
import json
import scipy
import math

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)

def mel(wav1_path):
    with open(Path("./cfg/cfg.json").absolute()) as file:
        para = json.load(file)
    wav1, _ = librosa.load(wav1_path, sr=para['preprocess']['sr'],
                          offset=para['preprocess']['offset'], duration=None)

    wav1 = wav1 / np.abs(wav1).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav1, para['preprocess']['preemph']),
                                         sr=para['preprocess']['sr'],
                                         n_fft=para['preprocess']['n_fft'],
                                         n_mels=para['preprocess']['n_mels'],
                                         hop_length=para['preprocess']['hop_length'],
                                         win_length=para['preprocess']['win_length'],
                                         fmin=para['preprocess']['fmin'],
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=para['preprocess']['top_db'])
    logmel = logmel / para['preprocess']['top_db'] + 1
    return logmel

def dist(mat1, mat2):
    d = mat1.shape[0]
    if mat1.shape[1] > mat2.shape[1]:
        return dist(mat2,mat1)
    lmin = min(mat1.shape[1],mat2.shape[1])
    lmax = max(mat1.shape[1],mat2.shape[1])
    ret = 1e5

    for ofs in range(0,lmax-lmin, 2):
        for j in range(0,lmin):
            product = 0
            len1 = 0
            len2 = 0
            for i in range(0,d):
                product += (mat1[i][j]-mat2[i][j+ofs])**2
                len1 += mat1[i][j]*mat1[i][j]
                len2 += mat2[i][j+ofs]*mat2[i][j+ofs]
        ret = min(ret, 2*product/(len1+len2))
    return 1-ret

def cal_similarity(wav1_path, wav2_path):
    logmel1 = mel(wav1_path)
    logmel2 = mel(wav2_path)

    return dist(logmel1,logmel2)

if __name__ == '__main__':
    out_path = Path('./output').absolute()
    out_list = [x for x in out_path.rglob('*.wav')]
    val_path = Path('./validation').absolute()
    val_list = [x for x in val_path.rglob('*.wav')]
    cnt = 0
    res = 0
    for a in out_list:
        print(cnt)
        name = a.name
        for j in val_list:
            if name == j.name:
                cnt += 1
                res += cal_similarity('./output/'+name, './validation/'+name)
                break
    print(cnt)
    print('similarity:'+str(res/cnt))