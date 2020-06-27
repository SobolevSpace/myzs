from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from functools import partial


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def process_wav(wav_path, out_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    return out_path, logmel.shape[-1]



def preprocess_dataset():
    in_dir = Path('./dataset/english').absolute()
    out_dir = Path('./preprocessed_file').absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path("./cfg/cfg.json").absolute()) as file:
        para = json.load(file)


    for split in ["test\\", "train\\unit\\","train\\voice\\"]:
        print("Extracting features for {} set".format(split))
        split_path = in_dir.joinpath(split)
        print(split_path)
        t_dir = out_dir.joinpath(split)
        print(t_dir)
        for wave_file in split_path.rglob('*.wav'):
            process_wav(wave_file, t_dir.joinpath(wave_file.name),sr=para['preprocess']['sr'], preemph=para['preprocess']['preemph'], n_fft=para['preprocess']['n_fft'],
                        n_mels=para['preprocess']['n_mels'], hop_length=para['preprocess']['hop_length'], win_length=para['preprocess']['win_length'],
                        fmin=para['preprocess']['fmin'], top_db=para['preprocess']['top_db'], bits=para['preprocess']['bits'], offset=para['preprocess']['offset'], duration=None)

'''
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in metadata:
                wav_path = in_dir / in_path
                out_path = out_dir / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

                process_wav(wav_path, out_path)

        results = [future.result() for future in tqdm(futures)]

        lengths = [x[-1] for x in results]
        frames = sum(lengths)
        frame_shift_ms = cfg.preprocessing.hop_length / cfg.preprocessing.sr
        hours = frames * frame_shift_ms / 3600
        print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))
'''

if __name__ == "__main__":
    preprocess_dataset()
