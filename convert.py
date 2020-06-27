import json
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm

from preprocess import preemphasis
from model import Encoder, Decoder


def convert():
    '''
    dataset_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    with open(dataset_path / "speakers.json") as file:
        speakers = sorted(json.load(file))
    '''
    dataset_path = Path('./cfg').absolute()
    with open(dataset_path / "speakers.json") as file:
        speakers = sorted(json.load(file))
    with open(Path("./cfg/cfg.json").absolute()) as file:
        para = json.load(file)

    synthesis_list_path = Path('./dataset/english/synthesis.txt').absolute()
    with open(synthesis_list_path) as file:
        synthesis_list = json.load(file)
    in_dir = Path('./dataset/english').absolute()
    out_dir = Path('./output').absolute()
    out_dir.mkdir(exist_ok=True, parents=True)
    print(synthesis_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(in_channels=para['encoder']['in_channels'], channels=para['encoder']['channels'],
                      n_embeddings=para['encoder']['n_embeddings'], embedding_dim=para['encoder']['embedding_dim'],
                      jitter=para['encoder']['jitter'])
    decoder = Decoder(in_channels=para['decoder']['in_channels'],
                      conditioning_channels=para['decoder']['conditioning_channels'],
                      n_speakers=para['decoder']['n_speakers'],
                      speaker_embedding_dim=para['decoder']['speaker_embedding_dim'],
                      mu_embedding_dim=para['decoder']['mu_embedding_dim'],
                      rnn_channels=para['decoder']['rnn_channels'], fc_channels=para['decoder']['fc_channels'],
                      bits=para['decoder']['bits'], hop_length=para['decoder']['hop_length'])
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format('./checkpoint/model.pt'))
    checkpoint_path = Path('./checkpoint/model.pt').absolute()
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    #meter = pyloudnorm.Meter(160000)
    print('load finish')
    for wav_path, speaker_id, out_filename in tqdm(synthesis_list):
        wav_path = in_dir / wav_path
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),
            sr=para['preprocess']['sr'])
        #ref_loudness = meter.integrated_loudness(wav)
        wav = wav / np.abs(wav).max() * 0.999

        mel = librosa.feature.melspectrogram(
            preemphasis(wav, para['preprocess']['preemph']),
            sr=para['preprocess']['sr'],
            n_fft=para['preprocess']['n_fft'],
            n_mels=para['preprocess']['n_mels'],
            hop_length=para['preprocess']['hop_length'],
            win_length=para['preprocess']['win_length'],
            fmin=para['preprocess']['fmin'],
            power=1)
        logmel = librosa.amplitude_to_db(mel, top_db=para['preprocess']['top_db'])
        logmel = logmel / para['preprocess']['top_db'] + 1

        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device)
        with torch.no_grad():
            z, _ = encoder.encode(mel)
            output = decoder.generate(z, speaker)

        #output_loudness = meter.integrated_loudness(output)
        #output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
        path = out_dir / out_filename
        librosa.output.write_wav(path.with_suffix(".wav"), output.astype(np.float32), sr=para['preprocess']['sr'])


if __name__ == "__main__":
    convert()
