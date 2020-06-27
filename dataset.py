import numpy as np
import torch
from torch.utils.data import Dataset
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sr, sample_frames):
        self.root = Path(root).absolute()
        root1 = self.root / 'voice'
        root2 = self.root / 'unit'
        self.hop_length = hop_length
        self.sample_frames = sample_frames

        self.metadata = [x for x in root1.rglob('*.*')]
        self.metadata += [x for x in root2.rglob('*.*')]
        print(self.metadata)

        with open(Path("./cfg/speakers.json").absolute()) as file:
            self.speakers = sorted(json.load(file))


        '''
        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = (sample_frames + 2) * hop_length / sr
        with open(self.root / "train.json") as file:
            metadata = json.load(file)
            self.metadata = [
                Path(out_path) for _, _, duration, out_path in metadata
                if duration > min_duration
            ]
        '''
    def __len__(self):
        return len(self.metadata)//2

    def __getitem__(self, index):
        path_mel = self.metadata[index*2]
        path_wav = self.metadata[index*2+1]

        audio = np.load(path_wav)
        mel = np.load(path_mel)

        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = -1
        for i in range(0,len(self.speakers)):
            if self.speakers[i] in path_mel.name:
                speaker = i
                break
        if speaker==-1:
            print("Error in dataloader : get item")
        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker
