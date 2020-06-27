import numpy as np
import json
from pathlib import Path
from itertools import chain
from tqdm import tqdm

import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeechDataset
from model import Encoder, Decoder

def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        #"amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


def train_model(resume):

    with open(Path("./cfg/cfg.json").absolute()) as file:
        para = json.load(file)
    tensorboard_path = Path("./tensorboard/writer").absolute()
    checkpoint_dir = Path("./checkpoint").absolute()
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(in_channels=para['encoder']['in_channels'], channels=para['encoder']['channels'],
                      n_embeddings=para['encoder']['n_embeddings'], embedding_dim=para['encoder']['embedding_dim'], jitter=para['encoder']['jitter'])
    decoder = Decoder(in_channels=para['decoder']['in_channels'], conditioning_channels=para['decoder']['conditioning_channels'],
                      n_speakers = para['decoder']['n_speakers'], speaker_embedding_dim=para['decoder']['speaker_embedding_dim'],
                      mu_embedding_dim=para['decoder']['mu_embedding_dim'], rnn_channels=para['decoder']['rnn_channels'], fc_channels=para['decoder']['fc_channels'],
                      bits=para['decoder']['bits'], hop_length=para['decoder']['hop_length'])


    encoder.to(device)
    decoder.to(device)



    if resume:
        print("Resume checkpoint from: {}:".format(resume))
        resume_path = Path("./checkpoint/model.pt").absolute()
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        print(checkpoint.keys())
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer = optim.Adam(
            chain(encoder.parameters(), decoder.parameters()),
            lr=1e-5)

        # [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[300000, 400000],
            gamma=0.5)
        optimizer.load_state_dict(checkpoint["optimizer"])
        #amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]


    else:
        global_step = 0


    sdataset = SpeechDataset(
        root='./preprocessed_file/train',
        hop_length=para['preprocess']['hop_length'],
        sr=para['preprocess']['sr'],
        sample_frames=para['preprocess']['sample_frames'])
    print(len(sdataset))
    dataloader = DataLoader(
        dataset=sdataset,
        batch_size=16,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True)

    print(len(dataloader))
    n_epochs = 1
#    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(global_step, global_step+n_epochs):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            #audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)
            #print(speakers)
            optimizer.zero_grad()
            z, vq_loss, perplexity = encoder(mels)
            output = decoder(audio[:, :-1], z, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            loss.backward()

            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1


        save_checkpoint(
                encoder, decoder, optimizer, amp,
                scheduler, global_step, checkpoint_dir)

        writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
        writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))


if __name__ == '__main__':
    train_model(True)