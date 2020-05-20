import os
import argparse
import json
import sys
import numpy as np
import torch


from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write


# import IPython.display as ipd
import string
import keyboard
import simpleaudio
import threading

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


class args:
    config = 'config.json'
    params = []
    flowtron_path = 'models/flowtron_ljs.pt'
    waveglow_path = 'models/waveglow_256channels_v4.pt'
    speaker_id = 0
    output_dir = 'results/'
    sigma = 0.5
    seed = 1234
    n_frames = 300  # the smaller it is, the easier to put it in memory
    gate_threshold = 0.5


with open(args.config) as f:
    data = f.read()
        
config = json.loads(data)
update_params(config, args.params)
data_config = config["data_config"]
model_config = config["model_config"]

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load waveglow
waveglow = torch.load(args.waveglow_path)['model'].cuda().eval()
waveglow.cuda().half()
for k in waveglow.convinv:
    k.float()
waveglow.eval()

# load flowtron
model = Flowtron(**model_config).cuda()
state_dict = torch.load(args.flowtron_path, map_location='cpu')['state_dict']
model.load_state_dict(state_dict)
model.eval()
print("Loaded checkpoint '{}')" .format(args.flowtron_path))



def play(text):
    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(data_config['training_files'], **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))
    speaker_vecs = trainset.get_speaker_id(args.speaker_id).cuda()
#     print(f'Using text: {text}')
    text_cuda = trainset.get_text(text).cuda()
    speaker_vecs = speaker_vecs[None]
    text_cuda = text_cuda[None]

    with torch.no_grad():
        residual = torch.cuda.FloatTensor(1, 80, args.n_frames).normal_() * args.sigma
        mels, attentions = model.infer(residual, speaker_vecs, text_cuda, gate_threshold=args.gate_threshold)

    audio = waveglow.infer(mels.half(), sigma=0.8).float()
    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    print(audio.shape)
    simpleaudio.play_buffer(audio, num_channels=1, bytes_per_sample=4, sample_rate=data_config['sampling_rate'])

print('Listening')
cur_text = ''
while True:  # making a loop
    key = keyboard.read_key()
    if keyboard.is_pressed(key):
#         print(key)
        if key in string.ascii_lowercase:
            cur_text = cur_text + key
        elif key == 'space':
            cur_text = cur_text + ' '
        elif key == '.':
            cur_text = cur_text + key
            print(cur_text)
            play_thread = threading.Thread(target=play, args=(cur_text,))
            play_thread.start()
            cur_text = ''
        if key == 'alt gr':
            break
            

