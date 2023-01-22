"""
C Major
-c '1,0,1,0,1,1,0,1,0,1,0,1'
C Minor
-c '1,0,1,1,0,1,0,1,1,0,0,1'

"""

from core.models import ModelFactory
from core.dataloaders import DataLoaderFactory
import argparse
from pathlib import Path
import torch
import numpy as np
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Instrument
from pprint import pprint
from tqdm import tqdm
from core.dataloaders.dataset import D2MIDIDataset
from core.models.music_transformer_dev.music_transformer import MusicTransformer
from core import utils
from core.utils.make_data import process_single_piece_inference
from core.utils.encoding import ison,char2int,str2pit
import os
import json

PAD = 1
EOS = 2
BOS = 0

DEVICE = torch.device('cuda')
RATIO = 4
MAX_POS_LEN = 4096
PI_LEVEL = 2
IGNORE_META_LOSS = 1 
NOTON_PAD = BOS if IGNORE_META_LOSS == 0 else PAD
NOTON_PAD_DUR = NOTON_PAD 
NOTON_PAD_TRK = NOTON_PAD 

class Dictionary(object):
    def __init__(self):
        self.vocabs = {}
        self.voc2int = {}
        self.str2int = {}
        self.merges = None
        self.merged_vocs = None

    def load_vocabs_bpe(self, DATA_VOC_DIR, BPE_DIR=None):
        for i in range(RATIO):
            with open(f'{DATA_VOC_DIR}vocab_{i}.json', 'r') as f:
                self.vocabs[i] = json.load(f)
                # str:int
                self.voc2int[i] = {v:int(k)for k, v in self.vocabs[i].items()}
        with open(f'{DATA_VOC_DIR}ori_dict.json', 'r') as f:
            self.str2int = json.load(f)

        self.str2int.update({x:(PAD if IGNORE_META_LOSS == 1 else BOS) for x in ('RZ', 'TZ', 'YZ')}) 

    def index2word(self, typ, i):
        return self.vocabs[typ][str(i)]

    def word2index(self, typ, i):
        return self.voc2int[typ][i]

    def is_bom(self, idx):
        return self.index2word(0, idx)[0].lower() == 'm'


def get_video_name_list(video_root):
    videos = os.listdir(video_root)
    video_list = {}
    for video in videos:
        key = video.split('.')[0]
        value = os.path.join(video_root, video)
        video_list[key] = value
    return video_list


def get_video_path(video_root, vid_name: str):  # get correspond video for ffmpeg
    videos = get_video_name_list(video_root)
    vid_path = videos[vid_name]
    return vid_path


def change_time_format(time):
    return str(int(time / 60)).zfill(2) + ':' + str(int(time % 60)).zfill(2)

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word

def get_trk_ins_map(prime, ins_list, music_dict):

    track_map = {}

    idx = 0
    for (e,d,t,_, _, _),ins in zip(prime, ins_list):
        ee = music_dict.index2word(0, e)
        idx += 1
        
        # judge if a event str is a bpe token
        if ison(ee):
            track_map.setdefault(t, []).append(ins)

    # get instrument list for each track
    trk_ins_map = {}
    for k in track_map:
        v = torch.stack(track_map[k])
        logits = torch.mean(v, axis=0)

        ins_word = sampling(logits,p=0.9)
        trk_ins_map[k] = ins_word
    return trk_ins_map

def get_note_seq(prime,music_dict):

    note_seq = []
    measure_time = 0
    last_bom = 0
    error_note = 0

    for (e,d,t,i, _, _) in prime[1:]:

        ee = music_dict.index2word(0, e)
        # BOM Token
        if ee[0].lower() == 'm':
            measure_time += last_bom
            last_bom = char2int(ee[1])+(62 if ee[0] == 'M' else 0)
            last_pos = -1
        # pos token
        elif ee[0].lower() == 'p':
            last_pos = char2int(ee[1]) + (62 if ee[0] == 'P' else 0)
        # CC/NT Token
        elif ee == 'NT':
            last_pos = -1
        # chord token
        elif ee[0].lower() == 'h':
            pass
        # event token 
        elif ison(ee):
            if t != NOTON_PAD_TRK and d != NOTON_PAD_DUR:
                dd = music_dict.index2word(1, d)
                tt = music_dict.index2word(2, t)
                ii= music_dict.index2word(3, i)
                assert last_pos != -1, 'Invalid generation: there must be a <pos> between <on> and <cc>'
                start = measure_time + last_pos
                trk = char2int(tt[1])+(62 if tt[0] == 'T' else 0)
                dur = char2int(dd[1])+(62 if dd[0] == 'R' else 0)
                ins = char2int(ii[1])+(62 if ii[0] == 'X' else 0)

                for i in range(0, len(ee), 2):
                    eee = ee[i:i+2]
                    note_seq.append((str2pit(eee), ins, start, start + dur, trk))
            else:
                error_note += 1
        else:
            assert False, ('Invalid generation: unknown token: ', (ee, d, t))
    # print(f'error note cnt: {error_note}')
    return note_seq

def note_seq_to_midi_file(note_seq, filename, ticks_per_beat=480):

    tickes_per_32th = ticks_per_beat // 8
    tracks = {}
    for pitch, program, start, end, track_id in note_seq:
        program=int(program)


        tracks.setdefault((track_id, program), []).append(mtkNote(90, pitch, start * tickes_per_32th, end * tickes_per_32th))

    midi_out = MidiFile(ticks_per_beat=ticks_per_beat)

    for tp, notes in tracks.items():
        program = tp[1]
        instrument = Instrument(program % 128, is_drum=program >= 128)
        instrument.notes = notes
        instrument.remove_invalid_notes(verbose=False)
        midi_out.instruments.append(instrument)
    midi_out.dump(filename)

def main(args):
    # load data dict
    music_dict = Dictionary()
    DATA_VOC_DIR=f"data/vocabs/"
    music_dict.load_vocabs_bpe(DATA_VOC_DIR)

    # load checkpoint
    torch.set_grad_enabled(False)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    if args.control is not None:
        control_tensor = utils.midi.pitch_histogram_string_to_control_tensor(args.control)
    else:
        control_tensor = None

    cp = torch.load(checkpoint_path)


    model_factory = ModelFactory(args)
    dataloader_factory = DataLoaderFactory(args)

    model: MusicTransformer = model_factory.build(device=DEVICE)

    model.load_state_dict(cp['state_dict'])
    model.eval()

    dl = dataloader_factory.build(split='val')
    ds: D2MIDIDataset = dl.dataset
    pprint(ds.samples[:5])

    os.makedirs(output_dir / 'audio', exist_ok=True)
    os.makedirs(output_dir / 'video', exist_ok=True)


    for data in tqdm(ds):
        index = data['index']
        pose = data['pose']

        pose = pose.cuda(non_blocking=True)
        if control_tensor is not None:
            control_tensor = control_tensor.cuda(non_blocking=True)
        sample = ds.samples[index]
        
        events_toks = model.generate(
            pose.unsqueeze(0),
            target_seq_length=args.decoder_max_seq,
            beam=0,
            pad_idx=ds.NPAD,
            sos_idx=ds.NBOS,
            eos_idx=ds.NEOS,
            control=control_tensor,
        )
        
        events_toks=events_toks.cpu().detach().numpy()
        events_toks=events_toks[0]
        events_toks=events_toks.tolist()
        # remove BOS token
        events_toks=events_toks[1:]
        print(events_toks)
        print(len(events_toks))


        midi_dir = output_dir / 'midi' / f'{sample.vid}'
        os.makedirs(midi_dir, exist_ok=True)
        midi_path = midi_dir / f'{sample.vid}.mid'

        measures, _, _, _ = process_single_piece_inference((events_toks, music_dict.str2int), RATIO, MAX_POS_LEN)

        prime_nums = [[EOS]*RATIO + [0, 0]]
 
        prime_nums[0][3] = 1 # set instrument to vanilla pos

        for mea_id, mea in enumerate(measures):
            for id in range(0, len(mea), RATIO+1):
                mea_pos = (mea_id+1) * 3
                if id == 0:
                    mea_pos -= 2
                elif id == RATIO+1:
                    mea_pos -= 1
                cur_tok = mea[id:id+RATIO+1] + [mea_pos]
                prime_nums.append(cur_tok)
        
        note_seq = get_note_seq(prime_nums,music_dict)
        note_seq_to_midi_file(note_seq, midi_path) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('-o', '--output',default='exps/output/')
    parser.add_argument('-c', '--control', default=None)
    parser.add_argument('-oa', '--only_audio', action="store_true")

    parser.add_argument('--num_workers', default=512, type=int, help='number of workers')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--load_checkpoint', default=None, type=str, help='load checkpoint')
    # dataset config
    parser.add_argument('--dset', default='D2MIDI', type=str, help='dataset name')
    parser.add_argument('--duration', default=30, type=float, help='duration of the video')
    parser.add_argument('--fps', default=22, type=int, help='fps of the video')
    parser.add_argument('--pose_layout', default='body25', type=str, help='pose layout')
    parser.add_argument('--duplication', default=1, type=int, help='duplication')
    parser.add_argument('--events_per_sec', default=80, type=int, help='events per sec')
    parser.add_argument('--random_shift_rate', default=0.2, type=float, help='random shift rate')  
    parser.add_argument('--split_csv_dir', default='../', type=str)
    parser.add_argument('--midi', default='../video_trim_midi/', type=str)
    parser.add_argument('--pose', default='../video_trim_output_npy/', type=str)
    # models config
    parser.add_argument('--name', default='music_transformer', type=str)
    parser.add_argument('--emb_dim', default=512, type=int)
    parser.add_argument('--clip', default='null', type=str)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--rpr', default=True, type=bool)
    parser.add_argument('--decoder_max_seq', default=1024+1, type=int)
    parser.add_argument('--pose_net_layers', default=10, type=int)

    args = parser.parse_args()
    main(args)
