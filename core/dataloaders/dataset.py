from torch.utils.data import Dataset
from typing import List, Optional, Dict
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import random
from core import utils
import copy
import torch
from pretty_midi import PrettyMIDI
import copy


@dataclass
class Sample:

    vid: str
    start_time: float
    duration: float
    row: dict
    midi_path: Optional[str] = None
    pose_path: Optional[str] = None
    audio_path: Optional[str] = None
    rgb_path: Optional[str] = None
    flow_path: Optional[str] = None


class BaseDataset(Dataset):

    NBOS=261 #start token
    NEOS=262 #end token 
    NPAD=263 #pad token

    BODY_PARTS = {
        'body25': 25,
    }

    def __init__(
            self,
            split_csv_dir: str,
            streams: Dict[str, str],
            duration=6.0,
            duplication=100,
            fps=60,
            events_per_sec=20,
            audio_rate=22050,
            random_shift_rate=0.2,
            pose_layout='body25',
            split='train',
            max_seq=4096,
    ):
        self.split_csv_dir = Path(split_csv_dir)
        self.streams = streams
        self.duration = duration
        self.duplication = duplication
        self.fps = fps
        self.events_per_sec = events_per_sec
        self.audio_rate = audio_rate
        self.pose_layout = pose_layout
        self.random_shift_rate = random_shift_rate

        self.audio_duration = self.duration + 0.5  # sound is slightly longer than action

        assert split in ['train', 'val', 'test'], split
        self.split = split

        self.csv_path = self.split_csv_dir / f'{split}.csv'
        self.df = pd.read_csv(str(self.csv_path))

        self.samples = self.build_samples_from_dataframe(
            self.df,
            self.streams,
        )

        if split == 'train':
            self.samples *= duplication
        else:
            self.samples = self.build_samples_from_dataframe(
            self.df,
            self.streams,
            )

        self.num_frames = int(duration * fps)
        self.num_events = max_seq
        self.num_audio_frames = int(self.audio_duration * audio_rate)
         
        self.body_part = self.BODY_PARTS.get(pose_layout, -1) 

        self.use_pose = 'pose' in self.streams
        self.use_midi = 'midi' in self.streams
        self.use_audio = 'audio' in self.streams
        self.use_control = 'control' in self.streams
        self.use_rgb = 'rgb' in self.streams
        self.use_flow = 'flow' in self.streams

    def __getitem__(self, index):
        sample = self.samples[index]
        
        start_time = 0
        result = {}
        start_frame = int(start_time * self.fps)

        if self.use_pose:
            
            pose = utils.io.read_pose_from_npy(sample.pose_path, start_frame, self.num_frames, part=self.body_part)
            result['pose'] = torch.from_numpy(pose)

        if self.use_rgb:
            rgb = utils.io.read_feature_from_npy(
                sample.rgb_path, start_frame, self.num_frames
            )
            result['rgb'] = torch.from_numpy(rgb.astype(np.float32))

        if self.use_flow:
            flow = utils.io.read_feature_from_npy(
                sample.flow_path, start_frame, self.num_frames
            )
            result['flow'] = torch.from_numpy(flow.astype(np.float32))

        if self.use_midi:
            midi_x, control = utils.io.pm_to_list(sample.midi_path, readonly=False,use_control=self.use_control)  # Input midi
            midi_y, _ = utils.io.pm_to_list(sample.midi_path, readonly=False,use_control=False)  # Target midi, no predict control

            midi_x, control = self.pad_midi_events(midi_x, control=control)
            midi_y, _ = self.pad_midi_events(midi_y, control=None)

            # padding first
            midi_x = midi_x[:-1]
            midi_y = midi_y[1:]

            result['midi_x'] = torch.LongTensor(midi_x)
            result['midi_y'] = torch.LongTensor(midi_y)

            if self.use_control:
                control = control[:-1]  # keep the same as midi_x
                result['control'] = torch.from_numpy(control)

        if self.use_audio:
            start_index = int(start_time * self.audio_rate)
            audio = utils.io.read_wav(
                sample.audio_path, start_index, self.num_audio_frames
            )
            result['audio'] = torch.from_numpy(audio)

        if self.split != 'train':
            result['start_time'] = start_time
            result['index'] = index

        return result

    def get_samples_by_indices(self, indices) -> List[Sample]:
        result = []
        for index in indices:
            sample = self.samples[index]
            result.append(sample)
        return result

    def pad_midi_events(self, midi: List[int],control: Optional[np.ndarray] = None):
        
        new_midi = [self.NBOS] + midi + [self.NEOS]

        if control is not None:
            control = np.pad(control, ((1, 1), (0, 0)), 'constant')

        num_events = self.num_events+1

        if len(new_midi) > num_events:
            new_midi = new_midi[:num_events]
            new_midi[-1] = self.NEOS

            if control is not None:
                control = control[:num_events]
                control[-1, :] = 0

        elif len(new_midi) < num_events:
            pad = num_events - len(new_midi)
            new_midi = new_midi + [self.NPAD] * pad 

            if control is not None:
                control = np.pad(control, ((0, pad), (0, 0)), 'constant')

        return new_midi, control

    def midi_transform(self, pm: PrettyMIDI):
        notes = pm.instruments[0].notes  # Ref
        num_notes = len(notes)
        indices = random.sample(range(num_notes), int(self.random_shift_rate * num_notes))

        def get_random_number():
            return (random.random() - 0.5) * 0.2

        for index in indices:
            notes[index].start += get_random_number()
            notes[index].end += get_random_number()

        return pm

    @staticmethod
    def build_samples_from_dataframe(
            df: pd.DataFrame,
            streams: Dict[str, str],
    ):
        new_streams = {k: Path(v) for k, v in streams.items()}
        samples = []
        for _i, row in df.iterrows():
            sample = Sample(
                row.vid,
                row.start_time,
                row.duration,
                row.to_dict()
            )

            vid = row.vid
            if 'midi' in new_streams:
                midi_path = new_streams['midi'] / f'{vid}.mid'
                if not midi_path.is_file():
                    midi_path = new_streams['midi'] / f'{vid}.mid'
                sample.midi_path = str(midi_path)

            if 'pose' in streams:
                pose_path = new_streams['pose'] / f'{vid}.npy'
                sample.pose_path = str(pose_path)

            if 'audio' in streams:
                audio_path = new_streams['audio'] / f'{vid}.wav'
                sample.audio_path = str(audio_path)

            if 'rgb' in streams:
                rgb_path = new_streams['rgb'] / f'{vid}.npy'
                sample.rgb_path = rgb_path

            if 'flow' in streams:
                flow_path = new_streams['flow'] / f'{vid}.npy'
                sample.flow_path = flow_path

            samples.append(sample)
        return samples

    @staticmethod
    def split_val_samples_into_small_pieces(samples, duration: float):
        new_samples = []

        for sample in samples:
            stop = sample.duration
            # Todo 
            pieces = np.arange(0, stop, duration)[:-1]
            for new_start in pieces:
                new_sample = copy.deepcopy(sample)
                new_sample.start_time = new_start
                new_sample.duration = duration
                new_samples.append(new_sample)

        return new_samples

    @classmethod
    def from_cfg(cls, args, split='train'):
        cfg=args
        streams={
            'midi': cfg.midi,
            'pose': cfg.pose,
        }
        return cls(
            cfg.split_csv_dir,
            streams,
            split=split,
            duration=cfg.duration,
            duplication=cfg.duplication,
            fps=cfg.fps,
            events_per_sec=cfg.events_per_sec,
            random_shift_rate=cfg.random_shift_rate,
            pose_layout=cfg.pose_layout,
            max_seq=cfg.decoder_max_seq,
        )

    def __len__(self):
        return len(self.samples)

class D2MIDIDataset(BaseDataset):
    @staticmethod
    def build_samples_from_dataframe(
            df: pd.DataFrame,
            streams: Dict[str, str],
    ):
        new_streams = {k: Path(v) for k, v in streams.items()}
        samples = []
        for _i, row in df.iterrows():
            sample = Sample(
                row.vid,
                row.start_time,
                row.duration,
                row.to_dict()
            )

            vid = row.vid
            if 'midi' in new_streams:
                midi_path = new_streams['midi'] / f'{vid}.mid'
                if not midi_path.is_file():
                    midi_path = new_streams['midi'] / f'{vid}.mid'
                sample.midi_path = str(midi_path)

            if 'pose' in streams:
                pose_path = new_streams['pose'] / f'{vid}.npy'
                sample.pose_path = str(pose_path)

            if 'audio' in streams:
                audio_path = new_streams['audio'] / f'{vid}.wav'
                sample.audio_path = str(audio_path)

            if 'rgb' in streams:
                rgb_path = new_streams['rgb'] / f'{vid}.npy'
                sample.rgb_path = rgb_path

            if 'flow' in streams:
                flow_path = new_streams['flow'] / f'{vid}.npy'
                sample.flow_path = flow_path

            samples.append(sample)
        return samples
