
import torch

class ModelFactory:

    def __init__(self, args):
        self.config = args

    def build(self, device=torch.device('cpu'), wrapper=lambda x: x):
        emb_dim = self.config.emb_dim

        duration = self.config.duration
        fps = self.config.fps
        layout = self.config.pose_layout
        streams={
            'midi': self.config.midi,
            'pose': self.config.pose,
        }


        if self.config.name == 'music_transformer':
            from .music_transformer_dev.music_transformer import music_transformer_dev_baseline

            pose_seq2seq = music_transformer_dev_baseline(
                264,
                d_model=emb_dim,
                dim_feedforward=emb_dim * 2,
                encoder_max_seq=int(duration * fps),
                decoder_max_seq=self.config.decoder_max_seq,
                layout=layout,
                num_encoder_layers=self.config.num_encoder_layers,
                num_decoder_layers=self.config.num_decoder_layers,
                rpr=self.config.rpr,
                use_control='control' in streams,
                layers=self.config.pose_net_layers
            )

        else:
            raise Exception

        pose_seq2seq = pose_seq2seq.to(device)
        pose_seq2seq = wrapper(pose_seq2seq)

        return pose_seq2seq
