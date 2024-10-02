import torch.nn as nn
import torch.nn.functional as F
from hear21passt.base import AugmentMelSTFT, PasstBasicWrapper
from hear21passt.base import get_model_passt
from transformers import RobertaModel, RobertaTokenizer


class WindowedPrediction(nn.Module):

    def __init__(self, model, segment_length=10, hop_size=10, sr=32000):
        super(WindowedPrediction, self).__init__()
        self.model = model
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.sr = sr

    def forward(self, audio):
        # print("Audio shape:", audio.shape)
        B = len(audio)
        audio = split_audio(audio, segment_length=self.segment_length, hop_size=self.hop_size, sr=self.sr)
        # print("Audio reshape:", audio.shape)
        audio_embeds = self.model(audio)  # [B*s, d]
        # print("Audio embeds shape:", audio_embeds.shape)
        audio_embeds = audio_embeds.reshape(B, len(audio_embeds) // B, audio_embeds.shape[-1])  # [B, s, d]
        # print("Audio embeds reshape:", audio_embeds.shape)
        # Average windowed audio embeddings
        audio_embeds = audio_embeds.mean(dim=1)  # [B, d]
        # print("Audio output shape:", audio_embeds.shape)
        return audio_embeds


def split_audio(x, segment_length=10, hop_size=10, sr=32000):
    segment_length = int(segment_length * sr)
    hop_size = int(hop_size * sr)
    if x.shape[1] < segment_length:
        return x
    x = x.unfold(dimension=1, size=segment_length, step=hop_size).reshape(-1, segment_length)
    return x


def get_passt(**kwargs):
    net = get_model_passt(arch="passt_s_p16_s16_128_ap468", input_tdim=1000, fstride=16, tstride=16,
                          s_patchout_t=15, s_patchout_f=2)

    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0,
                         timem=0, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)
    # mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
    #                      timem=192, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
    #                      fmax_aug_range=2000)

    model = PasstBasicWrapper(mel=mel, net=net, mode="embed_only", **kwargs)

    return WindowedPrediction(model, segment_length=10, hop_size=10, sr=32000)


def get_roberta(**kwargs):
    model = RobertaModel.from_pretrained("roberta-large",
                                         add_pooling_layer=False, hidden_dropout_prob=0.2,
                                         attention_probs_dropout_prob=0.2, output_hidden_states=False)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    return model, tokenizer


class DualEncoderModel(nn.Module):

    def __init__(self, **kwargs):
        super(DualEncoderModel, self).__init__()

        self.audio_encoder = get_passt()
        self.text_encoder, self.tokenizer = get_roberta()

        self.audio_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

    def audio_branch(self, audio):
        audio_input = audio.to(self.text_encoder.device)
        audio_feats = self.audio_encoder(audio_input)
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        return audio_embeds

    def text_branch(self, text):
        text_input = self.tokenizer(text,
                                    add_special_tokens=True,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=32,
                                    return_tensors="pt").to(self.text_encoder.device)
        text_feats = self.text_encoder(input_ids=text_input.input_ids,
                                       attention_mask=text_input.attention_mask)[0]
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def forward(self, audio, text):
        audio_embeds = self.audio_branch(audio)  # [B, d]
        text_embeds = self.text_branch(text)  # [B, d]
        return audio_embeds, text_embeds
