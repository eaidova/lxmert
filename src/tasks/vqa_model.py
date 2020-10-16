# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        self.logit_fc = None

        
        # VQA Answer heads
        self.create_head(num_answers)

    def create_head(self, num_answers):
        hid_dim = self.lxrt_encoder.dim
        if self.logit_fc is None:
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_answers)
            )
            init_weights = (
                self.lxrt_encoder.model.init_bert_weights
                if not isinstance(self.lxrt_encoder.model, nn.DataParallel)
                else self.lxrt_encoder.model.module.init_bert_weights
            )
            self.logit_fc.apply(init_weights)
            return
        self.logit_fc[-1] = nn.Linear(hid_dim * 2, num_answers)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


