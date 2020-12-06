import os
import torch
from torch import nn
import numpy as np

from lxrt.entry import convert_sents_to_features
from lxrt.tokenization import BertTokenizer
from inter_bert.modeling import InterBertForVLTasks, BertConfig


class InterBERTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        config = BertConfig.from_json_file(args.config_file)
        # Build Model
        self.model = InterBertForVLTasks.from_pretrained(
            "snap/pretrained/inter_bert/pytorch_model.bin", config
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        image_mask = torch.tensor(np.ones((input_mask.shape[0], 36)), dtype=torch.long).cuda()
        multimodal_mask = torch.cat((image_mask, input_mask), dim=-1)
        output = self.model(
            input_ids, *feats,
            segment_ids, input_mask, image_mask, multimodal_mask
            )
        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_InterBERT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load InterBERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_InterBERT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)
