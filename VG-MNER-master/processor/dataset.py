import random
import os
from typing import Any
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer, RobertaTokenizer
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)


class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
    def load_from_file(self, mode="train"):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """   
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

            comput_sent = sorted(raw_words, key=lambda x:len(x), reverse=True)
            print("The longest length of all data:",len(comput_sent[0]))
        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets), len(imgs))

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping

    
class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.mode = mode
    
    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])

        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

        # assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return word_list, labels, image


class PadCollate:
    def __init__(self, args, processor) -> None:
        self.args = args
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.ignore_idx = 0
    
    def pad_collate(self, batch):
        batch_imgs = list(map(lambda t: t[2].clone().detach(), batch))
        batch_imgs = torch.stack(batch_imgs)

        batch_texts = list(map(lambda t: t[0], batch))
        batch_tokens = self.tokenizer(
            text=batch_texts,
            is_split_into_words=True,
            truncation=True,
            return_tensors='pt',
            padding=True
        )

        batch_labels = [[self.label_mapping["[CLS]"]] + label + [self.label_mapping["[SEP]"]] +
                        [self.ignore_idx] * (batch_tokens['input_ids'].size(1) - len(label) - 2)
                        for label in list(map(lambda t: t[1], batch))]
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return batch_tokens['input_ids'], batch_tokens['token_type_ids'], batch_tokens['attention_mask'], \
            batch_labels, batch_imgs

    def __call__(self, batch):
        return self.pad_collate(batch=batch)
