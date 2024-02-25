import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, dataframe, max_len, tokenizer, label2id, training=True):
        self.len = len(dataframe)
        self.data = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.training = training

    def __getitem__(self, index):
        if self.training:
            labels = self.data.tags[index]
        tokenized_sentence = self.data.tokens[index]

        maxlen = self.max_len

        if len(tokenized_sentence) > maxlen:
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            if self.training:
                labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + [
                "[PAD]" for _ in range(maxlen - len(tokenized_sentence))
            ]
            if self.training:
                labels = labels + ["O" for _ in range(maxlen - len(labels))]

        attn_mask = [1 if tok != "[PAD]" else 0 for tok in tokenized_sentence]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        if self.training:
            label_ids = [self.label2id[label] for label in labels]

        result = {
            "identifier": self.data.id[index],
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }

        if self.training:
            result["targets"] = torch.tensor(label_ids, dtype=torch.long)

        return result

    def __len__(self):
        return self.len
