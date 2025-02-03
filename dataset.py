import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetv1(Dataset):
    """
    A GPT-style PyTorch dataset collator for training Language Models

    args:
    text - The input text which is to be collated into the next-token prediciton task
    max_length - The maximum length through to which prediction will be followed 
    (This is the context window essentially) (Default:4)
    stride - How much to shift consecutive input-output pairs (Default: 4) 
    """

    def __init__(self, text, tokenizer, max_length = 4, stride = 4):
        self.inputs = []
        self.outputs = []
        tokens = tokenizer.encode(text)
        for idx in range(0, len(tokens) - max_length, stride):
            input_pair = tokens[idx:idx + max_length]
            output_pair = tokens[idx+1:idx + max_length + 1]
            self.inputs.append(torch.tensor(input_pair))
            self.outputs.append(torch.tensor(output_pair))

    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


#Num workers = 0 utilises maximum number of possible workers (threads)
def create_dataloader_v1(text, batch_size = 4, max_length = 256,
                         stride = 128, shuffle = True, drop_last = True,
                         num_workers = 0):
    """
    Generates a dataloader using the GPTDatasetV1 Dataset

    args:
    text - The input text used for creating dataloader object
    batch_size - The number of batches of data processed in parallel by the model
    max_length - The context window basically, over how many variables at the most are you going to 
    condition the next word predicition.
    stride - The amount to shift between continuous pairs
    shuffle - Shuffle the dataloader entries (should be true for training, false for validation)
    drop_last - Drop the last entry in the dataloader, as the size may be inconsistent to the other
    batches
    num_workers - Number of threads that will collate the data for the batch 
    

    return:
    dataloader - The DataLoader object for the provided dataset with the 
    provided DataLoader settings.
    """

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetv1(text=text,
                           tokenizer = tokenizer,
                           max_length = max_length,
                           stride = stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader