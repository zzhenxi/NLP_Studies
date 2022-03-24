'''
스크립트 내 포함해야하는 함수
- set_device()
- custom_collate_fn()
포함해야하는 클래스
- CustomDataset
- CustomClassifier
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    print(device)



def custom_collate_fn(batch):
  """
  - batch : list of tuples (input_data(string), target_data(int))  
  - output : (input, target) 튜플 형태를 반환.
  """
  global tokenizer_bert
  
  input_list, target_list = [],[]

  for x,y in batch:
    input_list.append(x)
    target_list.append(y)
  
  tensorized_input = tokenizer_bert(text=input_list,
                                    add_special_tokens=True,
                                    padding='longest',
                                    return_tensors='pt')
  
  tensorized_label = torch.tensor(target_list)
  
  return tensorized_input, tensorized_label
  


class CustomDataset(Dataset):
  """
  - input_data: list of string
  - target_data: list of int
  """

  def __init__(self, input_data:list, target_data:list) -> None:
      self.X = input_data
      self.Y = target_data

  def __len__(self):
      return len(self.Y)

  def __getitem__(self, index):
      return self.X[index], self.Y[index]



class CustomClassifier(nn.Module):

  def __init__(self, hidden_size: int, n_label: int):
    super(CustomClassifier, self).__init__()

    self.bert = BertModel.from_pretrained("klue/bert-base")

    dropout_rate = 0.1
    linear_layer_hidden_size = 32

    self.classifier = nn.Sequential(
        nn.Linear(hidden_size, linear_layer_hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(linear_layer_hidden_size, n_label)
    )
  
  def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
    
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    cls_token_last_hidden_states = outputs['pooler_output']

    logits = self.classifier(cls_token_last_hidden_states)

    return logits