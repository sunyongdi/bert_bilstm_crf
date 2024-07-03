import torch.nn as nn
import torch
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class ModelOutput:
  def __init__(self, logits, labels, loss=None):
    self.logits = logits
    self.labels = labels
    self.loss = loss

class BertNer(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.bert = BertModel(config)
    hidden_size = config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = config.max_seq_len
    self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, config.num_labels)
    self.crf = CRF(config.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0]  # [batchsize, max_len, 768]
    batch_size = seq_out.size(0)
    seq_out, _ = self.bilstm(seq_out) # seq_out [batchsize, max_len, lstm_hiden * 2]
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2) # [batchsize * max_len, lstm_hiden * 2]
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1) # [batchsize, max_len, lstm_hiden * 2]
    seq_out = self.linear(seq_out) # [batchsize, max_len, num_labels]
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    # logits = torch.as_tensor(logits, dtype=torch.long)
    loss = None
    if labels is not None:
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output
  
  
if __name__ == '__main__':
  from transformers import BertConfig, BertTokenizerFast
  config = BertConfig.from_pretrained('/root/sunyd/nlp-tutorial/ner/model_hub/chinese-bert-wwm-ext')
  config.num_labels = 5
  config.max_seq_len = 126
  config.create_name = 'syd'
  model = BertNer(config)
  tokenize = BertTokenizerFast.from_pretrained('/root/sunyd/nlp-tutorial/ner/model_hub/chinese-bert-wwm-ext')
  tokenize.save_pretrained('ner')
  inputs = tokenize('你好', max_length=126, truncation=True, padding='max_length', return_tensors='pt')
  res = model(inputs['input_ids'], inputs['attention_mask'])
  model.save_pretrained('ner')
  print(res.logits)
