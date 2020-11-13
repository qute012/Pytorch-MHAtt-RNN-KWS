import torch
import torch.nn as nn
import torch.nn.functional as F

class KWS(nn.Module):
    def __init__(self, num_classes=len(CLASSES), in_channel=1, hidden_dim=128, n_head=4):
        super(KWS, self).__init__()
        self.num_classes = num_classes
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channel, 10, (5,1), stride=(1,1), dilation=(1,1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 1, (5,1), stride=(1,1), dilation=(1,1)),
            nn.BatchNorm2d(45),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        
        self.rnn = nn.LSTM(1,self.hidden_dim, num_layers=2, bidirectional=True, 
                             batch_first=True)
        self.q_emb = nn.Linear(self.hidden_dim<<1, (self.hidden_dim<<1)*self.n_head)
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Sequential(
            nn.Linear(1024,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.Linear(32,self.num_classes)
        )
        #self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.cnn_extractor(x)
        x = x.reshape(x.size(0),-1,x.size(1))
        x,_ = self.rnn(x)
        
        middle = x.size(1)//2
        mid_feature = x[:,middle,:]
        
        multiheads = []
        queries = self.q_emb(mid_feature).view(self.n_head, batch_size, -1, self.hidden_dim<<1)
        for query in queries:
            att_weights = torch.bmm(query,x.transpose(1, 2))
            att_weights = F.softmax(att_weights, dim=-1)
            multiheads.append(torch.bmm(att_weights, x).view(batch_size,-1))
        x = torch.cat(multiheads, dim=-1)
        x = self.dropout(x)
        
        x = self.fc(x)
        
        return x
