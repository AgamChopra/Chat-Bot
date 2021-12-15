### NOTE: Change relative paths accordingly ###
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from tqdm import trange
from datasets import load_dataset
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'

print('Active device:',device)

#%% PLEASE RUN THIS CODE BLOCK THE FIRST TIME ONLY

dataset = load_dataset('daily_dialog')
print(dataset)
dataset['train'][0]

user = []
bot = []

for i in dataset['train']:
    for j in range(len(i['dialog'])-1):
        if i['dialog'][j][0] == ' ': user.append(i['dialog'][j][1:])
        else: user.append(i['dialog'][j])
        bot.append(i['dialog'][j+1])

print(len(user),len(bot))
print(user[10])
print(bot[10])

mask = random.sample(range(len(user)), len(user))
ds = ([],[])
for i in mask:
    ds[0].append(user[i])
    ds[1].append(bot[i])

print(len(ds[0]),len(ds[1]))
print(ds[0][10])
print(ds[1][10])

vocab = []
for i in dataset['train']:
    for j in i['dialog']:
        for k in j.split():
            if k not in vocab:
                vocab.append(k) 
vocab.append('<ignore>')
vocab.append('<sos>')
vocab.append('<eos>')
vocab.append('<unk>')
print(len(vocab))
print(vocab[:3],vocab[-1])

def tokenizer(stringlist):
    tokelist = []
    for s in stringlist:
        toksens = [len(vocab)-3]
        for w in s.split():
            try:
                [tok] = [idx for idx, wrd in enumerate(vocab) if w == wrd]
            except:
                tok = len(vocab)-1
            toksens.append(tok)
        toksens.append(len(vocab)-2)
        tokelist.append(toksens)
    return tokelist

def untokenizer(tokelist):
    senlist = []
    for s in tokelist:
        sens = ''
        for idx in s:
            sens = sens+vocab[idx]+' '
        senlist.append(sens[:-1])
    return(senlist)

inx = ["Hi , How are you doing today ?"]
print(inx)
tokens = tokenizer(inx)
print(tokens)
rev_inx = untokenizer(tokens)
print(rev_inx)

lens = []
for i in dataset['train']:
    for j in i['dialog']:
        #if len(j) > maxlen:
            #maxlen = len(j) + 2
        lens.append(len(j))
sen_len = int(sum(lens)/len(lens)) + 2
print(sen_len)

def resize_toks(tokelist):
    tokelist_ = []
    for s in tokelist:
        if len(s) > sen_len:
            s = s[-sen_len:]
            s[0] = len(vocab)-3
            tokelist_.append(s)
        elif len(s) < sen_len:
            for _ in range(sen_len-len(s)):
                s.append(len(vocab)-4)
            tokelist_.append(s)
        else: tokelist_.append(s)
    return tokelist_
inx = ["Hi , How are you doing today ?","Conversational models are a hot topic in artificial intelligence research . Chatbots can be found in a variety of settings , including customer service applications and online helpdesks . These bots are often powered by retrieval-based models , which output predefined responses to questions of certain forms . In a highly restricted domain like a company’s IT helpdesk , these models may be sufficient , however , they are not robust enough for more general use-cases. Teaching a machine to carry out a meaningful conversation with a human in multiple domains is a research question that is far from solved . Recently , the deep learning boom has allowed for powerful generative models like Google’s Neural Conversational Model , which marks a large step towards multi-domain generative conversational models . In this tutorial , we will implement this kind of model in PyTorch ."]
print(inx)
tokens = tokenizer(inx)
print(tokens)
resized_tokens = resize_toks(tokens)
print(resized_tokens)
rev_inx = untokenizer(resized_tokens)
print(rev_inx)
print(len(resized_tokens[0]),len(resized_tokens[1]))

def raw_preprocess(inx):
    tokens = tokenizer(inx)
    tokens = resize_toks(tokens)
    return tokens
train_x = raw_preprocess(ds[0])
train_y = raw_preprocess(ds[1])
print(train_x[1])
print(train_y[1])

shuffler = np.random.permutation(len(train_x))
train_x = np.asarray(train_x)[shuffler]
train_y = np.asarray(train_y)[shuffler]
print(train_x.shape,train_y.shape)
ts = untokenizer(train_x[:10])
tss = untokenizer(train_y[:10])
print(ts[0],'\n'+tss[0])

tr_x = torch.from_numpy(train_x[:int(len(train_x)*0.90)])
tr_y = torch.from_numpy(train_x[:int(len(train_x)*0.90)])
val_x = torch.from_numpy(train_x[int(len(train_x)*0.90):])
val_y = torch.from_numpy(train_x[int(len(train_x)*0.90):])

torch.save(vocab,r'R:\classes 2020-22\Fall 2021\vocab.pt')
torch.save(tr_x,r'R:\classes 2020-22\Fall 2021\train_tensor.pt')
torch.save(tr_y,r'R:\classes 2020-22\Fall 2021\train_label.pt')
torch.save(val_x,r'R:\classes 2020-22\Fall 2021\validation_tensor.pt')
torch.save(val_y,r'R:\classes 2020-22\Fall 2021\validation_label.pt')

#%% Helper functions

sen_len = 63
vocab = torch.load(r'R:\classes 2020-22\Fall 2021\vocab.pt')
tr_x = torch.load(r'R:\classes 2020-22\Fall 2021\train_tensor.pt')
tr_y = torch.load(r'R:\classes 2020-22\Fall 2021\train_label.pt')
val_x = torch.load(r'R:\classes 2020-22\Fall 2021\validation_tensor.pt')
val_y = torch.load(r'R:\classes 2020-22\Fall 2021\validation_label.pt')

vocab_ = {}

for i in range(len(vocab)-4):
    vocab_[vocab[i]] = i
vocab_['<unk>'] = len(vocab)-1
print(vocab_)

def tokenizer(stringlist):
    # used to tokenize a batch of dialogs
    tokelist = []
    for s in stringlist:
        toksens = [len(vocab)-3]
        for w in s.split():
            try:
                [tok] = [idx for idx, wrd in enumerate(vocab) if w == wrd]
            except:
                tok = len(vocab)-1
            toksens.append(tok)
        toksens.append(len(vocab)-2)
        tokelist.append(toksens)
    return tokelist

def untokenizer(tokelist):
    # used to untokenize a batch of dialog tokens
    senlist = []
    for s in tokelist:
        sens = ''
        for idx in s:
            sens = sens+vocab[idx]+' '
        senlist.append(sens[:-1])
    return(senlist)

def resize_toks(tokelist):
    # used to resize dialogs to mean length
    tokelist_ = []
    for s in tokelist:
        if len(s) > sen_len:
            s = s[-sen_len:]
            s[0] = len(vocab)-3
            tokelist_.append(s)
        elif len(s) < sen_len:
            for _ in range(sen_len-len(s)):
                s.append(len(vocab)-4)
            tokelist_.append(s)
        else: tokelist_.append(s)
    return tokelist_

def raw_preprocess(inx):
    #helper function
    tokens = tokenizer(inx)
    tokens = resize_toks(tokens)
    return tokens

def cold2hot(x,vocab):
    #encode tokens to one-hot
    out = torch.zeros(x.shape[0],x.shape[1],len(vocab))
    y = nn.functional.one_hot(x%len(vocab))
    out[:,:,:y.shape[2]] = y
    return out
    
def hot2cold(x,d=2):
    #decode one-hot to index tokens
    return torch.argmax(x,dim=d).to(x.device)

def count_parameters(model):
    #count the number of trainable parameters of a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%% ### LSTM ###
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers        
        self.embedding = nn.Embedding(input_dim, emb_dim)        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers 
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

def train(model,optimizer,criterion,x,y,xe,ye,vocab,batch=16,epochs=20,lr=0.01,tf=0.5):
    train_loss_list = []
    eval_loss_list = []    
    for ep in trange(1,epochs+1):
        idx = torch.randperm(x.shape[0])
        x = x[idx]
        y = y[idx]
        model.train()
        temploss = 0
        for b in range(0, x.shape[0]-batch, batch):
            optimizer.zero_grad()
            t = y[b:b+batch].T.to(model.device)
            yp = model.forward(x[b:b+batch].T.to(model.device),t,tf)
            yp = torch.swapaxes(torch.swapaxes(yp, 1, 2),0,2).to(model.device)
            t=t.T.to(dtype=torch.int64)
            loss = criterion.forward(yp, t)
            temploss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss_list.append(temploss/int(x.shape[0]/batch))
        temploss = 0
        model.eval()
        for b in range(0, xe.shape[0]-batch, batch):
            t = ye[b:b+batch].T.to(model.device)
            yp = model.forward(xe[b:b+batch].T.to(model.device),t,tf)
            yp = torch.swapaxes(torch.swapaxes(yp, 1, 2),0,2).to(model.device)
            t=t.T.to(dtype=torch.int64)
            loss = criterion.forward(yp, t)
            temploss += float(loss)
        eval_loss_list.append(temploss/int(xe.shape[0]/batch))
        if ep % 2 == 0 or ep == 1:
            print('Epoch', ep, ':\n      Training loss:',
                  train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
    return train_loss_list,eval_loss_list
#%% initialize/load
e1 = LSTMEncoder(input_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=5, dropout=0.1)
d1 = LSTMDecoder(output_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=5, dropout=0.1)
model1 = Seq2Seq(e1,d1,device)
#model1.load_state_dict(torch.load('R:\classes 2020-22\Fall 2021\LSTM2.pth'))
print(f'The model has {count_parameters(model1):,} trainable parameters')

a = model1.forward(tr_x[:10].to('cuda'),tr_y[:10].to('cuda'),0)
print(a.shape)
a = untokenizer(hot2cold(a,2))
gt = untokenizer(tr_y[:10])
for i in range(10):
    print(gt[i])
    print(a[i])
    print('\n')
#%% train
optimizer = optim.Adam(model1.parameters(),weight_decay=1E-9,amsgrad=True)
criterion = nn.CrossEntropyLoss(ignore_index = int(len(vocab)-4),reduction='mean')
BATCH_SIZE = 32
l1,l2 = train(model1,optimizer,criterion,tr_x,tr_y,val_x,val_y,vocab,batch=BATCH_SIZE,tf=0.4,epochs=10,lr=1E-6)
plt.plot(l1, label='Train')
plt.plot(l2, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss(Cross-Entropy)')
plt.legend()
plt.title('LSTM')
plt.plot()
a = model1.forward(tr_x[:10].to('cuda'),tr_y[:10].to('cuda'),0)
a = untokenizer(hot2cold(a,2))
gt = untokenizer(tr_y[:10])
for i in range(10):
    print(gt[i])
    print(a[i])
    print('\n')
#%% save
#torch.save(model1.state_dict(), 'R:\classes 2020-22\Fall 2021\LSTM2.pth')
#%% ### CNN ###
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device,max_length = len(vocab)):
        super().__init__()
        assert kernel_size % 2 == 1
        self.device = device
        self.dropout = nn.Dropout(dropout).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(input_dim, emb_dim).to(device)
        self.pos_embedding = nn.Embedding(max_length, emb_dim).to(device)
        self.emb2hid = nn.Linear(emb_dim, hid_dim).to(device)
        self.hid2emb = nn.Linear(hid_dim, emb_dim).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, out_channels = 2 * hid_dim, kernel_size = kernel_size, padding = (kernel_size - 1) // 2) for _ in range(n_layers)]).to(device)
        
    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = nn.functional.glu(conved, dim = 1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        conved = self.hid2emb(conved.permute(0, 2, 1))
        combined = (conved + embedded) * self.scale
        return conved, combined
    
class CNNDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device,max_length = len(vocab)):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.dropout = nn.Dropout(dropout).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(output_dim, emb_dim).to(device)
        self.pos_embedding = nn.Embedding(max_length, emb_dim).to(device)
        self.emb2hid = nn.Linear(emb_dim, hid_dim).to(device)
        self.hid2emb = nn.Linear(hid_dim, emb_dim).to(device)
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim).to(device)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim).to(device)
        self.fc_out = nn.Linear(emb_dim, output_dim).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, out_channels = 2 * hid_dim, kernel_size = kernel_size) for _ in range(n_layers)]).to(device)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        combined = (conved_emb + embedded) * self.scale
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        attention = nn.functional.softmax(energy, dim=2)
        attended_encoding = torch.matmul(attention, encoder_combined)
        attended_encoding = self.attn_emb2hid(attended_encoding)
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        return attention, attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1)
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)
            conved = conv(padded_conv_input)
            conved = nn.functional.glu(conved, dim = 1)
            attention, conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)
            conved = (conved + conv_input) * self.scale
            conv_input = conved     
        conved = self.hid2emb(conved.permute(0, 2, 1))
        output = self.fc_out(self.dropout(conved))     
        return output, attention
    
class cnnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg):
        encoder_conved, encoder_combined = self.encoder(src)
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        return output, attention
    
def traincnn(model,optimizer,criterion,x,y,xe,ye,vocab,batch=16,epochs=20,lr=0.01):
    train_loss_list = []
    eval_loss_list = []    
    for ep in range(1,epochs+1):
        idx = torch.randperm(x.shape[0])
        x = x[idx]
        y = y[idx]
        model.train()
        temploss = 0
        for b in range(0, x.shape[0]-batch, batch):
            optimizer.zero_grad()
            t = y[b:b+batch].to(model.device)
            yp,_ = model.forward(x[b:b+batch].to(device),t[:,:-1])
            yp = torch.swapaxes(yp,1,2).to(model.device)
            t=t.to(dtype=torch.int64)
            loss = criterion.forward(yp, t[:,:-1])
            temploss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss_list.append(temploss/int(x.shape[0]/batch))
        temploss = 0
        model.eval()
        for b in range(0, xe.shape[0]-batch, batch):
            t = ye[b:b+batch].T.to(model.device)
            yp,_ = model.forward(xe[b:b+batch].T.to(device),t[:,:-1])
            yp = torch.swapaxes(yp, 1, 2).to(model.device)
            t=t.to(dtype=torch.int64)
            loss = criterion.forward(yp, t[:,:-1])
            temploss += float(loss)
        eval_loss_list.append(temploss/int(xe.shape[0]/batch))
        if ep % 2 == 0 or ep == 1:
            print('Epoch', ep, ':\n      Training loss:',
                  train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
    return train_loss_list,eval_loss_list
#%% initialize/load
e1 = CNNEncoder(input_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=1, dropout=0.1, kernel_size=3, device = device,max_length=len(vocab))
d1 = CNNDecoder(output_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=1, dropout=0.1, kernel_size=3, trg_pad_idx=len(vocab)-4, device = device,max_length=len(vocab))
model1 = cnnSeq2Seq(e1,d1,device)
model1.load_state_dict(torch.load('R:\classes 2020-22\Fall 2021\CNN.pth'))
print(f'The model has {count_parameters(model1):,} trainable parameters')
a,_ = model1.forward(tr_x[:10].to('cuda'),tr_y[:10].to('cuda'))
a = untokenizer(hot2cold(a,2))
gt = untokenizer(tr_y[:10])
for i in range(10):
    print(gt[i])
    print(a[i])
    print('\n')
#%% train
optimizer = optim.Adam(model1.parameters(),weight_decay=1E-6)
criterion = nn.CrossEntropyLoss(ignore_index = int(len(vocab)-4),reduction='mean')
BATCH_SIZE = 32
l1,l2 = traincnn(model1,optimizer,criterion,tr_x,tr_y,val_x,val_y,vocab=vocab,batch=BATCH_SIZE,epochs=1,lr=1E-2)
plt.plot(l1, label='Train')
plt.plot(l2, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss(Cross-Entropy)')
plt.legend()
plt.title('CNN')
plt.plot()
a,_ = model1.forward(tr_x[:10].to('cuda'),tr_y[:10].to('cuda'))
a = untokenizer(hot2cold(a,2))
gt = untokenizer(tr_y[:10])
for i in range(10):
    print(gt[i])
    print(a[i])
    print('\n')
#%% save
#torch.save(model1.state_dict(), 'R:\classes 2020-22\Fall 2021\CNN.pth')
#%% ### TRANSFORMER ###
class attEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim,dropout, device,max_length = 100):
        super().__init__()
        self.device = device        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)       
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim,dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
     
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim,  dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class attDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device,max_length = 100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention
    
class attSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention
    
def trainatt(model,optimizer,criterion,x,y,xe,ye,vocab,batch=16,epochs=20,lr=0.01):
    train_loss_list = []
    eval_loss_list = []    
    for ep in range(1,epochs+1):
        idx = torch.randperm(x.shape[0])
        x = x[idx]
        y = y[idx]
        model.train()
        temploss = 0
        for b in range(0, x.shape[0]-batch, batch):
            optimizer.zero_grad()
            t = y[b:b+batch].to(model.device)
            yp,_ = model.forward(x[b:b+batch].to(device),t)
            yp = torch.swapaxes(yp,1,2).to(model.device)
            t=t.to(dtype=torch.int64)
            loss = criterion.forward(yp, t)
            temploss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss_list.append(temploss/int(x.shape[0]/batch))
        temploss = 0
        model.eval()
        for b in range(0, xe.shape[0]-batch, batch):
            t = ye[b:b+batch].T.to(model.device)
            yp,_ = model.forward(xe[b:b+batch].T.to(device),t)
            yp = torch.swapaxes(yp, 1, 2).to(model.device)
            t=t.to(dtype=torch.int64)
            loss = criterion.forward(yp, t)
            temploss += float(loss)
        eval_loss_list.append(temploss/int(xe.shape[0]/batch))
        if ep % 2 == 0 or ep == 1:
            print('Epoch', ep, ':\n      Training loss:',
                  train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
    return train_loss_list,eval_loss_list
#%% initialize/load
e1 = attEncoder(input_dim=len(vocab), hid_dim=256, n_layers=3, n_heads=8, pf_dim=512, dropout=0.1, device=device,max_length =len(vocab))
d1 = attDecoder(output_dim=len(vocab), hid_dim=256, n_layers=3, n_heads=8, pf_dim=512, dropout=0.1, device=device,max_length =len(vocab))
model1 = attSeq2Seq(e1,d1,len(vocab)-4,len(vocab)-4,device)
model1.load_state_dict(torch.load('R:\classes 2020-22\Fall 2021\ATT.pth'))
print(f'The model has {count_parameters(model1):,} trainable parameters')
print(tr_x.shape,tr_y.shape)
#torch.Size([68446, 63]) torch.Size([68446, 63])
a,_ = model1.forward(tr_x[:1].to('cuda'),tr_y[:1].to('cuda'))
a = untokenizer(hot2cold(a,2))
gt = untokenizer(tr_y[:1])
for i in range(1):
    print(gt[i])
    print(a[i])
    print('\n')
#%% train
optimizer = optim.Adam(model1.parameters(),weight_decay=1E-11,amsgrad=True)
criterion = nn.CrossEntropyLoss(ignore_index = int(len(vocab)-4),reduction='mean')
BATCH_SIZE = 32
l1,l2 = trainatt(model1,optimizer,criterion,tr_x,tr_y,val_x,val_y,vocab=vocab,batch=BATCH_SIZE,epochs=10,lr=1E-9)
plt.plot(l1, label='Train')
plt.plot(l2, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss(Cross-Entropy)')
plt.legend()
plt.title('ATT')
plt.plot()
a,_ = model1.forward(val_x[:10].to('cuda'),val_y[:10].to('cuda'))
a = untokenizer(hot2cold(a,2))
gt = untokenizer(val_y[:10])
for i in range(10):
    print(gt[i])
    print(a[i])
    print('\n')
#%% save
#torch.save(model1.state_dict(), 'R:\classes 2020-22\Fall 2021\ATT.pth')
#%% Prep data for ensemble
modelLSTM = Seq2Seq(LSTMEncoder(input_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=2, dropout=0.1),LSTMDecoder(output_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=2, dropout=0.1),device)
modelLSTM.load_state_dict(torch.load('R:\classes 2020-22\Fall 2021\LSTM.pth'))
modelLSTM.eval()

modelCNN = cnnSeq2Seq(CNNEncoder(input_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=1, dropout=0.1, kernel_size=3, device = device,max_length=len(vocab)),CNNDecoder(output_dim=len(vocab), emb_dim=256, hid_dim=512, n_layers=1, dropout=0.1, kernel_size=3, trg_pad_idx=len(vocab)-4, device = device,max_length=len(vocab)),device)
modelCNN.load_state_dict(torch.load('R:\classes 2020-22\Fall 2021\CNN.pth'))
modelCNN.eval()

modelATT = attSeq2Seq(attEncoder(input_dim=len(vocab), hid_dim=256, n_layers=3, n_heads=8, pf_dim=512, dropout=0.1, device=device,max_length =len(vocab)),attDecoder(output_dim=len(vocab), hid_dim=256, n_layers=3, n_heads=8, pf_dim=512, dropout=0.1, device=device,max_length =len(vocab)),len(vocab)-4,len(vocab)-4,device)
modelATT.load_state_dict(torch.load('R:\classes 2020-22\Fall 2021\ATT.pth'))
modelATT.eval()

BATCH_SIZE = 32

tr_pred_lstm = []
tr_pred_cnn = []
tr_pred_att = []

val_pred_lstm = []
val_pred_cnn = []
val_pred_att = []
#%%
with torch.no_grad():  
    print('computing train data')
    for i in range(0,len(tr_x)-BATCH_SIZE,BATCH_SIZE):
        # Save predictions from each model corrosponding to expected outcome
        a = modelLSTM.forward(tr_x[i:i+BATCH_SIZE].to('cuda'),tr_y[i:i+BATCH_SIZE].to('cuda'),0)
        tr_pred_lstm.append(hot2cold(a,2))
        a,_ = modelCNN.forward(tr_x[i:i+BATCH_SIZE].to('cuda'),tr_y[i:i+BATCH_SIZE].to('cuda'))
        tr_pred_cnn.append(hot2cold(a,2))
        a,_ = modelATT.forward(tr_x[i:i+BATCH_SIZE].to('cuda'),tr_y[i:i+BATCH_SIZE].to('cuda'))
        tr_pred_att.append(hot2cold(a,2))
    print('Finished!\ncomputing validation data')
    for i in range(0,len(val_x)-BATCH_SIZE,BATCH_SIZE):
        # Save predictions from each model corrosponding to expected outcome
        a = modelLSTM.forward(val_x[i:i+BATCH_SIZE].to('cuda'),val_y[i:i+BATCH_SIZE].to('cuda'),0)
        val_pred_lstm.append(hot2cold(a,2))
        a,_ = modelCNN.forward(val_x[i:i+BATCH_SIZE].to('cuda'),val_y[i:i+BATCH_SIZE].to('cuda'))
        val_pred_cnn.append(hot2cold(a,2))
        a,_ = modelATT.forward(val_x[i:i+BATCH_SIZE].to('cuda'),val_y[i:i+BATCH_SIZE].to('cuda'))
        val_pred_att.append(hot2cold(a,2))
#%%
b1 = torch.Tensor(len(tr_pred_lstm), 32, 63).to(device)
torch.cat(tr_pred_lstm, out=b1)
print(b1.shape)  

b2 = torch.Tensor(len(tr_pred_cnn), 32, 63).to(device)
torch.cat(tr_pred_cnn, out=b2)
print(b2.shape)

b3 = torch.Tensor(len(tr_pred_att), 32, 63).to(device)
torch.cat(tr_pred_att, out=b3)
print(b3.shape)

b4 = torch.Tensor(len(val_pred_lstm), 32, 63).to(device)
torch.cat(val_pred_lstm, out=b4)
print(b4.shape)

b5 = torch.Tensor(len(val_pred_cnn), 32, 63).to(device)
torch.cat(val_pred_cnn, out=b5)
print(b5.shape)

b6 = torch.Tensor(len(val_pred_att), 32, 63).to(device)
torch.cat(val_pred_att, out=b6)
print(b6.shape)     
#%% Compute F1 Scores    

with torch.no_grad():
    print('Finished!\ncomputing F1 scores...')
    tr_pred_lstm = b1.to(dtype=torch.int32)
    tr_pred_cnn = b2.to(dtype=torch.int32)
    tr_pred_att = b3.to(dtype=torch.int32)

    val_pred_lstm = b4.to(dtype=torch.int32)
    val_pred_cnn = b5.to(dtype=torch.int32)
    val_pred_att = b6.to(dtype=torch.int32)
        
    # F1 scores/weights per model
    for i in range(int(len(val_y)/2)):
        #extract only sentences
        gt = torch.tensor_split(val_y[i],(1,1+torch.argmax(val_y[i,1:]).to('cpu')))[1].to('cpu',dtype = torch.int32)
        pred_lstm = torch.tensor_split(val_pred_lstm[i],(1,1+torch.argmax(val_pred_lstm[i,1:]).to('cpu')))[1].to('cpu',dtype = torch.int32)
        pred_cnn = torch.tensor_split(val_pred_cnn[i],(1,1+torch.argmax(val_pred_cnn[i,1:]).to('cpu')))[1].to('cpu',dtype = torch.int32)
        pred_att = torch.tensor_split(val_pred_att[i],(1,1+torch.argmax(val_pred_att[i,1:]).to('cpu')))[1].to('cpu',dtype = torch.int32)
        
        if i%1000 == 0:
            sens = ''
            for idx in gt:
                sens = sens+vocab[int(idx)]+' '
            print(sens)
            sens = ''
            for idx in pred_lstm:
                sens = sens+vocab[int(idx)]+' '
            print(sens)
            sens = ''
            for idx in pred_cnn:
                sens = sens+vocab[int(idx)]+' '
            print(sens)
            sens = ''
            for idx in pred_att:
                sens = sens+vocab[int(idx)]+' '
            print(sens,'\n')
        
        TP_global = [0,0,0]
        FP_global = [0,0,0]
        FN_global = [0,0,0]
        
        for j in range(len(gt)):
            try:
                if gt[j] == pred_lstm[j]:
                    TP_global[0] += 1
                else:
                    FN_global[0] += 1
                    FP_global[0] += 1
            except:
                trash = 0
            try:
                if gt[j] == pred_cnn[j]:
                    TP_global[1] += 1
                else:
                    FN_global[1] += 1
                    FP_global[1] += 1
            except:
                trash = 0
            try:
                if gt[j] == pred_att[j]:
                    TP_global[2] += 1
                else:
                    FN_global[2] += 1
                    FP_global[2] += 1
            except:
                trash = 0
    
    print(TP_global,FP_global,FN_global)
    #calculate presision, recall, F1
    epsilon = 1E-12
    
    precision_lstmG = TP_global[0] / (TP_global[0] + FP_global[0] + epsilon)
    precision_cnnG = TP_global[1] / (TP_global[1] + FP_global[1] + epsilon)
    precision_attG = TP_global[2] / (TP_global[2] + FP_global[2] + epsilon)
        
    recall_lstmG = TP_global[0] / (TP_global[0] + FN_global[0] + epsilon)
    recall_cnnG = TP_global[1] / (TP_global[1] + FN_global[1] + epsilon)
    recall_attG = TP_global[2] / (TP_global[2] + FN_global[2] + epsilon)
    
    F1_lstmG = (2*precision_lstmG*recall_lstmG)/(precision_lstmG + recall_lstmG + epsilon)
    F1_cnnG = (2*precision_cnnG*recall_cnnG)/(precision_cnnG + recall_cnnG + epsilon)
    F1_attG = (2*precision_attG*recall_attG)/(precision_attG + recall_attG + epsilon)
    
    print('finished!\nassigning weights...')
    #calculate weights 
    w_lstmG = F1_lstmG
    w_cnnG = F1_cnnG
    w_attG = F1_attG
    print('finished!\nconstructing ensembling datasets...')
    # Merge saved predictions for training my ensamble. The labels remain the same.
    ensemble_weights = [w_lstmG,w_cnnG,w_attG]
    ensemble_train_x = [tr_pred_lstm.to('cpu',dtype=torch.int32),tr_pred_cnn.to('cpu',dtype=torch.int32),tr_pred_att.to('cpu',dtype=torch.int32)]
    ensemble_val_x = [val_pred_lstm[int(len(val_y)/2):].to('cpu',dtype=torch.int32),val_pred_cnn[int(len(val_y)/2):].to('cpu',dtype=torch.int32),val_pred_att[int(len(val_y)/2):].to('cpu',dtype=torch.int32)]
    print('done!\nEnd of task')
    print('LSTM F1 weight:',round(w_lstmG,3),'CNN F1 weight:',round(w_cnnG,3),'ATT F1 weight:',round(w_attG,3))
#%% Prep data for ensemble
ensemble_val_y = val_y[int(len(val_y)/2):].to('cpu',dtype=torch.int32)
#torch.save(ensemble_val_y,r'R:\classes 2020-22\Fall 2021\ensemble_val_y.pt')
#torch.save(ensemble_weights,r'R:\classes 2020-22\Fall 2021\ensemble_weights.pt')
#torch.save(ensemble_train_x,r'R:\classes 2020-22\Fall 2021\ensemble_train_x.pt')
#torch.save(ensemble_val_x,r'R:\classes 2020-22\Fall 2021\ensemble_val_x.pt')
#%% Ensemble and BLEU
# paper ensemble
    # Output predictions using ensemble weights
    # belu scores on 2nd half of validation set 
    
#!!! Since w2 > w1+w0, output of w2 will always be selected... Hence the algorithm can be commented out and a simple Transformer evaluation is performed.
paper_ensemble_predictions = []

def equal_(a,b):
    a = a[1:torch.argmax((a==2.5189e+04).to(dtype=torch.uint8))]
    b = b[1:torch.argmax((b==2.5189e+04).to(dtype=torch.uint8))]
    equal = True
    min_len = len(a)
    if min_len > len(b):
        min_len = len(b)
    
    for i in range(min_len):
        #print(vocab[a[i]],vocab[b[i]])
        if a[i] != b[i]:
            equal = False
    return equal
   
'''
for i in range(len(ensemble_val_x[0])):
    if equal_(ensemble_val_x[0][i],ensemble_val_x[1][i]) and equal_(ensemble_val_x[0][i],ensemble_val_x[2][i]):
        paper_ensemble_predictions.append(ensemble_val_x[0][i][1:torch.argmax((ensemble_val_x[0][i]==2.5189e+04).to(dtype=torch.uint8))])
    elif equal_(ensemble_val_x[0][i],ensemble_val_x[1][i]):
        if ensemble_weights[0]+ensemble_weights[1] > ensemble_weights[2]:
            paper_ensemble_predictions.append(ensemble_val_x[0][i][1:torch.argmax((ensemble_val_x[0][i]==2.5189e+04).to(dtype=torch.uint8))])
        else: paper_ensemble_predictions.append(ensemble_val_x[2][i][1:torch.argmax((ensemble_val_x[2][i]==2.5189e+04).to(dtype=torch.uint8))])
    elif equal_(ensemble_val_x[0][i],ensemble_val_x[2][i]):
        if ensemble_weights[0]+ensemble_weights[2] > ensemble_weights[1]:
            paper_ensemble_predictions.append(ensemble_val_x[0][i][1:torch.argmax((ensemble_val_x[0][i]==2.5189e+04).to(dtype=torch.uint8))])
        else: paper_ensemble_predictions.append(ensemble_val_x[1][i][1:torch.argmax((ensemble_val_x[1][i]==2.5189e+04).to(dtype=torch.uint8))])
    elif equal_(ensemble_val_x[1][i],ensemble_val_x[2][i]):
        if ensemble_weights[1]+ensemble_weights[2] > ensemble_weights[0]:
            paper_ensemble_predictions.append(ensemble_val_x[1][i][1:torch.argmax((ensemble_val_x[1][i]==2.5189e+04).to(dtype=torch.uint8))])
        else: paper_ensemble_predictions.append(ensemble_val_x[0][i][1:torch.argmax((ensemble_val_x[0][i]==2.5189e+04).to(dtype=torch.uint8))])
    else:
        if ensemble_weights[0] > ensemble_weights[1] and ensemble_weights[0] > ensemble_weights[2]:
            paper_ensemble_predictions.append(ensemble_val_x[0][i][1:torch.argmax((ensemble_val_x[0][i]==2.5189e+04).to(dtype=torch.uint8))])
        elif ensemble_weights[1] > ensemble_weights[0] and ensemble_weights[1] > ensemble_weights[2]:
            paper_ensemble_predictions.append(ensemble_val_x[1][i][1:torch.argmax((ensemble_val_x[1][i]==2.5189e+04).to(dtype=torch.uint8))])
        else: paper_ensemble_predictions.append(ensemble_val_x[2][i][1:torch.argmax((ensemble_val_x[2][i]==2.5189e+04).to(dtype=torch.uint8))])
'''       
paper_preds = [] 
ground_preds = []  
for i in range(len(ensemble_val_x[0])):
    paper_preds.append(ensemble_val_x[2][i][1:torch.argmax((ensemble_val_x[2][i]==2.5189e+04).to(dtype=torch.uint8))].tolist())
    ground_preds.append(ensemble_val_y[i][1:torch.argmax((ensemble_val_y[i]==2.5189e+04).to(dtype=torch.uint8))].tolist())
paper_preds = untokenizer(paper_preds)
ground_preds = untokenizer(ground_preds)

print(paper_preds[0].split())

a = []
b = []
for i in range(len(ensemble_val_x[0])):
    a.append(paper_preds[i].split())
    b.append(ground_preds[i].split())

score = []
for i in range(len(a)):
    score.append(sentence_bleu([b[i]],a[i]))
    
average_score = sum(score)/len(score)
print('The BELU score for the paper based method = that of the Attention model =', round(average_score,3),'         [where 1.0 = perfect score and 0.0 = worst]')
#!!! The BELU score for the paper based method = that of the Transformer model = 0.952          [where 1.0 = perfect score and 0.0 = worst]
#%% Prep data for autoo ensemble
ensemble_weights = torch.load(r'R:\classes 2020-22\Fall 2021\ensemble_weights.pt')
ensemble_val_y = torch.load(r'R:\classes 2020-22\Fall 2021\ensemble_val_y.pt')[:3781]
ensemble_train_x = torch.load(r'R:\classes 2020-22\Fall 2021\ensemble_train_x.pt')
ensemble_val_x = torch.load(r'R:\classes 2020-22\Fall 2021\ensemble_val_x.pt')
ensemble_train_y = tr_y
input_x = torch.zeros(3,63,68416)
torch.cat([ensemble_train_x[1].view(1,63,68416),ensemble_train_x[1].view(1,63,68416),ensemble_train_x[2].view(1,63,68416)], out=input_x)
input_x = input_x.T
input_xe = torch.zeros(3,63,3781)
torch.cat([ensemble_val_x[1].view(1,63,3781),ensemble_val_x[1].view(1,63,3781),ensemble_val_x[2].view(1,63,3781)], out=input_xe)
input_xe = input_xe.T
print('F1 scores:', round(ensemble_weights[0],3), round(ensemble_weights[1],3), round(ensemble_weights[2],3))
print(input_x.shape,input_xe.shape)
print(ensemble_train_y.shape,ensemble_val_y.shape)
#F1 scores: 0.081 0.857 1.0
#%% Auto Ensemble
class m2(nn.Module):
    def __init__(self,device):
        super(m2, self).__init__()
        self.device = device
        self.condense = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=int(len(vocab)/100), kernel_size=1),nn.LeakyReLU(),
                                      nn.Conv1d(in_channels=int(len(vocab)/100), out_channels=len(vocab), kernel_size=1)).to(self.device)
        
    def forward(self,x):
        outputs = torch.swapaxes(x, 1, 2).to(self.device)
        output = self.condense(outputs)
        return output
    
def trainm2(model,optimizer,criterion,x,y,xe,ye,vocab,batch=16,epochs=20,lr=0.01):
    train_loss_list = []
    eval_loss_list = []    
    for ep in trange(1,epochs+1):
        idx = torch.randperm(x.shape[0])
        x = x[idx]
        y = y[idx]
        model.train()
        temploss = 0
        for b in range(0, x.shape[0]-batch, batch):
            optimizer.zero_grad()
            t = y[b:b+batch].to(model.device)
            yp = model.forward(x[b:b+batch].to(device))
            t=t.to(dtype=torch.int64)
            loss = criterion.forward(yp, t)
            loss = criterion.forward(yp, t)
            temploss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss_list.append(temploss/int(x.shape[0]/batch))
        temploss = 0
        model.eval()
        for b in range(0, xe.shape[0]-batch, batch):
            t = ye[b:b+batch].to(model.device)
            yp = model.forward(xe[b:b+batch].to(device))
            t=t.to(dtype=torch.int64)
            loss = criterion.forward(yp, t)
            loss = criterion.forward(yp, t)
            temploss += float(loss)
        eval_loss_list.append(temploss/int(xe.shape[0]/batch))
        if ep % 2 == 0 or ep == 1:
            print('Epoch', ep, ':\n      Training loss:',
                  train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
    return train_loss_list,eval_loss_list
#%% initialize model
model1 = m2(device)
print(f'The model has {count_parameters(model1):,} trainable parameters')
a = torch.swapaxes(model1.forward(input_x[:2].to(device,dtype=torch.float)),1,2).to(dtype=torch.int)
a = hot2cold(a)
for i in range(len(a)):
    for j in range(63):
        if a[i,j] > vocab_['<unk>'] or a[i,j]<0:
            a[i,j] = vocab_['<unk>']
print(a.shape)
a = untokenizer(a)
gt = untokenizer(ensemble_train_y[:2])
for i in range(2):
    print(gt[i])
    print(a[i])
    print('\n')
#%% train
#NOTE: Finetuning took way too long so I wasnt able to use the best possible model for BELU calculation
optimizer = optim.Adam(model1.parameters(),weight_decay=1E-9,amsgrad=True)
criterion = nn.CrossEntropyLoss(ignore_index = int(len(vocab)-4),reduction='mean')
BATCH_SIZE = 64
l1,l2 = trainm2(model1,optimizer,criterion,input_x[:68000],ensemble_train_y[:68000],input_xe[:3000],ensemble_val_y[:3000],vocab=vocab,batch=BATCH_SIZE,epochs=15,lr=1E-3)
plt.plot(l1, label='Train')
plt.plot(l2, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss(Cross-Entropy)')
plt.legend()
plt.title('autoE')
plt.plot()
a = torch.swapaxes(model1.forward(input_x[:2].to(device,dtype=torch.float)),1,2).to(dtype=torch.int)
a = hot2cold(a)
for i in range(len(a)):
    for j in range(63):
        if a[i,j] > vocab_['<unk>'] or a[i,j]<0:
            a[i,j] = vocab_['<unk>']
print(a.shape)
a = untokenizer(a)
gt = untokenizer(ensemble_train_y[:2])
for i in range(2):
    print(gt[i])
    print(a[i])
    print('\n')
#%% save
#torch.save(model1.state_dict(),r'R:\classes 2020-22\Fall 2021\autoEnsemble.pth')
#%% BLEU score calculation for auto ensemble
score = []
for i in range(68446-int(68446/2)):
    a = torch.swapaxes(model1.forward(input_x[i:i+1].to(device,dtype=torch.float)),1,2).to(dtype=torch.int)
    a = hot2cold(a).flatten()
    for j in range(len(a)):
        if a[j] > vocab_['<unk>'] or a[j]<0:
            a[j] = vocab_['<unk>']
    a = untokenizer(a.view(1,63))
    gt = untokenizer([ensemble_train_y[i+int(68446/2)]])
    score.append(sentence_bleu([gt[0].split()],a[0].split()))
print('BELU score for my_model:', round(sum(score)/len(score),3))
#BELU score for my_model: 0.283