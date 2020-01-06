import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
import pandas as pd
from tqdm import tqdm
import itertools
import io
from sklearn.manifold import TSNE
import math
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
#from gensim.models import KeyedVectors
# ==============================================================================
# set random seed
torch.manual_seed(42)
np.random.seed(42)
# ==============================================================================
# check gpu

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    device = 'cpu'
else:
    print("CUDA is available!\nNumber of gpu's detected: ", torch.cuda.device_count(), '\nTraining on GPU ...')
    device = 'cuda:0'
# ==============================================================================
# load data

# load
data = pd.read_csv('new_hate_data.csv')  # 'hate_dataset.csv')
print('LEN:', len(data))
# data = data.dropna(axis=0, how='any')  # drop empty rows/cells

# drop content/entries=nan rows
for idx, r in data.iterrows():
    if pd.isnull(r['content']):
        data = data.drop(idx)

print('LEN:', len(data))
print(data['labels'].value_counts())

data = data[:7200]

def scale_data(data):
    """split data in equal parts for every label"""

    df1 = data[data['labels'] == 1]   # 'binary label'

    df0 = data[data['labels'] == 0]

    idxs = len(df0)-len(df1)

    while (len(df0)-idxs+len(df1)) % 1000 != 0:
        idxs -= 1  # add to class 0 untill len class samples is divisble by 100
    else:
        idx_lst = list(range(idxs))

        df0 = df0.drop(df0.index[idx_lst])

        merger = pd.concat([df0, df1])

        merger = merger.sample(frac=1).reset_index(drop=True)  # shuffle rows inplace

    return merger


#data = scale_data(data)
print('kakakka', len(data))

# data = data[]
X = data['content']  # 'entries'
y = np.array(data['labels'])  # 'binary labels'
# y2 = data['scale label']

# check
print('DATA CHECK, LEN, SHAPE, TYPE:')
print(X[:20])
print(y[:20])
print(X.shape)
print(y.shape)
print(len(X))
print(len(y))
print(type(X))
print(type(y))
# ==============================================================================
# tsv

# ==============================================================================
# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)  # set lower case False for cased model
tokenized_text = [tokenizer.tokenize(sent) for sent in X]

print('TOKENIZED: ', tokenized_text[:10])

masked_idx = 6

seq_len = 100

for sen in tokenized_text:
    trunc_txt = [i[:seq_len] if len(i) > seq_len else i for i in tokenized_text]

# input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(sent) for sent in trunc_txt],
#                          maxlen=150, dtype="long", truncating="post", padding="post")

input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in trunc_txt]
X = pad_sequences(input_ids, maxlen=50, dtype="long",
                          truncating="post", padding="post")

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in X:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
# ==============================================================================
# preprocess

X_prep = []
all_words = []

for i, s in enumerate(X):
    #print(i, s)
    s = s.lower()
    s = ''.join([c for c in s if c not in punctuation])
    s = s.split(' ')
    X_prep.append(s)
    [all_words.append(w) for w in s]

word_count = Counter(all_words)
vocab = word_count.keys()  # get sorted words from Counter
count_dict = {w: i for (i, w) in enumerate(vocab, 1)}  # enumerate ipv counter loop

coded_X = []

for s in X_prep:
    coded_X.append([count_dict[w] for w in s])  # use split() directly to avoid 'if in dict'/ignore spaces

# check prepped data and len
print(coded_X[:20])
print(len(coded_X))
# ==============================================================================
# pad and truncate data to len of 200

def pad_feats(coded_lst, seq_len):
    """pad or truncate items in given lst to len of 200. Return np array"""

    # feats = np.zeros((len(coded_lst), seq_len), dtype=int)  # make empty np arrays with only zeros first and fill up next
    # for i, row in enumerate(coded_lst):
    #    feats[i, -len(row):] = np.array(row)[:seq_len]
    coded_lst = [i[:seq_len] if len(i) > seq_len else i for i in coded_lst]
    coded_lst = [((seq_len-len(i))*[0])+i if len(i) < seq_len else i for i in coded_lst]

    return np.array(coded_lst)

input_X = np.array(pad_feats(coded_X, 200))
# ==============================================================================
# split train val test data

p = 0.1
l = int(len(X_prep)*p)
l_test = int(l*p)

print(l)
print(l_test)

X_train = input_X[:7000]  #input_X[:7000]  # input_X[:-l]
X_val = input_X[7000:7160]  #input_X[7000:7160]  # input_X[-l:-l_test]
X_test = input_X[7160:]  #input_X[7160:]  # input_X[-l_test:]
y_train = y[:7000]  #y[:7000]  # y[:-l]
y_val = y[7000:7160]  #y[7000:7160]  # y[-l:-l_test]
y_test = y[7160:]  #y[7160:]  # y[-l_test:]

# test len
print('X')
print(len(X_train))
print(len(X_val))
print(len(X_test))
print('y')
print(len(y_train))
print(len(y_val))
print(len(y_test))

print(X_train[:20])
print(y_train[:20])

# create tensors
train_data = TensorDataset(torch.from_numpy(X_train),
                           torch.from_numpy(y_train))

print('kakakakakakkkkkkkkkkkkkakakakkakaaaaaaaaaaaaaaaa')
print(train_data)
valid_data = TensorDataset(torch.from_numpy(X_val),
                           torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test),
                          torch.from_numpy(y_test))

# create dataloader generators
bs = 20  # len train/val/test moet deelbaar hierdoor zijn!!!! altijd checken!!!#print('pipipip', t[0])
    #print('lolollo', t[1])
train_loader = DataLoader(train_data, shuffle=True, batch_size=bs)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=bs)
test_loader = DataLoader(test_data, shuffle=True, batch_size=bs)

# test print shapes and sizes
dataiter = next(iter(train_loader))
x, y = dataiter

print(x)
print(y)
print(x.shape)
print(y.shape)

print(len(train_loader))
print(len(valid_loader))
print(len(test_loader))

# ==============================================================================
# define network class

class SentRNN(nn.Module):
    """sentiment rnn network for imdb reviews"""

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim,
                 hidden_dim2, n_layers, drop_prob=0.5):

        """initialize model, set up layers"""
        super(SentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # as many rows as there are word ints (encoded words) * embedding dim = cols (number of batches???)
        # lstm takes in embedding_dim as rows (400, batch size???) with word int count. produces hidden state and hidden size (choose your self)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,  # choose n_layers (2/3)
                            dropout=drop_prob, batch_first=True)  # batch firs because of dataloader

        # pass lstm output to dropout layer
        self.dropout = nn.Dropout(0.3)

        # pass lstm to fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim2)  # nn.Linear(, hidden_dim2) gives better results??
        self.fc2 = nn.Linear(hidden_dim2, output_size)
        # sigmoid to convert output between 0 and 1
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """make forward pass with input and hiddenstate"""

        # get batch_Size
        x = x.long()  # convert to long integer (infinite precision)
        # pass x to embedding
        batch_size = x.size(0)
        embeds = self.embedding(x)
        # pass embedding through lstm with hidden state, gives lstm output and new hidden state
        # embedding is more efficient alternative for oen hot encoding (not 26 chars, but 1000's of words)
        output, hidden = self.lstm(embeds, hidden)

        output = output.contiguous().view(-1, self.hidden_dim)

        output = self.dropout(output)

        out = self.fc(output)  # .view(bs, -1)
        #print('OUT1:', out.shape)
        out = self.fc2(out)
        #print('OUT2:', out.shape)
        # sigmoid activation function
        out = self.sig(output)  # deactivate
        #print('OUT3:', out.shape)
        #print('kakakakak', out)
        #out = F.log_softmax(sig_out, dim=1)

        # reshape so batch_size is first
        out = out.view(batch_size, -1)  # deactivate
        #print('OUT4:', out.shape)
        out = out[:, -1]  # get last batch of labels (all rows plus last col) = output  # deactivat
        #print('OUT5:', out.shape)

        return out, hidden

    def init_hidden(self, batch_size):
        """initialize hidden state"""
        weight = next(self.parameters()).data

        # hidden state and cell state is tuple of values, each of these is size layers, batch_size, hidden_dim. Hidden weights are all zeros.
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self .n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden

# ==============================================================================
# initializing the model

vocab_size = len(count_dict)+1  # +1 for the added 0 token thdr manhattenrough padding
output_size = 1  # == 1 proba outcome
embedding_dim = 400  # between 200 and 500 should work, = random, not the total of batches!!!
hidden_dim = 128
hidden_dim2 = 256
n_layers = 2

net = SentRNN(vocab_size, output_size, embedding_dim, hidden_dim, hidden_dim2, n_layers)

print(net)
net.to(device)
# ==============================================================================
# train

valid_loss = []
valid_acc = []
training_loss = []
training_acc = []

def train_val(net, train_data, val_data, epochs, batch_size, lr,
              clip, print_every):
    """train rnn lstm sentiment analysis network with imbd review text data"""

    criterion = nn.BCELoss()  # CrossEntropyLoss()  #BCELoss()  # binary cross entropy loss aplies cross entropy on 1 value between 0 and 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    step_counter = 0
    min_valid_loss = np.Inf
    new_valid_loss = 0
    max_valid_acc = 0

    for e in range(epochs):

        train_losses = []
        train_accs = []

        print('training...')
        print('\nepoch: {} out of {}'.format(e+1, epochs), '\n',)

        h = net.init_hidden(batch_size)
        val_counter = 0

        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)

            step_counter += 1
            net.train()

            # get hidden data from hidden/cell tuple
            h = tuple([each.data for each in h])

            net.zero_grad()

            # pass x/input/integer words and hidden state
            output, h = net(inputs, h)

            #print('pipipipi', output)
            #print(len(output))
            #print(len(labels))
            #print(len(output.squeeze()))
            #print(len(labels.float()))

            # calculate loss
            loss = criterion(output.squeeze(), labels.float())  # 350 = padding  # verwijder squeze en float
            loss.backward()

            # clip grads to prevent exploding
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            # optimize grads
            optimizer.step()

            matches = [torch.round(i) == j for i, j in zip(output, labels)]  # .round round to nearest full integer
            acc = matches.count(True)/len(matches)  #pred/true

            train_losses.append(loss.item())
            train_accs.append(acc)

            # validate

            if step_counter % print_every == 0:

                print('validating...')
                val_counter += 1

                with torch.no_grad():

                    net.eval()

                    val_h = net.init_hidden(batch_size)

                    val_losses = []
                    val_accs = []

                    for inputs, labels in tqdm(val_data):

                        inputs, labels = inputs.to(device), labels.to(device)

                        val_h = tuple([each.data for each in val_h])
                        output, val_h = net(inputs, val_h)

                        val_loss = criterion(output.squeeze(), labels.float())
                        val_losses.append(val_loss.item())
                        new_valid_loss = np.mean(val_losses)
                        matches = [torch.round(i) == j for i, j in zip(output, labels)]
                        acc = matches.count(True)/len(matches)
                        val_accs.append(acc)

                    valid_loss.append(np.mean(val_losses))
                    valid_acc.append(np.mean(val_accs))
                    training_loss.append(np.mean(train_losses))
                    training_acc.append(np.mean(train_accs))

                    print('\nvalidation: {} in epoch: {}'.format(val_counter, e+1),
                          '\ntrain loss: {}'.format(loss.item()),
                          '\nvalidation loss: {}'.format(np.mean(val_losses)),
                          '\ntrain accuracy:', acc,
                          '\nvalidation accuracy: ', np.mean(val_accs))


                    # save model with smallest validation loss

                    #if new_valid_loss <= min_valid_loss:
                    if np.mean(val_accs) >= max_valid_acc:

                        #print("\nSmallest validation loss detected. Saving model...")
                        print("\nMax validation accuracy detected. Saving model...")

                        #min_valid_loss = np.mean(val_losses)
                        max_valid_acc = np.mean(val_accs)

                        model_name = 'sent_lstm3'

                        checkpoint = {'hidden_dim': net.hidden_dim,
                                      'hidden_dim2': net.hidden_dim2,  # fill dict with self. atrributes or values parameters class
                                      'embedding_dim': embedding_dim,
                                      'vocab_size': vocab_size,
                                      'n_layers': net.n_layers,
                                      'output_size': net.output_size,
                                      'state_dict': net.state_dict()
                                      }

                        with open(model_name, 'wb') as f:
                            torch.save(checkpoint, f)


# train and val
train_val(net=net, train_data=train_loader, val_data=valid_loader, epochs=3,
          batch_size=bs, lr=0.001, clip=5, print_every=20)

# ==============================================================================
# plot and visualize

# losses
def plt_loss(train, valid):

    plt.clf()
    plt.plot(train, label='training loss')
    plt.plot(valid, label='validation loss')
    plt.title('Loss at end of each epoch')
    plt.legend()
    plt.show()

plt_loss(training_loss, valid_loss)

# accuracy
def plt_acc(train, valid):

    plt.clf()
    plt.plot(train, label='training acc')
    plt.plot(valid, label='validation acc')
    plt.title('Accuracy at end of each epoch')
    plt.legend()
    plt.show()

plt_acc(training_acc, valid_acc)

# ==============================================================================
# load model

with open('sent_lstm3', 'rb') as f:
    checkpoint = torch.load(f)

loaded = SentRNN(embedding_dim=checkpoint['embedding_dim'], hidden_dim=checkpoint['hidden_dim'],
                 n_layers=checkpoint['n_layers'], output_size=checkpoint['output_size'],
                 hidden_dim2=checkpoint['hidden_dim2'], vocab_size=checkpoint['vocab_size'])


loaded.load_state_dict(checkpoint['state_dict'])
# ==============================================================================
# test net

def test(net, test_data, bs, f, lr=0.001,):
    """test sentiment rnn data"""

    criterion = nn.BCELoss()  # binary cross entropy loss aplies cross entropy on 1 value between 0 and 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    test_losses = []
    num_correct = 0

    # initialize hidden state
    h = net.init_hidden(bs)

    net.to(device)
    # set to evaluation state
    net.eval()

    for inputs, labels in test_data:

        sentences = []
        for tensor in inputs:
            sen = [[k for k, v in count_dict.items() if count_dict[k] == i] for i in tensor if i != 0]
            sen = (' ').join(list(itertools.chain(*sen)))
            sentences.append(sen)

        # for s in sentences:
        #    print(s)

        inputs, labels = inputs.to(device), labels.to(device)

        # get hidden part of cell state
        h = tuple([each.data for each in h])

        # give input and h to net and get output and new h
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output proba to 0 or 1
        pred = torch.round(output.squeeze())

        # compare to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # print stats
    preds = [int(i) for i in pred.cpu().detach().numpy()]
    labels = labels.view_as(pred).cpu().numpy()
    # zipperd = list(zip(preds, labels))
    data = {'instances': sentences, 'true': labels, 'pred': preds}
    df = pd.DataFrame(data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    print(f'test loss: {np.mean(test_losses)}')
    test_acc = num_correct/len(test_data.dataset)
    print(f'test accuracy: {test_acc}')

    return np.mean(test_losses), test_acc

def cross_val(test_X, test_y, f, size, bs):  # size = number of batches = slice, f=n_folds

    accs = []
    losses = []

    for i in tqdm(range(f)):

        random_start = np.random.randint(len(test_X)-size)  # take random validation slice
        cross_X, cross_y = (test_X[random_start:random_start+size],
                            test_y[random_start:random_start+size])
        cross_data = TensorDataset(torch.from_numpy(cross_X),
                                   torch.from_numpy(cross_y))
        cross_loader = DataLoader(cross_data, shuffle=True, batch_size=bs)
        cross_loss, cross_acc = test(net=loaded, test_data=cross_loader, bs=bs, f=i+1)
        losses.append(cross_loss)
        accs.append(cross_acc)


    print('\n', f, 'fold cross validation gives:\nmean losses=', np.mean(losses),
          '\nmean accuracy=', np.mean(accs))


#test(net=loaded, test_data=test_loader, bs=bs)
cross_val(X_test, y_test, 5, bs, bs)  # cross val slice size has to be divisble by batch size
# ==============================================================================
# predict

def preprocess(txt, seq_len=200):
    """preprocess txt for sentiment analysis in rnn/lstm"""

    # lower case all
    lower_txt = txt.lower()
    # get rid of punctuation
    just_txt = ''.join([c for c in lower_txt if c not in punctuation])
    # splittin/listing by enters
    # enter_split = just_txt.split('\n')
    # join again to strng
    # just_txt = ''.join(enter_split)
    # split on spaces
    words = just_txt.split(' ')
    coded_txt = []
    coded_txt.append([count_dict[w] for w in txt.split() if w in count_dict.keys()])
    coded_feats = pad_feats(coded_txt, seq_len)

    return torch.from_numpy(coded_feats)  # make tensor

def predict(net, txt, batch_size=1):

    net.to(device)
    net.eval()

    # preprocess/tokenize/tensorize:
    txt_tensor = preprocess(txt)
    txt_tensor = txt_tensor.to(device)

    batch_size = txt_tensor.size(0)

    # set hidden state
    h = net.init_hidden(batch_size)

    output, h = net(txt_tensor, h)

    # print prediction proba
    print('Prediction proba: {}'.format(output.item()))

    # convert output probas to predicted class 0/1
    pred = torch.round(output.squeeze())

    if(pred.item()==1):
        print('Positive review detected! Value: 0')
    else:
        print('Negative review detected! Value: 1')


    return output

# prdict
with open('clerks.txt', 'r') as f:
    rev = f.read()

# print(preprocess(rev))
p#redict(loaded, rev)
