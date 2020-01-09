import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from tqdm import tqdm
import time
import logging
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
import warnings
import sys
# ==============================================================================
# extra settings

# ignore harmless warnings
warnings.filterwarnings("ignore")

# set seed
torch.manual_seed(42)
np.random.seed(42)

# set print options
pd.set_option('display.max_rows', 5000)  # enlarge number if stil does not show all cols or rows
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)
np.set_printoptions(threshold=sys.maxsize)
# ==============================================================================
# check gpufrom transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    device = 'cpu'
else:
    print(f"CUDA is available!\nNumber of gpu's detected: {torch.cuda.device_count()}")
    print(f'Name of device: {torch.cuda.get_device_name(0)}\nTraining on GPU ...')
    device = 'cuda:0'
# ==============================================================================
# load data

# load
data = pd.read_csv('new_hate_data.csv')  # 'hate_dataset.csv')
print('LEN:', len(data))
# data = data.dropna(axis=0, how='any')  # drop empty rows/cells

# drop content/entries=nan rowsconfig = BertConfig.from_pretrained('bert-base-uncased')
for idx, r in data.iterrows():
    if pd.isnull(r['content']):
        data = data.drop(idx)

print('LEN:', len(data))
print(data['labels'].value_counts())

data = data.iloc[:7200, :]

data2 = pd.DataFrame()

data2['content'] = data['content']
data2['labels'] = data['labels']
data = data2

print(f'DF: {data.head()}')
# ==============================================================================
# configure transformer

config = BertConfig.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)  # set lower case False for cased model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

# get labels ready for Bert

config.num_labels = len(data['labels'])

print(config.num_labels)
# ==============================================================================
# feature preparation


def prepare_features(seq_1, max_seq_length=300, zero_pad=False,
                     include_CLS_token=True, include_SEP_token=True):

    # Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    # Truncate upto given max len
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # Initialize Tokens, cls=begin token of sequence
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)

    # add tokenized txt
    for token in tokens_a:
        tokens.append(token)

    # add ending token SEP
    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    # convert to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Input Mask
    input_mask = [1] * len(input_ids)

    # Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

    return torch.tensor(input_ids).unsqueeze(0), input_mask
# ==============================================================================
# splt data
train_data = data.iloc[:7000, :]
train_data = train_data.reset_index()
test_data = data.iloc[7000:, :]
test_data = test_data.reset_index()  # to work in transform class

print('SPLIT')
print(train_data.head())
print(test_data.head())
# ==============================================================================
# encode data

class Transform_feats(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.content[index]
        y = self.data.labels[index]
        X, _  = prepare_features(utterance)
        #y = label_to_ix[self.data.labels[index]]

        return X, y

    def __len__(self):

        return self.len


training_set = Transform_feats(train_data)
testing_set = Transform_feats(test_data)

#for i in testing_set:
#    print(f'testing {i}')

# getitem__(5000) == X and y row 5000 in tensor
# getitem__(5000)[0] == X row 5000 ([1] == y)
print('training_set:', training_set.__getitem__(5000)[0])

# put encoded X in BERT model
print(model(training_set.__getitem__(5000)[0]))
print('test_set:', training_set.__getitem__(1)[0])
# ==============================================================================
# create data loaders

train_loader = DataLoader(training_set, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
test_loader = DataLoader(testing_set, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
# ===============================================================================
# set model

model = model.to(device)
print(model)

# number of params
total_param  = []
for p in model.parameters():
    total_param.append(int(p.numel()))

print(sum(total_param))
# ==============================================================================
# train

def train_val(model, train_data, val_data, epochs, lr,
              clip, print_every):
    """train rnn lstm sentiment analysis network with imbd review text data"""

    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss()  #BCELoss()  # binary cross entropy loss aplies cross entropy on 1 value between 0 and 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_counter = 0
    min_test_loss = np.Inf
    new_test_loss = 0
    max_test_acc = 0
    start_time = time.time()

    for e in tqdm(range(epochs)):

        train_losses = []
        train_correct = 0
        train_total = 0
        #train_accs = []
        test_losses = []
        test_correct = 0
        test_total = 0

        print('training...')
        print('\nepoch: {} out of {}'.format(e+1, epochs), '\n',)

        # h = net.init_hidden(batch_size)
        val_counter = 0

        for X_train, y_train in tqdm(train_data):
            X_train, y_train = X_train.to(device), y_train.to(device)
            X_train = X_train.squeeze(0)

            step_counter += 1
            model.train()

            # get hidden data from hidden/cell tuple
            # h = tuple([each.data for each in h])

            model.zero_grad() # safer to call model. ipv optimizer.zero_grad() to be sure you set all grads to zero

            # pass x/input/integer words and hidden state
            output = model.forward(X_train)[0]  # explicitly need .forward??
            #print(f'OUTPUT: {output}')
            #print(f'SQUEEZE: {output.squeeze()}')

            _, predicted = torch.max(output, 1)
            #print(f'PREDICTED {predicted}, {_}, len, {len(predicted)}')
            #print(f'PREDICTED {predicted.item()}, {_}')
            #print(f'TRUE: {y_train}')

            # calculate loss
            loss = criterion(output, y_train)  # 350 = padding  # verwijder squeze en float
            loss.backward()
            train_losses.append(loss.item())
            # clip grads to prevent exploding
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            # optimize grads
            optimizer.step()

            train_correct += (predicted.cpu() == y_train.cpu()).sum()
            train_total += len(predicted)
            #print(f'TRAIN_CORRECT: {train_correct}')  # .round round to nearest full integer
            #acc = correct.count(True)/len(matches)  #pred/true
            #train_accs.append(acc)

            # validate

            if step_counter % print_every == 0:

                print('\nVALIDATING DATA...')
                val_counter += 1

                with torch.no_grad():

                    model.eval()

                    # val_h = net.init_hidden(batch_size)

                    #loss_list = []
                    #val_accs = []

                    for X_test, y_test in val_data:
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        X_test = X_test.squeeze(0)


                        #val_h = tuple([each.data for each in val_h])
                        output = model.forward(X_test)[0]
                        _, predicted = torch.max(output.data, 1)  # why .data here???

                        test_loss = criterion(output, y_test)
                        test_losses.append(test_loss.item())
                        test_correct += (predicted.cpu() == y_test.cpu()).sum()
                        test_total += len(predicted)

                        # correct = [torch.round(i) == j for i, j in zip(output, y_test)]
                        # acc = correct.count(True)/len(matches)
                        # val_accs.append(acc)

                    #test_losses.append(np.mean(loss_list))
                    #valid_acc.append(np.mean(val_accs))
                    #train_losses.append(np.mean(train_losses))
                    #training_acc.append(np.mean(train_accs))

                    print('\ntest: {} in epoch: {}'.format(val_counter, e+1),
                          '\ntrain loss: {}'.format(np.mean(train_losses)),
                          '\nvalidation loss: {}'.format(np.mean(test_losses)),
                          '\ntrain accuracy:', train_correct.numpy()/train_total,
                          '\nvalidation accuracy: ', test_correct.numpy()/test_total)


                    # save model with smallest validation loss
                    new_test_loss = np.mean(test_losses)
                    #if new_valid_loss <= min_valid_loss:
                    if test_correct.numpy()/test_total >= max_test_acc:

                        #print("\nSmallest validation loss detected. Saving model...")
                        print("\nMax validation accuracy detected. Saving model...")

                        #min_valid_loss = np.mean(val_losses)
                        max_test_acc = np.mean(test_correct.numpy()/test_total)

                        model_name = 'bert_tryout'
                        torch.save(model.state_dict(), model_name)

    duration = time.time() - start_time
    print(f'Training and testing duration = {duration/60} minutes')


# train and val
train_val(model=model, train_data=train_loader, val_data=test_loader, epochs=3,
          lr=0.00001, clip=5, print_every=200)

# ==============================================================================
# plot and visualize
