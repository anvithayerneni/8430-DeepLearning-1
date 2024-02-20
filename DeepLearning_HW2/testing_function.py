import sys
import torch
import json
from torch.utils.data import DataLoader
from Rnn4 import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from bleu_eval import BLEU
import pickle

model = torch.load('SavedModel/model0.h5', map_location=lambda storage, loc: storage)
model = model.cuda()
filepath = '/home/ayernen/assignment2/MLDS_hw2_1_data/testing_data/feat'
print(sys.argv[1])
dataset = test_data('{}'.format(sys.argv[1]))
test_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

with open('i2w.pickle', 'rb') as h:
    i2w = pickle.load(h)

test1 = test(test_loader, model, i2w)

with open(sys.argv[2], 'w') as f:
    for i, s in test1:
        f.write('{},{}\n'.format(i, s))


test = json.load(open('/home/ayernen/assigment2/MLDS_hw2_1_data/testing_label.json'))
op = sys.argv[2]
arr = {}
bleu=[]
with open(op,'r') as f:
    for l in f:
        l = l.rstrip()
        c = l.index(',')
        tid = l[:c]
        name = l[c+1:]
        arr[tid] = name

for i in test:
    b_score = []
    captions = [x.rstrip('.') for x in i['caption']]
    b_score.append(BLEU(arr[i['id']],captions,True))
    bleu.append(b_score[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))