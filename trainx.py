from __future__ import print_function

import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from scipy.io.wavfile import write
from readpiano import Dataset, Testset, RandomCrop, ToTensor
#from wavenet import Wavenet
from nvwavenet import Wavenet
from transformData import mu_law_decode

# In[2]:


sampleSize = 16000  # the length of the sample size
quantization_channels = 256
sample_rate = 16000
dilations0 = [2 ** i for i in range(10)] * 3
#dilations1 = [2 ** i for i in range(13)] * 4
dilations1 = [2 ** i for i in range(10)] * 5
residualDim = 128  #
skipDim = 512
shapeoftest = 190500
songnum=45
filterSize = 3
savemusic='vsCorpus/piano{}.wav'
#savemusic0='vsCorpus/nus10xtr{}.wav'
#savemusic1='vsCorpus/nus11xtr{}.wav'
resumefile = 'model/piano2'  # name of checkpoint
lossname = 'pianoloss2.txt'  # name of loss file
continueTrain = False  # whether use checkpoint
pad = np.sum(dilations0)
field = np.sum(dilations1) + 2

lossrecord = []  # list for record loss
sampleCnt=0
#pad=0
print("receptive field", field)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use specific GPU

# In[4]:


use_cuda = torch.cuda.is_available()  # whether have available GPU
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
# device = 'cpu'
# torch.set_default_tensor_type('torch.cuda.FloatTensor') #set_default_tensor_type as cuda tensor


transform=transforms.Compose([RandomCrop(pad=field),ToTensor()])
#training_set = Dataset(np.array([6,7,8,9,11,12,14,16,26]), 'ccmixter3/',pad=field,transform=transform)
#validation_set = Testset(np.array([6,7,8,9,11,12,14,16,26]), 'ccmixter3/',pad=field)
#training_set = Dataset(np.array([0]), 'ccmixter3/' ,pad=field,transform=transform)
#validation_set = Testset(np.array([0]), 'ccmixter3/',pad=field,dilations1=dilations1,device=device)
training_set = Dataset(np.arange(6), 'ccmixter3/' ,pad=field,transform=transform)
validation_set = Testset(np.arange(6), 'ccmixter3/',pad=field,dilations1=dilations1,device=device)
loadtr = data.DataLoader(training_set, batch_size=1,shuffle=True,num_workers=0,worker_init_fn=np.random.seed)
loadval = data.DataLoader(validation_set,batch_size=1,num_workers=0)
# In[6]:

#model = Unet(skipDim, quantization_channels, residualDim,device)
model = Wavenet(field, skipDim, residualDim, dilations0,dilations1,device)
#model = nn.DataParallel(model)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
# in wavenet paper, they said crossentropyloss is far better than MSELoss
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# use adam to train

maxloss=np.zeros(50)+100
# In[7]:

start_epoch=0
if continueTrain:  # if continueTrain, the program will find the checkpoints
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resumefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))


# In[9]:


def test(epoch):  # testing data
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for iloader, y in loadval:
            iloader=iloader.item()
            #music = model.infer(queue)
            #music = model.slowInfer(queue, y, 16000 * 0.001)
            #music = model.slowInfer(queue, y, 16000 * 0.01)
            #music = model.slowInfer(queue, y, 16000 * 0.1)
            music = model.slowInfer(None, y, 16000 * 3)
            print(music[:1000])
            ans0 = mu_law_decode(music.numpy().astype('int'))

            if not os.path.exists('vsCorpus/'): os.makedirs('vsCorpus/')
            write(savemusic.format(epoch*6+iloader), sample_rate, ans0)
            print('test stored done', np.round(time.time() - start_time))



def train(epoch):
    for iloader, ytrain in loadtr:
        iloader=int(iloader)
        #startx = np.random.randint(0,sampleSize) #make results worse, biggest reason(wrong,the result is similar)
        startx = 0
        idx = np.arange(startx + field, ytrain.shape[-1] - field - sampleSize, sampleSize)
        #when sample rate is samplesize//2, the result is similar
        #np.random.shuffle(idx)
        #lens = 100
        #lens = 10
        #idx = idx[:lens]
        cnt, aveloss, aveacc = 0, 0, 0
        start_time = time.time()
        model.train()
        for i, ind in enumerate(idx):
            optimizer.zero_grad()
            target0 = ytrain[:, ind - field:ind + sampleSize - 1].to(device)
            target1 = ytrain[:, ind:ind + sampleSize].to(device)
            output = model(target0)
            a = output.max(dim=1, keepdim=True)[1].view(-1)
            b = target1.view(-1)
            assert (a.shape[0] == b.shape[0])
            aveacc += float(float(torch.sum(a.long() == b.long())) / float(a.shape[0]))
            loss = criterion(output, target1)
            loss.backward()
            optimizer.step()
            aveloss+=float(loss)
            if(float(loss) > 10):print(float(loss))
            cnt+=1
            if(cnt > 100):
                print('loss for train:{:.4f},acc:{:.4f},num{},epoch{},({:.3f} sec/step)'.format(
                    aveloss / cnt, aveacc / cnt, iloader, epoch, time.time() - start_time))
                cnt, aveloss, aveacc = 0, 0, 0
                start_time = time.time()
            lossrecord.append(float(loss))
            global sampleCnt
            sampleCnt+=1
            if sampleCnt % 10000 == 0 and sampleCnt > 0:
                for param in optimizer.param_groups:
                    param['lr'] *= 0.98

    if not os.path.exists('lossRecord/'): os.makedirs('lossRecord/')
    with open("lossRecord/" + lossname, "w") as f:
        for s in lossrecord:
            f.write(str(s) + "\n")
    if not os.path.exists('model/'): os.makedirs('model/')
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    #if(epoch % 4 == 0 and epoch != 0):
    torch.save(state, resumefile)
    print('write finish')
    print('epoch finished')


# In[ ]:

print('training...')
for epoch in range(100000):

    #if (epoch + start_epoch) % 64 == 0 and (epoch + start_epoch) > 0: test(epoch + start_epoch)
    #if (continueTrain == True and epoch == 0): test(epoch + start_epoch)
    train(epoch + start_epoch)
    test(epoch + start_epoch)
    #if (epoch + start_epoch) % 4 == 0 and (epoch + start_epoch) > 0: test(epoch + start_epoch)