import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        #if self.is_causal:
        #        padding = (int((self.kernel_size - 1) * self.dilation), 0)
        #        signal = torch.nn.functional.pad(signal, padding)
        return self.conv(signal)

class Wavenet(nn.Module):
    def __init__(self, pad, sd, rd, dilations0,dilations1,device):
        self.dilations1 = dilations1
        self.device=device
        sd = 512
        rd = 128
        self.sd = sd
        self.rd = rd
        self.init_filter=2
        self.field=np.sum(dilations1)+self.init_filter
        wd = 128
        print('sd rd:',sd,rd)
        self.wd=wd
        super(Wavenet, self).__init__()
        self.embedy = torch.nn.Embedding(256,wd)
        #self.casual = torch.nn.Conv1d(256,wd,self.init_filter)
        self.pad = pad
        self.ydcnn  = nn.ModuleList()
        self.ydense = nn.ModuleList()
        self.yskip = nn.ModuleList()

        for i, d in enumerate(self.dilations1):
            self.ydcnn.append(Conv(wd, wd*2,kernel_size=2, dilation=d, w_init_gain='tanh', is_causal=True))
            #self.ydcnn.append(Conv(rd, 2 * rd, kernel_size=2, dilation=d, w_init_gain='relu', is_causal=True))#try relu
            self.yskip.append(Conv(wd, sd,w_init_gain='relu'))
            self.ydense.append(Conv(wd, wd,w_init_gain='linear'))

        self.post1 = Conv(sd, sd, bias=False, w_init_gain='relu')
        self.post2 = Conv(sd, 256, bias=False, w_init_gain='linear')

    def forward(self, y):
        y = self.embedy(y.long())
        y = y.transpose(1, 2)

        finalout = y.size(2)-(self.field-1)

        for i, d in enumerate(self.dilations1):
            in_act = self.ydcnn[i](y)
            in_act = in_act
            t_act = torch.nn.functional.tanh(in_act[:, :self.wd, :])
            s_act = torch.nn.functional.sigmoid(in_act[:, self.wd:, :])
            acts = t_act * s_act

            res_acts = self.ydense[i](acts)

            if i == 0:
                output = self.yskip[i](acts[:,:,-finalout:])
            else:
                output = self.yskip[i](acts[:,:,-finalout:]) + output

            y = res_acts + y[:,:,d:]

        output = torch.nn.functional.relu(output, True)
        output = self.post1(output)
        output = torch.nn.functional.relu(output, True)
        output = self.post2(output)

        return output

    def infer(self,queue,l = 16000*1):
        #y = torch.randint(0, 255, (1,1)).to(self.device)
        y = torch.randint(0, 255, (1,1)).to(self.device)
        l = int(l)
        music=torch.zeros(l)
        for idx in range(l):
            y = self.embedy(y.long())
            y = y.transpose(1, 2)
            for i, d in enumerate(self.dilations1):
                y = torch.cat((queue[i],y),2)
                if(d == 1):
                    queue[i] = y[:,:,:1].clone()
                else:
                    queue[i] = torch.cat((queue[i][:, :, 1:], y[:, :, :1]), 2)
                in_act = self.ydcnn[i](y)
                t_act = torch.nn.functional.tanh(in_act[:, :self.wd, :])
                s_act = torch.nn.functional.sigmoid(in_act[:, self.wd:, :])
                acts = t_act * s_act

                res_acts = self.ydense[i](acts)

                if i == 0:
                    output = self.yskip[i](acts[:,:,-1:])
                else:
                    output = self.yskip[i](acts[:,:,-1:]) + output

                y = res_acts + y[:,:,d:]

            output = torch.nn.functional.relu(output, True)
            output = self.post1(output)
            output = torch.nn.functional.relu(output, True)
            output = self.post2(output)
            y = output.max(1, keepdim=True)[1].view(-1)[0]
            y = y.view(1,1)
            music[idx] = y.cpu()[0,0]
        return music

    def slowInfer(self,queue,input=None,l = 16000*0.01):
        l = int(l)
        label = input[:,self.field:self.field+l].clone().view(-1).to(self.device)
        input = input[:,:self.field].clone().to(self.device)

        music=torch.zeros(l)
        for idx in range(l):
            y = self.embedy(input)
            y = y.transpose(1, 2)
            for i, d in enumerate(self.dilations1):
                in_act = self.ydcnn[i](y)
                t_act = torch.nn.functional.tanh(in_act[:, :self.wd, :])
                s_act = torch.nn.functional.sigmoid(in_act[:, self.wd:, :])
                acts = t_act * s_act

                res_acts = self.ydense[i](acts)

                if i == 0:
                    output = self.yskip[i](acts[:,:,-1:])
                else:
                    output = self.yskip[i](acts[:,:,-1:]) + output

                y = res_acts + y[:,:,d:]

            output = torch.nn.functional.relu(output, True)
            output = self.post1(output)
            output = torch.nn.functional.relu(output, True)
            output = self.post2(output)
            #print(output.shape)
            #print(input.shape,output.max(1, keepdim=True)[1].shape)
            input = torch.cat((input[:,1:].long(),output.max(dim=1, keepdim=True)[1].view(1,1).long()),1)
            music[idx] = input.cpu()[0,-1]
        print(float(float(torch.sum(music.long() == label.long())) / float(music.shape[0])))
        return music

