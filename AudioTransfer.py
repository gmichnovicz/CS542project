import tensorflow as tf
import numpy as np 
import scipy.io.wavfile as wav 
import librosa
import argparse 
import time                       
import os
import matplotlib.pyplot as plt
import scipy.signal as signal


class AudioSignal():
    def __init__(self,content,style):
        self.contentFile = content
        self.styleFile = style

    def createSpectrogram(self):
        # FsCont, yCont = wav.read(self.contentFile)
        # FsSty, ySty = wav.read(self.styleFile)

        # self.num_samples = yCont.shape[0]
        # try:
        #     self.num_channels = yCont.shape[1]
        # except:
        #     self.num_channels = 1
        

        # self.contentSR = FsCont
        # self.styleSR = FsSty
    
        # fC, tC, ZxxContent = signal.stft(yCont,fs=FsCont, nfft=2048)
        # fS, tS, ZxxStyle = signal.stft(ySty,fs=FsSty, nfft=2048) #potentially change fs to FsCont
        
        # preC1 = np.angle(ZxxContent)
        # preC2 = np.log1p(np.abs(preC1))
        
        # preS1 = np.angle(ZxxStyle)
        # preS2 = np.log1p(np.abs(preS1))

        # self.contentSpec = preC2
        # self.styleSpec = preS2[:self.num_channels, :self.num_samples]
        N_FFT = 2048

        yCont, FScont = librosa.load(self.contentFile)
        ySty, FSsty = librosa.load(self.styleFile)

        contS = librosa.stft(yCont,N_FFT)
        styS = librosa.stft(ySty,N_FFT)

        contS = np.log1p(np.abs(contS))
        styS = np.log1p(np.abs(styS))

        self.num_channels = contS.shape[0]
        self.num_samples = contS.shape[1]

        self.contentSR = FScont
        self.styleSR = FSsty

        self.contentSpec = contS
        self.styleSpec = styS[:self.num_channels, :self.num_samples]



    def initSignal(self,typ):
        if typ == 'content':
            return content
        if typ == 'random':
            noise = np.random.uniform(-1.,1., size=self.contentSpec.shape)
            f, t, ZxxNoise = signal.stft(noise,fs=self.contentSR, nfft=2048)
            ZxxContent = np.angle(ZxxContent)
            ZxxContent = np.log1p(np.abs(ZxxContent))

        self.initialSignal = ZxxContent


class CNN():
    def __init__(self,audioModel):
        self.weights = 'imagenet-vgg-verydeep-19.mat'
        self.inputSpec = audioModel.contentSpec
        self.audioModel = audioModel

        self.alpha = 5e0
        self.beta = 1e4
        self.theta = 1e-3

       

    def buildModel(self,num_channels, num_samples):
        #get pre-trained vgg-layer weights
        raw_vgg_net = scipy.io.loadmat(self.weights)
        vgg_layers = raw_vgg_net['layers'][0]
       
        #h,w = self.inputSpec.shape
        net = {}
        net['input'] = tf.Variable(np.zeros((1,1,num_samples,num_channels), dtype=np.float32))

        #here using average pooling
        #layer 1
        net['conv1_1'] = tf.nn.conv2d(net['input'], W=vgg_layers[0][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[0][0][0][2][0][1]
        net['relu1_1'] = tf.nn.relu(net['conv1_1'] + tf.constant(np.reshape(bias,(bias.size))))
 
        net['conv1_2'] = tf.nn.conv2d(net['relu1_1'], W=vgg_layers[2][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[2][0][0][2][0][1]
        net['relu1_2'] = tf.nn.relu(net['conv1_2'] + tf.constant(np.reshape(bias,(bias.size))))

        #pool 1
        net['pool1'] = tf.nn.avg_pool(net['relu1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #layer 2
        net['conv2_1'] = tf.nn.conv2d(net['pool1'], W=vgg_layers[5][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[5][0][0][2][0][1]
        net['relu2_1'] = tf.nn.relu(net['conv2_1'] + tf.constant(np.reshape(bias,(bias.size))))
 
        net['conv2_2'] = tf.nn.conv2d(net['relu2_1'], W=vgg_layers[7][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[7][0][0][2][0][1]
        net['relu2_2'] = tf.nn.relu(net['conv2_2'] + tf.constant(np.reshape(bias,(bias.size))))

        #pool 2
        net['pool2'] = tf.nn.avg_pool(net['relu2_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #layer 3
        net['conv3_1'] = tf.nn.conv2d(net['pool2'],W=vgg_layers[10][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[10][0][0][2][0][1]
        net['relu3_1'] = tf.nn.relu(net['conv3_1'] + tf.constant(np.reshape(bias,(bias.size))))

        net['conv3_2'] = tf.nn.conv2d(net['relu3_1'],W=vgg_layers[12][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[12][0][0][2][0][1]
        net['relu3_2'] = tf.nn.relu(net['conv3_2'] + tf.constant(np.reshape(bias,(bias.size))))
        
        net['conv3_3'] = tf.nn.conv2d(net['relu3_2'],W=vgg_layers[14][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[14][0][0][2][0][1]
        net['relu3_3'] = tf.nn.relu(net['conv3_3'] + tf.constant(np.reshape(bias,(bias.size))))
        
        net['conv3_4'] = tf.nn.conv2d(net['relu3_3'],W=vgg_layers[16][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[16][0][0][2][0][1]
        net['relu3_4'] = tf.nn.relu(net['conv3_4'] + tf.constant(np.reshape(bias,(bias.size))))
        
        #pool 3
        net['pool3'] = tf.nn.avg_pool(net['relu3_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #layer 4
        net['conv4_1'] = tf.nn.conv2d(net['pool3'],W=vgg_layers[19][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[19][0][0][2][0][1]
        net['relu4_1'] = tf.nn.relu(net['conv4_1'] + tf.constant(np.reshape(bias,(bias.size))))

        net['conv4_2'] = tf.nn.conv2d(net['relu4_1'],W=vgg_layers[21][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[21][0][0][2][0][1]
        net['relu4_2'] = tf.nn.relu(net['conv4_2'] + tf.constant(np.reshape(bias,(bias.size))))
        
        net['conv4_3'] = tf.nn.conv2d(net['relu4_2'],W=vgg_layers[23][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[23][0][0][2][0][1]
        net['relu4_3'] = tf.nn.relu(net['conv4_3'] + tf.constant(np.reshape(bias,(bias.size))))
        
        net['conv4_4'] = tf.nn.conv2d(net['relu4_3'],W=vgg_layers[25][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[25][0][0][2][0][1]
        net['relu4_4'] = tf.nn.relu(net['conv4_4'] + tf.constant(np.reshape(bias,(bias.size))))
        
        #pool 4
        net['pool4'] = tf.nn.avg_pool(net['relu4_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


        #layer 5
        net['conv5_1'] = tf.nn.conv2d(net['pool4'],W=vgg_layers[28][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[28][0][0][2][0][1]
        net['relu5_1'] = tf.nn.relu(net['conv5_1'] + tf.constant(np.reshape(bias,(bias.size))))

        net['conv5_2'] = tf.nn.conv2d(net['relu5_1'],W=vgg_layers[30][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[30][0][0][2][0][1]
        net['relu5_2'] = tf.nn.relu(net['conv5_2'] + tf.constant(np.reshape(bias,(bias.size))))
        
        net['conv5_3'] = tf.nn.conv2d(net['relu5_2'],W=vgg_layers[32][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[32][0][0][2][0][1]
        net['relu5_3'] = tf.nn.relu(net['conv5_3'] + tf.constant(np.reshape(bias,(bias.size))))
        
        net['conv5_4'] = tf.nn.conv2d(net['relu5_3'],W=vgg_layers[34][0][0][2][0][0], strides=[1, 1, 1, 1], padding='SAME')
        bias =  vgg_layers[34][0][0][2][0][1]
        net['relu5_4'] = tf.nn.relu(net['conv5_4'] + tf.constant(np.reshape(bias,(bias.size))))
        
        #pool 5
        net['pool5'] = tf.nn.avg_pool(net['relu5_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.net = net
        return net
    
        
    @staticmethod
    def content_layer_loss(p, x):
        _, h, w, d = p.get_shape()
        M = h.value * w.value
        N = d.value
        K = 1. / (2. * N**0.5 * M**0.5)
        #K = 1. / (N * M)
        #K = 1. / 2.
        loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

    @staticmethod
    def style_layer_loss(a, x):
        _, h, w, d = a.get_shape()
        M = h.value * w.value
        N = d.value
        A = gram_matrix(a, M, N)
        G = gram_matrix(x, M, N)
        loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

    @staticmethod 
    def gram_matrix(x, area, depth):
        F = tf.reshape(x, (area, depth))
        G = tf.matmul(tf.transpose(F), F)
        return G


    def styleLoss(self,sess):
        styleSpec = self.audioModel.styleSpec
        net = self.net

        sess.run(net['input'].assign(styleSpec))
        styleloss = 0.
        styleLayersVGG = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        styleLayerWeights = [0.2, 0.2, 0.2, 0.2, 0.2] #equal weighting to all 5

        for layer, weight in zip(styleLayersVGG, styleLayerWeights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            styleloss += style_layer_loss(a, x) * weight

        styleloss /= float(len(styleLayersVGG))
        return styleloss

    def contentLoss(self,sess):
        contentSpec = self.audioModel.contentSpec
        net = self.net
        sess.run(net['input'].assign(contentSpec))     
        
        #content layers is conv4_2
        contentLayer = 'conv4_2'
        contentLayerWeight = 1.0
        
        p = sess.run(net[contentLayer])
        x = net[contentLayer]
        p = tf.convert_to_tensor(p)
        content_loss = content_layer_loss(p, x) * contentLayerWeight
        
        return content_loss

    def minimizeLBGFS(self,sess,optimizer):
        initSpec = self.audioModel.initSignal

        initOp = tf.global_variables_initializer()
        sess.run(initOp)
        sess.run(self.net['input'].assign(initSpec))
        optimizer.minimize(sess)

    def convertOutput(self,result):
        N_FFT=2048
        output = np.zeros_like(self.audioModel.contentSpec)
        output[:self.audioModel.num_channels,:] = np.exp(result[0,0].T) - 1

        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
        for i in range(500):
            S = output * np.exp(1j*p)
            x = signal.istft(S)
            p = np.angle(signal.stft(x, N_FFT))

        return x


    def run(self):
        graph = tf.Graph()

        with tf.Graph().as_default(), tf.device('/gpu:0'), tf.Session() as sess: #starting tensor flow
            self.buildModel(self.audioModel.num_channels,self.audioModel.num_samples)

            #self.net is the network
            LossStyle = self.styleLoss(sess)
            LossContent = self.contentLoss(sess)

            L_tv = tf.image.total_variation(self.net['input'])

            LossTotal  = self.alpha * LossContent
            LossTotal += self.beta  * LossStyle
            LossTotal += self.theta * L_tv

            #optimizer = lgbfs

            maxIterations = 1000
            optimizerLBFGS = tf.contrib.opt.ScipyOptimizerInterface(
                LossTotal, method='L-BFGS-B',
                options={'maxiter': maxIterations,
                         'disp': 0})

            self.minimizeLBGFS(sess,optimizerLBFGS)

            outputSpec = sess.run(self.net['input'])
            outputSignal = self.convertOutput(outputSpec)

            wav.write('output.wav',self.audioModel.styleSR,outputSignal)

            




def runNetwork():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--contentfile",required=True)
    ap.add_argument("-s","--stylefile",required=True)
    args = vars(ap.parse_args())

    audioModel = AudioSignal(args['contentfile'],args['stylefile'])
    audioModel.createSpectrogram()
    audioModel.initSignal('random')

    CNNmodel = CNN(audioModel)

    CNN.run()


