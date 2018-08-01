import itertools
import matplotlib.pyplot as plt
import numpy as np
import os, math
import matplotlib.image as mpimg
import torch
import matplotlib as mpl
from skimage import io
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import copy
from sklearn.decomposition import PCA
import seaborn as sns
import torchvision.models as models
import torch.nn.functional as F
from glob import glob
import cv2
from src.utility import preprocess_image, convert_image_np, getImageNetDict, getImageNetCodeDict
import torch.optim as optim
import torch.nn as nn
import copy, time
from torchvision import transforms, utils
import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from collections import Counter
from joblib import Parallel, delayed
import multiprocessing

alexnet = models.alexnet(pretrained=True)
imagenetdict = getImageNetDict()
imagenetcodedict = getImageNetCodeDict()
name2code = dict((v, k) for k, v in imagenetcodedict.items())


def to_var(x, *args, **kwargs):
    if type(x) is list or type(x) is tuple:
        x = [Variable(x_, *args, **kwargs) for x_ in x]
    else:
        x = Variable(x, *args, **kwargs)
    return x

# display the basis vectors
def preprocess_im(im, volatile=False):
    original_image = cv2.imread(im)
    prep_img = preprocess_image(original_image)
    if volatile:
        prep_img.volatile = True
    return prep_img

def show_image(images, title=''):
    seq = []
    for im in images:
        prep_img = preprocess_im(im, volatile=True)
        seq.append(prep_img)
    images_batch = torch.cat(seq, dim=0)
    grid = utils.make_grid(images_batch.data)   
    plt.imshow(convert_image_np(grid))
    plt.title(title)
    plt.show()
    
def show_concepts(concepts):
    prefix = 'tiny-imagenet-200/train/'
    res = []
    for c in concepts:
        name = prefix + name2code[c]
        all_pics = glob(name + '/images/*')
        rs = np.random.choice(range(len(all_pics)), 8, replace=False)
        res.extend([all_pics[r] for r in rs])
    
    plt.figure(figsize=(10,10))
    show_image(res)
    plt.show()
    
def generate_basis(concept_class, W):
    n_per_class = math.ceil(W.shape[1] / len(concept_class))
    prefix = 'tiny-imagenet-200/train/'
    res = []
    concepts = []
    for c in concept_class:
        name = prefix + name2code[c]
        all_pics = glob(name + '/images/*')
        rs = np.random.choice(range(len(all_pics)), n_per_class, replace=False)
        res.extend([all_pics[r] for r in rs])
        concepts.extend([name2code[c] for r in rs])
        
    order = np.random.permutation(W.shape[1])
    res = np.array(res)[order]
    concepts = np.array(concepts)[order]
    return res, concepts

def get_test_images(concept_class):
    anno = 'tiny-imagenet-200/val/val_annotations.txt'
    d = {}
    with open(anno) as f:
        for l in f:
            k, v = l.split()[:2]
            d[k] = v
            
    prefix = 'tiny-imagenet-200/val/images/'
    images = []
    labels = []
    for im in glob(prefix + '*'):
        c = d[im.split('/')[-1]]
        if imagenetcodedict[c] in concept_class:
            images.append(im)
            labels.append(c)
    return images, labels

def predict(net, im):
    net.eval()
    prep_img = preprocess_im(im)
    prediction = net(prep_img).data.cpu().numpy().argmax()
    return imagenetdict[prediction], prediction

def extract_features(net, images):
    net.eval()
    seq = []
    for im in tqdm.tqdm(images):
        prep_img = preprocess_im(im, volatile=True)
        seq.append(prep_img)
        
    features = []
    for i in tqdm.tqdm(range(0, len(seq), 100)):
        images_batch = torch.cat(seq[i:i+100], dim=0)
        features.append(net(images_batch))
    return torch.cat(features, dim=0)

def check_ortho(theta_list, plot=False, verbose=False):
    Q = [t/np.linalg.norm(t) for t in theta_list]
    Q = np.vstack(Q).T
    error = np.sum(np.abs(Q.T.dot(Q) - np.eye(Q.shape[1])))
    if verbose:
        print("orthogonal check error: ", error)
    if plot:
        print(Q.T.dot(Q))
        plt.matshow(Q.T.dot(Q))
        plt.show()
    return error

def ortho_inverse(P):
    out = copy.deepcopy(P.T)
    div = np.diag(P.T.dot(P))
    for i, v in enumerate(div):
        out[i] /= v
    return out
    
def orthogonize_(theta, prev_theta_list):
    '''goal: theta should be orthogonal to all vectors in prev_theta_list'''
    theta = theta / np.linalg.norm(theta)
    for t in prev_theta_list:
        theta = theta - theta.dot(t) / np.dot(t,t) * t 
    return theta

def orthogonize(theta, prev_theta_list, verbose=False):
    done = len(prev_theta_list) == 0
    while not done:
        theta = orthogonize_(theta, prev_theta_list)
        #check_ortho(prev_theta_list, plot=False)        
        #check_ortho(prev_theta_list + [theta], plot=False)
        #print('in middle', theta)
        #print(check_ortho(prev_theta_list + [theta]))
        #if  check_ortho(prev_theta_list + [theta]) > 1e-3:
            #print(check_ortho(prev_theta_list + [theta]))
        if np.linalg.norm(theta) < 1e-10: # give a random direction to theta to theta too close to 0
            if verbose:
                print('restart a vector')
            theta = np.random.randn(len(theta))
        else:
            done = True
        
    return theta / np.linalg.norm(theta) # to keep theta in a good range for numerical stability

def data_range(A, t):
    return A.dot(t/t.dot(t)).ravel().std()

def normalize_theta(A, t):
    drange = data_range(A, t)
    if drange == 0: # no need to normalize
        return t
    return t * drange # multiply so that in P^-1 is divide by

def print_result(W, concepts, errors, threshold=0.1, mask=True):
    coe = W
    print('"*" postfixed means error < %.4f' % threshold)
    print('%15s %10s %10s' % ('concepts', 'coeff', 'error'))
    print('-'*37)
    for w, c, e in sorted(zip(coe, concepts, errors), key=lambda x: abs(x[0]), reverse=True):
        if e < threshold:
            c = c+"*"
        else:
            if mask: c = 'unknown'
        print("%15s %10.4f %10.4f" % (c, w, e))   

def eval_direction(theta, c, l, A, plot=False, verbose=False):
    '''concept c, labels l, activation map A'''
    y = np.zeros(len(l))
    y[np.array(l) == c] = 1
    
    n, d = A.shape
    a = A.dot(theta).ravel()
    X = np.vstack([a, np.ones(n)]).T
    
    #print(X.shape, y.shape)
    
    m, b = np.linalg.lstsq(X, y)[0]
    error = np.mean((X.dot(np.array([m,b])) - y)**2)
    
    if verbose:
        print('%s error: %.4f, slope: %+.4f, intercept: %+4f' % (c.ljust(14), error, m, b))        
    if plot:
        colors = list(map(switch_color, l))
        transformed = X.dot(np.array([m,b]))
        plt.scatter(transformed, transformed + np.random.random(n), c=colors)
        plt.show()
    return error, m, b

def interpret_direction(t, concepts_list, labels, A):
    min_e = 1000
    min_c = ""
    min_m = 0
    for c in concepts_list:
        e, m, b = eval_direction(t, c, labels, A)
        if e < min_e:
            min_e = e
            min_c = c
            min_m = m
    return min_e, min_m, min_c 

def plot_inter_stats(directions, neuron_numbers, concepts, labels, A):
    theta_list = []
    errors = []
    for i in tqdm.tqdm(neuron_numbers):
        direction = directions[i]
        e, m, c = interpret_direction(direction, concepts, labels, A) 
        if m < 0:
            direction = -direction
        theta_list.append(direction)
        errors.append(e)

    sns.distplot(errors)
    plt.ylabel('frequency')
    plt.xlabel('interpretable error')
    plt.show()
    return errors

def print_mean_std(errors, name=""):
    m, s = np.mean(errors), np.std(errors)
    print(('%20s mean: %.3f, std: %.4f') % (name, m, s))
    
def concept_scatter(a, concepts, labels):
    for c in concepts:
        a_ = a[labels==c]
        plt.scatter(a_, a_ + np.random.random(len(a_))*3, label=imagenetcodedict[c])
    plt.legend()
    
class LR(nn.Module): # logistic regression with 2 neurons
    def __init__(self, input_size, output_size=2):
        super().__init__()
        
        self.i2o = nn.Linear(input_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.output_size = output_size
        self.input_size = input_size
        
    def forward(self, input):
        output = self.softmax(self.i2o(input))
        return output

def get_y(c, l, A):
    '''concept c, labels l, activation map A'''
    y = np.zeros(len(l))
    y[np.array(l) == c] = 1
    return y   

def plotDirection(direction, A, l, concepts):
    plt.figure(figsize=(7,7))
    e, m, c = interpret_direction(direction, concepts, l, A) 
    if m < 0:
        direction = -direction
    a = A.dot(direction)
    if a.std() != 0:
        a = a / a.std()
    concept_scatter(a, concepts, np.array(l))
    plt.title('%s error: %.4f' % (imagenetcodedict[c], e))    
    
def train(net, trainloader, criterion, optimizer, print_every=None, epochs=2, max_time=10):
    start = time.time()
    
    # max_time given in seconds
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            end = time.time()
            if end - start >= max_time:
                print('Finished Training in %ds' % (end-start))                
                return
            
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(to_var(inputs))
            loss = criterion(outputs, to_var(labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if print_every is not None and i % print_every == (print_every-1): 
                print('[%d, %5d] loss: %.10f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

    print('Finished Training in %ds' % (end-start))

def custom_solve(A, y, max_time=3):
    concept_lr_data = TensorDataset(torch.from_numpy(A), torch.from_numpy(y).long())
    trainloader = DataLoader(concept_lr_data, batch_size=32, num_workers=0)
    
    concept_net = LR(4096)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(concept_net.parameters())
    train(concept_net, trainloader, criterion, optimizer, print_every=None, epochs=5, max_time=max_time)   
    
    theta = concept_net.i2o.weight
    theta = (theta[1] - theta[0]).data.numpy()
    return theta

def custom_leastsq_concept(c, l, A, max_time=3):
    '''concept c, labels l, activation map A'''
    y = get_y(c, l, A)
    theta = custom_solve(A, y, max_time=max_time)
    #error, m, b = eval_direction(theta, c, l, A)
    return theta#, error

def buildP(A, concepts_list, labels, A_test, test_labels, threshold, name="theta_list_para2"):

    def data_range(A, t):
        return A.dot(t/t.dot(t)).ravel().std()

    def normalize_theta(A, t):
        drange = data_range(A, t)
        if drange == 0: # no need to normalize
            return t
        return t * drange # multiply so that in P^-1 is divide by
    
    def greedy_find_c(concepts_list, labels, A, A_test, test_labels):

        def inner_loop(c):
            theta = custom_leastsq_concept(c, labels, A)
            error, m, b = eval_direction(theta, c, test_labels, A_test)
            return theta, error, c

        n_cpu = min(int(multiprocessing.cpu_count() / 2), len(concepts_list))
        t_e_c = Parallel(n_jobs=n_cpu)(delayed(inner_loop)(c) for c in concepts_list)
        min_t, min_e, min_c = min(t_e_c, key=lambda x: x[1])
        return min_c, min_t, min_e

    A_backup = copy.deepcopy(A)
    theta_list = []
    
    concepts_used = []
    print('fitting directions......')
    errors = []
    for _ in tqdm.tqdm(range(A.shape[1])):
        min_c, theta, error = greedy_find_c(concepts_list, labels, A, A_test,test_labels)
        concepts_used.append(min_c)
        errors.append(error)
        np.save("data/" + name, theta_list)
        theta = orthogonize(theta, theta_list) # just make sure theta are orthogonal # note: randomness here
        theta_list.append(theta)
        A = A - np.outer(A.dot(theta).ravel(), theta) / np.sum(theta**2) # project data down 1 dimension

    A = copy.deepcopy(A_backup)
    P = np.vstack(theta_list).T

    errors = []
    slopes = []
    min_c = []
    for t in tqdm.tqdm(theta_list):
        e, m, c = interpret_direction(t, concepts_list, test_labels, A_test) 
        errors.append(e)
        slopes.append(m)
        min_c.append(c)

    
    # decorrelate unknown data using pca
    B = A.dot(ortho_inverse(P).T) # new activation
    A = []
    ind_changed = []
    for i in range(len(P)):
        if errors[i] >= threshold:
            A.append(B[:,i])
            ind_changed.append(i)
    if len(ind_changed) > 0:
        A = np.vstack(A).T
        print('doing pca %d directions need changes......' % len(ind_changed))
        pca = PCA(n_components=A.shape[1]) 
        pca.fit(A) # takes 5 min
        
        P_subset = np.vstack([P[:,i] for i in ind_changed]).T

        pca_comp = P_subset.dot(pca.components_.T) # n_comp x d
        for i, ind in enumerate(ind_changed):
            comp = pca_comp[:,i] 
            theta_list[ind] = comp

    # normalize variance
    print('normalizing variance......')
    A = copy.deepcopy(A_backup)
    theta_list = [normalize_theta(A, t) for t in theta_list]
    
    print('reevaluating direction errors......') 
    errors = []
    slopes = []
    min_c = []
    for t in tqdm.tqdm(theta_list):
        e, m, c = interpret_direction(t, concepts_list, test_labels, A_test) 
        errors.append(e)
        slopes.append(m)
        min_c.append(c)

    theta_list = list(map(lambda x: x[0] * np.sign(x[1]), zip(theta_list, slopes)))
    P = np.vstack(theta_list).T

    np.save("data/" + name, theta_list)
    return P, errors, theta_list, min_c


model = copy.deepcopy(alexnet)
model.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])
W = list(alexnet.classifier.children())[-1].weight.cpu().data.numpy()
net = alexnet

concept_classes = ['convertible', 'dugong', 'golden_retriever', 'stopwatch', 'orange', 'cauliflower', 'hourglass', 'pizza', 'African_elephant']
basis, labels = generate_basis(concept_classes, W)
concepts = set(labels)

features = extract_features(model, basis)
P = features.transpose(0,1).data.cpu().numpy()

activations = P.T
A = activations # A should be n x r

test_images, test_labels = get_test_images(concept_classes)
features_test = extract_features(model, test_images)
A_test = features_test.data.cpu().numpy()

A = copy.deepcopy(activations)
threshold = 0.071
P, errors, theta_list, min_c = buildP(A, concepts, labels, A_test, test_labels,
                                      threshold=threshold, name="theta_list_para2")
