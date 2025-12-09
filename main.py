import argparse

import numpy as np
import torch
import torch.optim as optim

import loader
import helper
import model

# %%
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Deep Learning Homework Main Program')
    parser.add_argument(
        '-m',
        '--model',
        default='NN',
        help='The Model you want to use')
    parser.add_argument('-t', '--transform', default='basic',
                        help='How do you want to transform your input data?')
    parser.add_argument(
        '-l',
        '--layers',
        nargs='+',
        help='Definition for the Convolutional Neural Network',
        required=False)
    parser.add_argument(
        '-d',
        '--dropout',
        default=False,
        type=str2bool,
        help='Do you want to use dropout during training phase? (True or False)')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=64,
        type=int,
        help='mini-batch size')
    parser.add_argument(
        '-e',
        '--epoch',
        default=10,
        type=int,
        help='Number of Epoches')
    
    parser.add_argument("-c", "--use-cuda", type=str2bool, default=True)
    
    args = parser.parse_args()
    
    torch.manual_seed(1)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    
    
    if args.model == "NN":
        net = model.NN()
    elif args.model == "SimpleCNN":
        net = model.SimpleCNN()
    elif args.model == "ResCNN":
        net = model.ResCNN()
    elif args.model == "DeepCNN":
        layers = args.layers
        for i in range(len(layers)):
            try:
                layers[i] = int(layers[i])
            except BaseException:
                pass
        net = model.DeepCNN(arr=layers)
    
        print(net)
        
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("There are", num_params, "parameters in this model") 
    
    print("Use %s transformer for training" % args.transform)
    if args.transform == "basic":
        train_transform = valid_transform = model.basic_transformer
    elif args.transform == "norm":
        train_transform = model.norm_transformer
        valid_transform = model.norm_transformer
        test_transform = model.norm_transformer
    elif args.transform == "aug":
        train_transform = model.aug_transformer
        valid_transform = model.norm_transformer
        test_transform = model.norm_transformer
        
    
    trainloader, validloader, testloader = loader.get_data_loader(train_transform, valid_transform, test_transform, args.batch_size)
    

    
    
    use_cuda = torch.cuda.is_available() and args.use_cuda
    
    if use_cuda:
        net = net.cuda()
    
    
    # %%
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        print(f"\nEpoch {epoch+1} / {args.epoch}")
        learning_rate = 0.01 * 0.8 ** epoch
        learning_rate = max(learning_rate, 1e-6)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
        loss, acc = helper.run("train", trainloader, net, optimizer, use_cuda=use_cuda)
        train_losses.append(loss)
        train_accs.append(acc)
        with torch.no_grad():
            loss, acc = helper.run("valid", validloader, net, use_cuda=use_cuda)
            valid_losses.append(loss)
            valid_accs.append(acc)
    
    
    print("-"*60)
    print("best validation accuracy is %.4f percent" % (np.max(valid_accs) * 100) )
    
    torch.save(net.state_dict(), "NN.pt")  # save the model for future reference