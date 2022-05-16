import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim,nn
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from myutils import choose_device,data_process
from model import Basicblock,ResNet
from model import AlexNet
import numpy as np
import myutils
import os,sys
import argparse

#device : GPU or CPU
print("CUDA Available: ",torch.cuda.is_available())

# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument('--source_data', type=str, help='source data, i.e. MNIST,USPS,...')
parser.add_argument('--model',type=str,help='pretrained model, i.e. Resnet_SVHN,Resnet_KMNIST, note that it should matches the pre_model file name')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--epoch',type=int,default=500, help='number of model training rounds')
parser.add_argument('--train_num',type=int,default=2500,help='fine tune data size ')

args=parser.parse_args()

# 对抗训练
def adv_train_and_test(model,loss_func,optimizer,trainloader,testloader,train_data_size,test_data_size,device,save_path,epochs=200):
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1,epochs))
        # Train model
        # set to train model
        model.train()

        # loss and Accuracy mode
        train_loss =0.0
        train_acc = 0.0

        test_loss =0.0
        test_acc =0.0

        for data in trainloader:
            inputs,targets =  data
            inputs,targets = inputs.to(device),targets.to(device)
            adv_images = fast_gradient_method(model, inputs, 0.15, np.inf, clip_min=0, clip_max=1)
            adv_images = adv_images.to(device)

            #  clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(adv_images)

            # Compute loss
            loss = loss_func(outputs,targets)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for batch and add it to train_loss
            train_loss = train_loss +loss.item()*inputs.shape[0]

            # Compute the accuracy
            predictions = outputs.argmax(1)
            accuracy = torch.sum(predictions==targets)
            train_acc = train_acc +accuracy.item()
            # print(loss,accuracy)

        total_train_acc = train_acc/train_data_size
        total_train_loss = train_loss/train_data_size
        print("epoch: {}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1,total_train_loss, total_train_acc))

        # Test - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Test loop
            for data in testloader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_func(outputs, targets)

                # Compute the total loss for batch and add it to valid_loss
                test_loss = test_loss + loss.item()*inputs.shape[0]

                # Compute the accuracy
                predictions = outputs.argmax(1)
                accuracy = torch.sum(predictions == targets)
                test_acc = test_acc + accuracy.item()

            total_test_loss = test_loss/test_data_size
            total_test_acc = test_acc/test_data_size
            print("epoch: {}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, total_test_loss,total_test_acc))

    torch.save(model, save_path)

    return total_train_acc, total_test_acc

# 加载数据
if args.source_data.upper()=='MNIST':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=myutils.transform1)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=myutils.transform1)
elif args.source_data.upper()=='USPS':
    trainset = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=myutils.transform2)
    testset = torchvision.datasets.USPS(root='./data', train=False, download=True, transform=myutils.transform2)
else:
    trainset = None
    testset = None
    print("没有微调数据集！")
    sys.exit()

trainloader, testloader, train_data_size, test_data_size = data_process(trainset,testset,data_num=args.train_num)
print('train_data_size: {}     test_data_size: {}'.format(train_data_size,test_data_size))

# 选择显卡型号
device = choose_device(args.device)

print("Start Training!")

if args.model.lower().startswith('resnet'):
    # 可以冻结的层数
    layers= [4,7,10,13,16,20,23]
    # 微调模型保存的位置
    save_file = 'results_resnet/adv_train'
# elif args.model.lower().startswith('mlpmixer'):
#     # 可以冻结的层数
#     layers = [0,2,4,8,10,14,16,20,22,28]
#     # 微调模型保存的位置
#     save_file = "results_MLPMixer/"+ args.model.split('_')[-1] + '_' + args.source_data + '_tune'
# elif args.model.startswith('alexnet'):
#     layers = [0,2,4,6,8,10,12,14]
#     # 微调模型保存的位置
#     save_file = "results_alexnet/" + args.model.split('_')[-1] +'_' + args.source_data
else:
    layers = None
    pre_model = None
    save_file = None
    sys.exit('预训练模型选择错误')

if os.path.exists(save_file) == False:
    os.makedirs(save_file)

with open(os.path.join(save_file+"/result.txt"), 'a') as f:
    f.write(str(args.train_num)+"\n")

for layer_num in layers:
    # 加载预训练模型
    pre_model = torch.load("results/pre_model_" + args.model + ".pth")

    # 保留前面层不训练，梯度为 False
    pre_model.requires_grad_(False)
    for i, param in enumerate(pre_model.parameters()):
        if i < layer_num:
            param.requires_grad = True

    pre_model.to(device)

    # 查看模型各层冻结情况
    for i,param in enumerate(pre_model.parameters()):
        print(i,param.shape,param.requires_grad)

    # 损失函数:这里用交叉熵
    loss_func = nn.CrossEntropyLoss()

    # 优化器 这里用Adam 微调训练的学习率要比预训练小10倍
    optimizer = optim.Adam(pre_model.parameters(), lr=0.0001)

    print("冻结 {} 层时，开始训练：".format(layer_num))

    model_filename = "final_model_adv_" + str(layer_num) + "layers.pth"
    save_path = os.path.join(save_file, model_filename)


    total_train_acc, total_test_acc = adv_train_and_test(pre_model,loss_func,optimizer,trainloader,testloader,train_data_size,test_data_size,device,save_path,epochs=args.epoch)

    print("adv train {} layers: train_acc:{},test_acc:{}".format(layer_num,total_train_acc,total_test_acc))
    with open(os.path.join(save_file+"/result.txt"), 'a') as f:
        f.write("adv train {} layers: train_acc:{},test_acc:{}".format(layer_num,total_train_acc,total_test_acc)+"\n")

    print(str(save_path))
