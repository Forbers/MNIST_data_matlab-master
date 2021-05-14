clear 
clc 
close all
load('MNIST_for_BP.mat')
% train_ima = loadMNISTImages(url);
% train_lab = loadMNISTLabels(url);
% test_ima  = loadMNISTImages(url); %data after reshape , original size is 28x28
% test_lab  = loadMNISTLabels(url); %label
% 读取四个文件
%-----------Training Part-----------%
%三层神经网络
sam_sum = 60000;
input = 784;
output = 10;%the number of output corresponds to 1~10
hid = 15;%number of hidden layer
w1 = randn([input,hid]);%initialization convolution kernal
Temporary_variable = w1;
w2 = randn([hid,output]);
bias1 = zeros(hid,1);
bias2 = zeros(output,1);
rate1 = 0.005;
rate2 = 0.005;                %设置学习率

%用作三层神经网络的暂时存储变量
temp1 = zeros(hid,1);
net = temp1;
temp2 = zeros(output,1);
z = temp2;


for num = 1:100
    num
for i = 1:sam_sum%训练集总数
    
    label = zeros(10,1);
    label(train_lab(i)+1,1) = 1;

    %forward
    %此处选用784*1的train_ima矩阵，因此需要加(:,1)
    temp1 = train_ima(:,i)'*w1 + bias1';%output of 1st layer
    net = sigmoid(temp1);%excited output
    %此处选用hid*1的隐藏层矩阵
    temp2 = net*w2 + bias2';%output of 2nd layer
    z = sigmoid(temp2);%final output
    z = z';net = net';
    %backward project 
    error = label - z;
    deltaZ = error.*z.*(1-z);% z.*(1-z) is the deriviation of sigmoid(2nd layer
    deltaNet = net.*(1-net).*(w2*deltaZ);% deltaNet is the recursive formula of deltaZ
    for j = 1:output
        w2(:,j) = w2(:,j) + rate2*deltaZ(j).*net;%backward project renew w
    end
    for j = 1:hid
        w1(:,j) = w1(:,j) + rate1*deltaNet(j).*train_ima(:,i);
    end
    bias2 = bias2 + rate2*deltaZ;%renew b
    bias1 = bias1 + rate1*deltaNet;
end
end
%-----------Training Part-----------%

%-----------Test Part-----------%
test_sum = 10000;
count = 0;
for i = 1:test_sum
    temp1 = test_ima(:,i)'*w1 + bias1';
    net = sigmoid(temp1);
    %此处选用hid*1的隐藏层矩阵
    temp2 = net*w2 + bias2';
    z = sigmoid(temp2);
    
    [maxn,inx] = max(z);
    inx = inx -1;
    if inx == test_lab(i)
        count = count+1;
    end
end
correctRate = count/test_sum