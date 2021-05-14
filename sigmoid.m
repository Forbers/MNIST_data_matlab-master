function output = sigmoid(x)
% sigmoid 激活函数
%sigmoid函数 f = 1/(e^-x + 1)   求导为 f' = f*(1-f)

output=(1-exp((-2).*x))./(1+exp((-2).*x));

end