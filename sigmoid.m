function output = sigmoid(x)
% sigmoid �����
%sigmoid���� f = 1/(e^-x + 1)   ��Ϊ f' = f*(1-f)

output=(1-exp((-2).*x))./(1+exp((-2).*x));

end