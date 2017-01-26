load('predict_input_transpose.mat')
load('predict_output_transpose.mat')
load('train_input_transpose.mat')
load('train_output_transpose.mat')
net=newff(train_input_transpose,train_output_transpose,{10,2});
[net,tr]=train(net,train_input_transpose,train_output_transpose);