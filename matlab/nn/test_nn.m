load('predict_input_transpose.mat')
load('predict_output_transpose.mat')
load('train_input_transpose.mat')
load('train_output_transpose.mat')
load('network_hit139.mat')
predict_result = sim( network_hit139, predict_input_transpose);
%对网络进行测试
avg=mean(predict_result(:)); 
for i=1:200
    if predict_result(1,i)>avg
        predict_result(1,i)=1;
    else predict_result(1,i)=-1;
    end
end
%输出为[-1,1]之间连续的值，所以还要进行二值化
accuracy=0;
for i=1:200
    if predict_result(1,i)==predict_output_transpose(1,i)
        accuracy=accuracy+1;
    end
end
accuracy/200
%测试准确率