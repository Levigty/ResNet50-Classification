load validation.mat
% class num
num_class = 8;
% Fold save path
path=('E:\Github\Resnet50-Classification\dataset\');
train_num=size(validation.train,2);
test_num=size(validation.test,2);
% create fold
for i=1:num_class
    mkdir([path,'train'],num2str(i));
    mkdir([path,'test'],num2str(i));
end
mkdir(path,'model');
% create img
for i=1:train_num
    ID = validation.train(1,i);
    img = validation.data{ID,1};
    label = validation.data{ID,2};
    store = [path,'train/',num2str(label),'/',num2str(ID),'.jpg'];
    imwrite(img,store);
end
disp('Train data finished.');
for i=1:test_num
    ID = validation.test(1,i);
    img = validation.data{ID,1};
    label = validation.data{ID,2};
    store = [path,'test/',num2str(label),'/',num2str(ID),'.jpg'];
    imwrite(img,store);
end
disp('Test data finished.');