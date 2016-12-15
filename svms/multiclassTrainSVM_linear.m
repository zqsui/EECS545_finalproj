function model = multiclassTrainSVM_linear(data_matrix)
% Input being a data matrix (nxm) - image descriptors - one feature vector for an image
% Each row of the data matrix is a feature vector (1x(m-1)) for an image in the
% dataset
% Last column of the data matrix is the label assigned to the image. Let
% the list of labels be (1xl)
% Output will be a trained model - list of hyperplanes (w, b) forming (lxm) matrix (for one-vs-all
% classification)

%Check for the input data matrix and set the parameters
[n, m] = size(data_matrix);
labels = unique(data_matrix(:, end));
%Randomize the datamatrix before the cross validation process
randorder = randperm(n);
feature_matrix = data_matrix(randorder, 1:end-1);
label_vector = data_matrix(randorder, end);

%5 fold Crossvalidated training by libsvm 
%model = svmtrain(label_vector, feature_matrix, '-v 5 -t 0 -c 10');
c_set = 1:5;
l=1;
best_c = 1;
for levels=1:3
    test_accuracy = [];
    for c=c_set
        model = svmtrain(label_vector, feature_matrix, ['-q -v 5 -t 0 -c ',num2str(c)]);
        test_accuracy(end+1)=model;
    end
    [v, i] = max(test_accuracy);
    mid = c_set(i);
    disp([mid, v]);
    c_set = ((mid-l*0.5)+0.2*l):0.2*l:(mid+l*0.5);
    l=0.2*l;
    best_c = mid;
end

%Do the actual training now
model = svmtrain(label_vector, feature_matrix, ['-q -t 0 -c ',num2str(best_c)]');

end