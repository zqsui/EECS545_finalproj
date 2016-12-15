function model = multiclassTrainSVM_nonlinear(data_matrix)
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
d_g= 1/3000;
d_c=5;
c_set = (d_c/5:d_c/5:2*d_c/5);
g_set = (d_g-0.5*d_g):0.2*d_g:(d_g+0.5*d_g);
l=2;
best_c = 2.^0;
best_g = 2.^0;

for levels=1:3
    test_accuracy = zeros(size(c_set, 2), size(g_set, 2));
    for c=1:size(c_set, 2)
        for g=1:size(g_set, 2)
            model = svmtrain(label_vector, feature_matrix, ['-q -v 5 -t 2 -c ',num2str(c_set(c)), ' -g ',num2str(g_set(g))]);
            test_accuracy(c, g)=model;
            disp([c_set(c), g_set(g)]);
        end
    end
    m_acc = max(max(test_accuracy));
    [row, col] = find(test_accuracy==m_acc);
    mid_c = c_set(row);
    mid_g = g_set(col);
    disp([mid_c, mid_g, m_acc]);
    c_set = (mid_c-mid_c/l):0.5*l:(mid_c+mid_c/l);
    g_set = (mid_g-mid_g/l):0.5*l:(mid_g+mid_g/l);
    l=2*l;
    best_c = mid_c;
    best_g = mid_g;
end

%Do the actual training now
model = svmtrain(label_vector, feature_matrix, ['-q -t 2 -c ',num2str(best_c), ' -g ', num2str(best_g)]');

end