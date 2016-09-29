clear

% model setup
model = Network();
model.add(Linear('fc1', 784, 128, 0.001));
model.add(Sigmoid('sigm1'));
model.add(Linear('fc2', 128, 10, 0.001));
model.add(Sigmoid('sigm2'));
loss = EuclideanLoss('loss');

% load data
if ~exist('data/mnist.mat', 'file')
	get_mnist('data')
end

load('data/mnist.mat');

% normalize to 0 ~ 1
train_data = reshape(mnist.train_data, [], 60000) / 255;  
test_data = reshape(mnist.test_data, [], 10000) / 255;
train_label = mnist.train_label;
test_label = mnist.test_label;

% Original config
% update.learning_rate = 0.01;
% update.weight_decay = 0.005;
% update.momentum = 0.9;
update.learning_rate = 0.5;
update.weight_decay = 0;
update.momentum = 0;

% solver config
% NOTE: one iteration means net forward-backprops one minibatch sample.
%       one epoch means net has gone through the whole training dataset
solver.update = update;
solver.shuffle = true;      % if shuffle when every epoch begins
solver.batch_size = 1;     % minibatch sample size 
solver.display_freq = 100;  % the number of iterations between displaying info. 
solver.max_iter = 100000;    % the maximum number of iterations
solver.test_freq = 10;    % The number of iterations between testing.

solve_mlp(model, loss, train_data, train_label, ...
      test_data, test_label, solver);
