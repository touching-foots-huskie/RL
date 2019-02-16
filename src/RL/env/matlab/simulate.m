function result = simulate(w1, b1, w2, b2, w3, b3, sigma)
% 2 hidden layer
% input is the weights and biases:
% dim should be: 7:14:1
% w1 should be (14,7); w2 (1, 7)

global W1 B1 W2 B2 W3 B3 Sigma
% reshape
%W
W1 = reshape(w1, 7, 35);
W2 = reshape(w2, 35, 5);
W3 = reshape(w3, 5, 1);
%b
B1 = reshape(b1, 35, 1);
B2 = reshape(b2, 5, 1);
B3 = reshape(b3, 1, 1);
%sigma
Sigma = sigma;
% run the simulate:
sim('env');
out_reward = reward.data; % 1
out_state = state.data; % 7
out_action = action.data; % 1
out_error = error.data; % 1
% merge:
result = [out_state, out_reward, out_action, out_error];
end