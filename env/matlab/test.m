% this script is used to test the network:
% add ILC simulate part
w1 = 1e-4*rand(7, 35);
w2 = 1e-4*rand(35, 5);
w3 = 1e-4*rand(5, 1);
b1 = zeros(35, 1);
b2 = zeros(5, 1);
b3 = zeros(1, 1);
sigma = 1e-4;
result = simulate(w1, b1, w2, b2, w3, b3, sigma);