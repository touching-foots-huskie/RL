function [sys,x0,str,ts] = NN_control(t,x,u,flag)
%NN_control: error compensate using neutral network  
switch flag,

  %%%%%%%%%%%%%%%%%%
  % Initialization %
  %%%%%%%%%%%%%%%%%%
  case 0,
    [sys,x0,str,ts]=mdlInitializeSizes();

  %%%%%%%%%%%
  % Outputs %
  %%%%%%%%%%%
  case 3,
    sys=mdlOutputs(t,x,u);
    
  %%%%%%%%%%%%%%%%%%%
  % Unhandled flags %
  %%%%%%%%%%%%%%%%%%%
  case { 1, 2, 4, 9},
    sys = [];

  %%%%%%%%%%%%%%%%%%%%
  % Unexpected flags %
  %%%%%%%%%%%%%%%%%%%%
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));

end
% end NN_control

%=============================================================================
% mdlInitializeSizes
% Return the sizes, initial conditions, and sample times for the S-function.
%=============================================================================
%
function [sys,x0,str,ts]=mdlInitializeSizes()
sizes = simsizes;
sizes.NumContStates  = 0;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 1;
sizes.NumInputs      = 7;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);
x0  = [];
str = [];
ts  = [0 0];


% end mdlInitializeSizes
%=============================================================================
% mdlOutputs
% Return the block outputs.
%=============================================================================
%
function sys=mdlOutputs(t,x,u)
% global opts net
% get net and opts:
% res(1).x = u;
% [net,res,opts] = net_ff(net,res,opts); 
% out = double(res(end).x);
global W1 B1 W2 B2 W3 B3 Sigma
% the first layer:
h1 = W1'*u + B1;
% activate here
h1 = tanh(h1);
% the second layer:
h2 = W2'*h1 + B2;
% activate here
h2 = tanh(h2);
% the second layer:
out = W3'*h2 + B3;
% activate here
out = tanh(out);
sys = out;
% end mdlOutputs
