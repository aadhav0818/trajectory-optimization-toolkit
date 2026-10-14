%% Complex (CX) Step Partial Derivative Gradient Evaluations for N-D Functions
% INPUTS
% ====================================================================================
% f = function handle
% x = Nx1 vector for an N-D point
% h = step size
% ====================================================================================
% OUTPUTS
% ====================================================================================
% grad = 1xN gradient vector 
% f_calls = number of function evaluations (scalar)
% ====================================================================================

function [grad, f_calls] = CXStepGradient(f, x, h)
    N = length(x);
    grad = zeros(1, N);
    for ii = 1:N
        x_complex = x;
        x_complex(ii) = x_complex(ii) + 1i * h;
        grad(ii) = imag(f(x_complex)) / h;
    end
    f_calls = N;
end