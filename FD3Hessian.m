%% Finite Difference (FD) 3-Point Stencil Hessian Evaluation for N-D Functions
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

function [H, f_calls] = FD3Hessian(f, x, h)
    x = x(:);
    N = length(x);
    H = zeros(N,N);
    f_calls = 0;

    fx = f(x);
    f_calls = f_calls + 1;

    for i = 1:N
        ei = zeros(N,1);
        ei(i) = 1;

        f_plus = f(x + h*ei);
        f_minus = f(x - h*ei);
        f_calls = f_calls + 2;

        H(i,i) = (f_plus - 2*fx + f_minus) / h^2;
    end

    for i = 1:N

        ei = zeros(N,1);
        ei(i) = 1;

        for j = i+1:N

            ej = zeros(N,1);
            ej(j) = 1;

            f_pp = f(x + h*ei + h*ej);
            f_pm = f(x + h*ei - h*ej);
            f_mp = f(x - h*ei + h*ej);
            f_mm = f(x - h*ei - h*ej);
            f_calls = f_calls + 4;
            
            H(i,j) = (f_pp - f_pm - f_mp + f_mm) / (4*h^2);

            H(j,i) = H(i,j);

        end
    end

end