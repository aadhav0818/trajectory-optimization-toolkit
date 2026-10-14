%% Fletcher-Reeves Conjugate Gradient N-D Unconstrained Optimizer using Quadratic Polynomial Line Search
% INPUTS
% ====================================================================================
% f = function handle
% x = location
% t = initial stride for line search
% eps = error tolerance
% ====================================================================================
%
% OUTPUTS
% ====================================================================================
% x_min = location of minimum in coordinates (x1 ... xn)^T
% f_min = function value at minimum
% calls = 1x3 vector [f_calls, fp_calls, fpp_calls]
% ====================================================================================
%
% DEPENDENCIES
% ====================================================================================
% [grad, f_calls] = CXStepGradient(f, x, h)
% [x_min, f_min, n_calls] = polyLineSearch(f, x, s, t, eps)
% ====================================================================================

function [x_min, f_min, calls] = fletcherReeves(f, x, t, eps)
    tol = eps;
    fpp_calls = 0;
    fp_calls = 0;
    f_calls = 0;
    s = zeros(length(x), 1);
    beta = 0;
    d_prev = zeros(length(x), 1);
    while tol >= eps
        [f_grad, f_CX_calls] = CXStepGradient(f, x, 1e-50);
        fp_calls = fp_calls + 1;
        f_calls = f_calls + f_CX_calls;

        d = -f_grad';

        if norm(d_prev) ~= 0
            beta = (d'*d)/(d_prev'*d_prev);
        end
        s = d + beta*s;
        d_prev = d;
        
        tol = norm(f_grad);
        if(tol < eps)
            x_min = x;
            f_min = f(x);
            f_calls = f_calls + 1;
            break;
        end
        
        [x_min, f_min, f_lineCalls] = polyLineSearch(f, x, s, t, eps);
        x = x_min;
        
        f_calls = f_calls + f_lineCalls;
    end
    
    calls = [f_calls fp_calls fpp_calls];
end