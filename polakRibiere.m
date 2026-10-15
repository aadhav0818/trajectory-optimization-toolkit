%% Polak-Ribiere Conjugate Gradient N-D Unconstrained Optimizer using Quadratic Polynomial Line Search
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

function [x_min, f_min, calls, X_hist, success] = polakRibiere(f, x, t, eps)
    fpp_calls = 0;
    fp_calls = 0;
    f_calls = 0;

    maxit = 1000;
    X_hist = x;
    success = false;

    s = zeros(length(x), 1);
    d_prev = zeros(length(x), 1);

    for ii = 1:maxit
        [f_grad, f_CX_calls] = CXStepGradient(f, x, 1e-50, 'col');
        fp_calls = fp_calls + 1;
        f_calls = f_calls + f_CX_calls;

        tol = norm(f_grad);

        if tol < eps
            success = true;
            break;
        end

        d = -f_grad;

        if ii == 1
            beta = 0;
        else
            beta = max(0, (d'*(d-d_prev))/(d_prev'*d_prev));
        end

        s = d + beta*s;

        if s'*f_grad >= 0
            s = d;
        end

        d_prev = d;

        [x_min, f_min, f_lineCalls] = polyLineSearch(f, x, s/norm(s), t, eps, 1);

        f_calls = f_calls + f_lineCalls;

        if norm(x_min-x) < 1e-14
            break;
        end

        x = x_min;
        X_hist = [X_hist, x];
    end

    x_min = x;
    f_min = f(x);
    f_calls = f_calls + 1;

    calls = [f_calls fp_calls fpp_calls];
end