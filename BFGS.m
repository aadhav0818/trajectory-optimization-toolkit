%% BFGS (Broyden–Fletcher–Goldfarb–Shanno) Quasi Newton N-D Unconstrained Optimizer using Quadratic Polynomial Line Searches
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

function [x_min, f_min, calls, X_hist, success] = BFGS(f, x, t, eps)
    fpp_calls = 0;
    fp_calls = 0;
    f_calls = 0;

    maxit = 1000;
    N = length(x);
    Qk = eye(N);

    X_hist = x;
    success = false;

    [g, f_CX_calls] = CXStepGradient(f, x, 1e-50, 'col');
    fp_calls = fp_calls + 1;
    f_calls = f_calls + f_CX_calls;

    for ii = 1:maxit
        if norm(g) < eps
            success = true;
            break;
        end

        s = -Qk*g;

        if s'*g >= 0
            Qk = eye(N);
            s = -g;
        end

        [x_new, f_min, f_lineCalls] = polyLineSearch(f, x, s/norm(s), t, eps, 1);

        f_calls = f_calls + f_lineCalls;

        p = x_new - x;

        if norm(p) < 1e-14
            break;
        end

        [g_new, f_CX_calls] = CXStepGradient(f, x_new, 1e-50, 'col');
        fp_calls = fp_calls + 1;
        f_calls = f_calls + f_CX_calls;

        y = g_new - g;
        sigma = p'*y;

        if sigma > 1e-12*norm(p)*norm(y)
            A = Qk*y*p';
            tau = y'*Qk*y;

            dQk = (sigma+tau)/sigma^2*(p*p') -(1/sigma)*(A+A');

            Qk = Qk + dQk;
            Qk = (Qk + Qk')/2;
        else
            Qk = eye(N);
        end

        x = x_new;
        g = g_new;
        X_hist = [X_hist, x];
    end

    x_min = x;
    f_min = f(x);
    f_calls = f_calls + 1;

    if norm(g) < eps
        success = true;
    end

    calls = [f_calls fp_calls fpp_calls];
end