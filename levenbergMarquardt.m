%% Levenberg-Marquardt N-D Unconstrained Optimizer
% INPUTS
% ====================================================================================
% f = function handle
% x = location
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
% [H, f_calls] = FD3Hessian(f, x, h)
% ====================================================================================

function [x_min, f_min, calls, X_hist, success] = levenbergMarquardt(f, x, eps)
    gamma = 2;
    lambda = 1e4;
    N = length(x);
    h_cx = 1e-50;
    h_fd = 1e-4;

    f_calls = 0;
    fp_calls = 0;
    fpp_calls = 0;

    maxit = 1000;
    X_hist = x;
    success = false;

    f_last = f(x);
    f_calls = f_calls + 1;

    [g, g_f_calls] = CXStepGradient(f, x, h_cx, 'col');
    fp_calls = fp_calls + 1;
    f_calls = f_calls + g_f_calls;

    [H, h_f_calls] = FD3Hessian(f, x, h_fd);
    fpp_calls = fpp_calls + 1;
    f_calls = f_calls + h_f_calls;

    lambda_min = min(eig(H));
    lambda = max(lambda, -lambda_min + eps);

    for ii = 1:maxit
        if norm(g) < eps && lambda_min > 0
            success = true;
            break;
        end

        s = -(H + lambda*eye(N)) \ g;
        x_test = x + s;

        f_test = f(x_test);
        f_calls = f_calls + 1;

        if ~isfinite(f_test) || f_test >= f_last
            lambda = lambda*gamma;
            continue;
        end

        x = x_test;
        f_last = f_test;
        X_hist = [X_hist, x];

        [g, g_f_calls] = CXStepGradient(f, x, h_cx, 'col');
        fp_calls = fp_calls + 1;
        f_calls = f_calls + g_f_calls;

        [H, h_f_calls] = FD3Hessian(f, x, h_fd);
        fpp_calls = fpp_calls + 1;
        f_calls = f_calls + h_f_calls;

        lambda_min = min(eig(H));

        lambda = lambda/gamma^2;
        lambda = max(lambda, -lambda_min + eps);

        if norm(s) < 1e-14 && norm(g) >= eps
            break;
        end
    end

    x_min = x;
    f_min = f_last;
    calls = [f_calls, fp_calls, fpp_calls];
end