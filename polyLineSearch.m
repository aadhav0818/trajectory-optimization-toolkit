%% Polynomial Quadratic Line Search Method for Minimizing N-D function
% f = function handle
% s = search direction
% t = initial stride
% x = location
% eps = tolerance to stop refining

function [x_min, f_min, n_calls, X_calls] = polyLineSearch(f, x, s, t, eps, growth)
    n_calls = 0;
    X_calls = [];
    maxit = 1000;

    f0 = f(x);
    f1 = f(x + t*s);
    f2 = f(x + 2*t*s);
    n_calls = 3;
    X_calls = [x, x+t*s, x+2*t*s];

    for ii = 1:maxit
        if f1 < f0
            break;
        end

        t = t/2;
        f1 = f(x + t*s);
        f2 = f(x + 2*t*s);
        n_calls = n_calls + 2;
        X_calls = [X_calls, x+t*s, x+2*t*s];

        if t < 1e-14
            x_min = x;
            f_min = f0;
            return;
        end
    end

    a0 = 0;
    a1 = t;
    a2 = 2*t;

    step = t;
    
    for ii = 1:maxit
        if f1 <= f0 && f1 <= f2
            break;
        end
    
        a0 = a1;
        f0 = f1;
    
        a1 = a2;
        f1 = f2;
    
        step = step*growth;
        a2 = a1 + step;
    
        f2 = f(x + a2*s);
        n_calls = n_calls + 1;
        X_calls = [X_calls, x+a2*s];
    end

    for ii = 1:maxit
        if abs(a2-a0) < eps
            break;
        end

        numer = f0*(a1^2-a2^2) + f1*(a2^2-a0^2) + f2*(a0^2-a1^2);
        denom = f0*(a1-a2) + f1*(a2-a0) + f2*(a0-a1);

        if ~isfinite(denom) || abs(denom) < 1e-14
            a_new = (a0+a2)/2;
        else
            a_new = 0.5*numer/denom;
        end

        if ~isfinite(a_new) || a_new <= a0 || a_new >= a2 || ...
                abs(a_new-a1) < 1e-12*(a2-a0)
            if a1-a0 > a2-a1
                a_new = (a0+a1)/2;
            else
                a_new = (a1+a2)/2;
            end
        end

        f_new = f(x + a_new*s);
        n_calls = n_calls + 1;
        X_calls = [X_calls, x+a_new*s];

        if ~isfinite(f_new)
            break;
        end

        if a_new < a1
            if f_new < f1
                a2 = a1;
                f2 = f1;
                a1 = a_new;
                f1 = f_new;
            else
                a0 = a_new;
                f0 = f_new;
            end
        else
            if f_new < f1
                a0 = a1;
                f0 = f1;
                a1 = a_new;
                f1 = f_new;
            else
                a2 = a_new;
                f2 = f_new;
            end
        end
    end

    x_min = x + a1*s;
    f_min = f1;
end