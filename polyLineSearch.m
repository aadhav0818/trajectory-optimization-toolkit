%% Polynomial Quadratic Line Search Method for Minimizing N-D function
% f = function handle
% s = search direction
% t = initial stride
% x = location
% eps = tolerance to stop refining

function [x_min, f_min, n_calls] = polyLineSearch(f, x, s, t, eps)
    n_calls = 0;
    tol = eps;

    bracket_found = false;
    bracket_step = 0;
    f0 = f(x + bracket_step*t*s);
    f1 = f(x + (bracket_step+1)*t*s);
    f2 = f(x + (bracket_step+2)*t*s);
    n_calls = n_calls + 3;

    while ~bracket_found
        if f1 < f2 && f1 < f0
            bracket_found = true;
        else
            f0 = f1;
            f1 = f2;
            f2 = f(x + (bracket_step+3)*t*s);
            n_calls = n_calls + 1;
            bracket_step = bracket_step + 1;
        end
    end
    
    a0 = bracket_step * t;
    a1 = (bracket_step+1) * t;
    a2 = (bracket_step+2) * t;

    while tol >= eps
        numer = (f0*a1^2 - f0*a2^2 - f1*a0^2 + f1*a2^2 + f2*a0^2 - f2*a1^2);
        denom =  (f0*a1 - f0*a2 - f1*a0 + f1*a2 + f2*a0 - f2*a1);

        if ~isfinite(denom) || abs(denom) < 1e-12
            break
        end

        a_new = (1/2) * numer/denom;
        
        if ~isfinite(a_new) || a_new <= a0 || a_new >= a2
            break
        end

        f_new = f(x + a_new*s);

        if ~isfinite(f_new)
            error('Function evaluation returned Inf or NaN.')
        end

        n_calls = n_calls + 1;
        if a_new < a1
            if f_new < f1
                a2 = a1;
                a1 = a_new;
                f2 = f1;
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

        f_evals = [f0 f1 f2];
        tol = (abs(max(f_evals) - min(f_evals)))/(max(1e-8, min(abs(f_evals))));
    end

    x_min = x + a1*s;
    f_min = f1;

end
