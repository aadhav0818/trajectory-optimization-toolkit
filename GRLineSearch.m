%% Golden Ratio Line Search Method for Minimizing N-D function
% f = function handle
% s = search direction
% t = initial stride
% x = location
% eps = tolerance to stop refining

function [x_min, f_min, n_calls, X_calls] = GRLineSearch(f, x, s, t, eps)
    n_calls = 0;
    X_calls = [];

    bracket_found = false;
    bracket_step = 0;
    f0 = f(x + bracket_step*t*s);
    f1 = f(x + (bracket_step+1)*t*s);
    f2 = f(x + (bracket_step+2)*t*s);
    X_calls = [X_calls, x, x+t*s, x+2*t*s];
    n_calls = n_calls + 3;

    while ~bracket_found
        if f1 < f2 && f1 < f0
            bracket_found = true;
        else
            f0 = f1;
            f1 = f2;
            f2 = f(x + (bracket_step+3)*t*s);
            X_calls = [X_calls, x + (bracket_step+3)*t*s];
            n_calls = n_calls + 1;
            bracket_step = bracket_step + 1;
        end
    end
    
    aL = bracket_step*t;
    aR = (bracket_step+2)*t;

    alpha = (3 - sqrt(5))/2;

    a1 = aL + alpha*(aR-aL);
    a2 = aR - alpha*(aR-aL);

    fL = f(x + aL*s);
    fR = f(x + aR*s);
    f1 = f(x + a1*s);
    f2 = f(x + a2*s);
    X_calls = [X_calls, x+aL*s, x+aR*s, x+a1*s, x+a2*s];
    n_calls = n_calls + 4;

    i = 0;

    tol = Inf;

    while tol >= eps

        fvals = [fL, f1, f2, fR];

        tol = (max(fvals)-min(fvals)) / max(1e-8, min(abs(fvals)));

        if tol < eps
            break;
        end

        if f1 > f2
            aL = a1;
            fL = f1;

            a1 = a2;
            f1 = f2;

            a2 = aR - alpha*(aR-aL);
            f2 = f(x + a2*s);
            X_calls = [X_calls, x+a2*s];
            n_calls = n_calls + 1;

        else
            aR = a2;
            fR = f2;

            a2 = a1;
            f2 = f1;

            a1 = aL + alpha*(aR-aL);
            f1 = f(x + a1*s);
            X_calls = [X_calls, x+a1*s];
            n_calls = n_calls + 1;
        end

        i = i + 1;
    end

    if f1 < f2
        a_min = a1;
        f_min = f1;
    else
        a_min = a2;
        f_min = f2;
    end

    x_min = x + a_min*s;
    

end
