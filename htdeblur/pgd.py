import numpy as np

# todo: don't require smoothing?
def projectedIterativeMax(x0, obj, f, gradf, proj, nextstep, smoothing, max_iter, threshold=0.005, verbose=True):
    """ projected gradient method for finding the maximum of a (smoothed) function

    Args:
        x0: initial point
        obj: objective functin object for tuning smoothing parameter
        f: function to maximize (should be a function within obj)
        gradf: gradient of f (should be a function within obj)
        nextstep: method for finding the next iterate.
                  Takes two previous iterates, f, gradf, projection method, and iteration number.
                  Returns the next iterate and the function value at the next iterate.
        smoothing: scheme for increasing the smoothing parameter as the optimization progresses.
                   function which takes in iteration numbers and outputs smoothing values.
        max_iter: maximum iterations to artificially stop the optimization even if convergence has not been detected.
        threshold: threshold for detecting iterate convergence
        verbose: argument for printing the iteration number

    Returns:
        x: the optimal argument
        i: the number of iterations performed
    """
    n = x0.size
    xprev = x0
    i = 0
    fprev = f(x0)
    while (i+1 < max_iter):
        if verbose:
            print(i, end=' ')
        obj.smoothing = smoothing(i+1)
        x, fval = nextstep(xprev, xprev, f, gradf, proj, i)
        if np.linalg.norm(x-xprev) < threshold and i>5:
            if verbose: print('converged:', np.linalg.norm(x-xprev), threshold)
            return x, i
        fprev = fval
        xprev = x
        i = i+1
    return x, i

def projectedIterativeMax_developement(x0, obj, f, gradf, proj, nextstep, smoothing, max_iter, verbose=True):
    """ verbose projected gradient method for finding the maximum of a (smoothed) function

    Args:
        x0: initial point
        obj: objective functin object for tuning smoothing parameter
        f: function to maximize (should be a function within obj)
        gradf: gradient of f (should be a function within obj)
        nextstep: method for finding the next iterate.
                  Takes two previous iterates, f, gradf, projection method, and iteration number.
                  Returns the next iterate and the function value at the next iterate.
        smoothing: scheme for increasing the smoothing parameter as the optimization progresses.
                   function which takes in iteration numbers and outputs smoothing values.
        max_iter: number of iterations to run
        verbose: argument for printing the iteration number

    Returns:
        x: all iterates
        fval: all function values

    To do: merge with other method
    """
    n = x0.size
    x = np.zeros([n,max_iter])
    fval = np.zeros(max_iter)
    x[:,0] = x0
    i = 0
    fval[0] = f(x0)
    while (i+1 < max_iter):
        if verbose:
            print(i, end=' ')
        obj.smoothing = smoothing(i+1)
        x[:,i+1], fval[i+1] = nextstep(x[:,i], x[:, i-1], f, gradf, proj, i) # maximizing => direction of gradient
        i = i+1
    return x, fval

def projectedIterativeMax_continuation(x0, obj, f, gradf, proj, nextstep, smoothing, max_iter, threshold=0.005, verbose=True):
    """ projected gradient method for finding the maximum of a (smoothed) function
        smoothing performed in a continuation scheme rather than at each step
        found not to work as well.

    Args:
        x0: initial point
        obj: objective functin object for tuning smoothing parameter
        f: function to maximize (should be a function within obj)
        gradf: gradient of f (should be a function within obj)
        proj: projection method
        nextstep: method for finding the next iterate.
                  Takes two previous iterates, f, gradf, projection method, and iteration number.
                  Returns the next iterate and the function value at the next iterate.
        smoothing: scheme for increasing the smoothing parameter as the optimization progresses.
                   function which takes in iteration numbers and outputs smoothing values.
        max_iter: number of iterations to run
        verbose: argument for printing the iteration number

    Returns:
        x: the optimal argument
        icount: the number of iterations performed
    """
    n = x0.size
    icount = 0
    xprev = x0
    i = 1
    while(i+1 < max_iter/10): # max epochs
        if verbose:
            print(i, end=' ')
        obj.smoothing = smoothing(i+1)
        x, j = projectedIterativeMax(x0, obj, f, gradf, proj, nextstep, lambda i: obj.smoothing, max_iter, threshold, verbose)
        icount = icount + j
        if np.linalg.norm(x-xprev) < threshold:
            return x, icount
        xprev = x
        i = 1+i
    return x, icount


def decayingstep(x, xprev, f, gradf, proj, i):
    """a step method which uses a simple decaying step size rule

    Args:
        x: current iterate
        xprev: previous iterate (not used)
        f: function
        gradf: gradient of the function
        proj: projection method
        i: iteration number

    Returns:
        next iterate and function value at iterate
    """
    a = 0.1
    step = proj(x + a/(i+1)*gradf(x))
    return step, f(step)

def backtrackingstep(x, xprev, f, gradf, proj, i):
    """a step method which uses backtracking linesearch

    Args:
        x: current iterate
        xprev: previous iterate (not used)
        f: function
        gradf: gradient of the function
        proj: projection method
        i: iteration number

    Returns:
        next iterate and function value at iterate
    """
    gradfx = gradf(x)
    t = 1
    a = 0.9
    step = proj(x + t*gradfx)
    i = 0
    fstep = f(step)
    while (fstep < f(x)+0.0004*t*gradfx.dot(gradfx)) and (i<100):
        t = t*a
        i = i+1
        step = proj(x + t*gradfx)
        fstep = f(step)
    return step, fstep

def linesearchstep(x, xprev, f, gradf, proj, i):
    """a step method which uses weak wolfe linesearch

    Args:
        x: current iterate
        xprev: previous iterate (not used)
        f: function
        gradf: gradient of the function
        proj: projection method
        i: iteration number

    Returns:
        next iterate and function value at iterate
    """
    gradfx = gradf(x)
    t = 1
    step = proj(x - t*gradfx)
    i = 0
    beta = np.inf
    alpha = 0
    while (i<100):
        if (f(step) < f(x)+0.0004*t*gradfx.dot(gradfx)):
            beta = t
            t = (alpha+beta)/2
        elif gradf(step).dot(gradfx) > 0.9*gradfx.dot(gradfx):
            alpha = t
            if np.isfinite(beta):
                t = (alpha+beta)/2
            else:
                t = 2*alpha
        else:
            break
        i = i+1
        step = proj(x + t*gradfx)
    return step, f(step)

# inefficient
def twopointstep(x, xprev, f, gradf, proj, i):
    """a step method which uses a two point step size method. found to be inefficient.

    Args:
        x: current iterate
        xprev: previous iterate
        f: function
        gradf: gradient of the function
        proj: projection method
        i: iteration number

    Returns:
        next iterate and function value at iterate
    """
    deltax = x - xprev
    deltag = gradf(x) - gradf(xprev)
    a = deltax.dot(deltax)/deltax.dot(deltag)
    step = proj(x - a*gradf(x))
    return step, f(step)

def smoothing_none(i):
    return 10000

def smoothing_quad(i):
    return i**2

def smoothing_pow(i):
    return np.power(1.1,i)

def randomsearch(f, n, proj, max_iter):
    xbest = np.zeros(n)
    for i in range(0, max_iter):
        x = proj(np.random.rand(n))
        if f(xbest) < f(x):
            xbest = x
    return xbest

def projectedIterativeMaxTest(x0, f, gradf, proj, nextstep, max_iter, verbose=True):
    n = x0.size
    x = np.zeros([n,max_iter])
    fval = np.zeros(max_iter)
    x[:,0] = x0
    i = 0
    while (i+1 < max_iter):
        if verbose:
            print(i,  end=' ')
        x[:,i+1],fval[i+1] = nextstep(x[:,i], x[:, i-1], f, gradf, proj, i) # maximizing => direction of gradient
        i = i+1
    return x, fval
