'''
    Fast projections onto the hypercube with a sum constraint (simplex).

    Two methods implemented: one that does direct search of kkt conditions,
    one that does iterative search (interpolation search).

    Uses numba to speed up computation.

    Benjamin Recht, March 6, 2017
    edited by Sarah Dean, March 7, 2017
'''
import numpy as np
#from numba import jit

#@jit('void(double[:], intp, double[:], intp[:])', nopython=True, nogil=True)
def project_kkt(z, r, zProj, sortOrder):
    '''
     Jitted code to compute the ppr projection.  Requires sort by numpy, but
     then goes to town.

     Input: z: a vector assumed to be a numpy array
            sortOrder: the arguments that sort z in descending order
            r: the targe sum

     Returns: zProj projected onto the set
    '''
    n = len(z)

    # check if there is a trivial solution
    if z[sortOrder[r-1]]>=z[sortOrder[r]]+1:
        for i in range(n):
            if i<r:
              zProj[sortOrder[i]] = 1
            else:
              zProj[sortOrder[i]] = 0
    else:
        # initialize theta to zero out all but first r components
        theta = -z[sortOrder[r]]

        # clip lower boundary
        C = -1
        # active set lower boundary
        A = -1
        # compute clipped and active set boundaries
        for i in range(n):
            if z[i] >= -theta:
                A+=1
                if z[i] >= 1-theta:
                    C+=1

        # compute sum of all entries in active set
        active_sum = 0.0
        for i in range(C+1,A+1):
            active_sum+=z[sortOrder[i]]

        while 1:
            # compute distance to increasing clipped set
            DeltaC = 1-theta-z[sortOrder[C+1]]
            # compute distance to increasing active set
            DeltaA = -theta-z[sortOrder[A+1]]
            Delta = min(DeltaA,DeltaC)

            # if we can take a step without changing active and clipped sets
            # and achive sum r, then we are done
            if C+1 + active_sum + (A-C)*(theta+Delta) >= r:
                beta = (r-C-1-active_sum)/(A-C)
                for i in range(n):
                    zProj[i] = max(min(z[i]+theta,1.0),0.0)
                break
            else:
                # otherwise, update the active and clipped sets, update
                # the sum of the active set, and update theta.
                if DeltaA==Delta:
                    while 1:
                        A+=1
                        active_sum = active_sum+z[sortOrder[A]]
                        if z[sortOrder[A+1]] < (-theta-Delta):
                            break
                if DeltaC==Delta:
                    while 1:
                        C+=1
                        active_sum = active_sum-z[sortOrder[C]]
                        if z[sortOrder[C+1]] < (1-theta-Delta):
                            break
                theta += Delta


#@jit('void(double[:], double, double[:], double)', nopython=True, nogil=True)
def project_is(z, r, zProj, tol=1.0e-6):
    '''
     Jitted code to compute the ppr projection via binary search.

     Input: z: a vector assumed to be a numpy array
            r: the targe sum
            tol: tolerance for the binary search

     Returns: zProj projected onto the set
    '''
    n = len(z)

    # compute minimum and maximum values of the input
    maxval = z[0]
    minval = z[0]
    for i in range(n):
        if z[i]>maxval:
            maxval = z[i]
        elif z[i]<minval:
            minval = z[i]

    # the alg runs long for large values
    #extreme = max(maxval, -minval)
    #if extreme > 1e+9:
    #    for i in range(n):
    #        z[i] = 100*z[i]/extreme
    #    maxval = 100*maxval/extreme
    #    minval = 100*minval/extreme

    # set incredibly conservative upper and lower bounds for theta
    L = -maxval
    U = 1-minval

    valU = 1.0*n
    valL = 0.0
    j = 0
    while 1:
        # interpolation search:
        theta = L + ((r - valL)/ (valU - valL))*(U - L)

        cur_sum = 0.0
        for i in range(n):
            zProj[i] = max(min(z[i]+theta,1.0),0.0)
            cur_sum += zProj[i]

        if abs(cur_sum-r)<r*tol:
            break
        elif cur_sum>r:
            U = theta
            valU = cur_sum
        else:
            L = theta
            valL = cur_sum
        j = j + 1
        if j>10000:
            raise ArithmeticError('no projection convergence')

def project(z, r, alg='is'):
    '''
     Computes a projection onto the subset of the hypercube that sums to r
     using the algorithms described in Barman, Liu, Draper, and Recht, 2011
     or by using interpolation search

     Input: z: a vector assumed to be a numpy array
            r: the target sum
            alg: desired algorithm to run (must be 'kkt' or 'is')

     Returns: zProj projected onto the set
    '''

    n = len(z)
    assert r<n
    #r = np.ceil(r) do we need this??

    # allocate memory for zProj
    zProj = np.zeros(n)

    if alg == 'kkt':
        # get the list of indices that sort z in descending order
        sortOrder = np.argsort(z)[::-1]
        project_kkt(z, r, zProj, sortOrder)
    elif alg == 'is':
        tol = 1.0e-6 # tolerance for quitting binary search
        try:
            project_is(z, r, zProj, tol)
        except ArithmeticError:
            print('extreme values in projection:',min(z),max(z), end=' ')
            raise ArithmeticError('no projection convergence')


    return zProj
