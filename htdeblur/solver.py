import numpy as np
from . import blurkernel
import scipy
import scipy.misc
#from numba import jit


class kernel_objectives(object):
    """functions related to illumination optimization

    Attributes:
        kernelMapCol: function returning columns of kernel map
        smoothing: parameter for smoothing objective functions
        precomputed: boolean indicated presence of precomputed kernel map
        (optional) kernelMap: precomputed full map
        (optional) KTK: kernel map conjugate transposed with itself
    """

    def __init__(self, *args, **kwargs):
        """object initialization

        Args:
            kernelMapCol: A function which returns columns of the kernel map
            smoothing: initial smoothing parameter
            kernelMap (kwarg): the (optional) precomputed map
        """
        self.kernelMapCol = args[0]
        self.smoothing = args[1]
        self.precomputed = False
        if 'kernelMap' in kwargs:
            self.kernelMap = kwargs['kernelMap']
            self.precomputed = True
        if 'kernelConjTrans' in kwargs:
            self.KTK = kwargs['kernelConjTrans']

    def getKernelMapCol(self, i):
        """function which gives access to columns of kernel map

            Args:
                i: the index of the desired column

            Returns:
                a column of the kernel map at index i
            """
        if self.precomputed:
            return self.kernelMap[:,i]
        else:
            return self.kernelMapCol(i)

######################## Objective Functions ####################

    def conditionNumber(self, v):
        """condition number of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed condition number
        """
        s = self.squaredSv(v)
        return np.sqrt(np.amax(s)/np.amin(s))

    def minSvSquared(self, v):
        """min squared singular value of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            # computed min singular value
        """
        s = self.squaredSv(v)
        return np.amin(s)

    def averageSvSquared(self, v):
        """average squared singular value of kernel with illumination v. computed via Parseval's

        Args:
            v: the illumination vector

        Returns:
            computed average singular value
        """
        prod =  v.T.dot(self.KTK.dot(v))
        #print(prod.imag)
        return prod.real

    def maxSvSquared(self, v):
        """max squared singular value of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed max singular value
        """
        s = self.squaredSv(v)
        return np.amax(s)

    def minSvSquaredSmooth(self, v):
        """smoothed (via logsumexp) min suared singular value of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed smoothed min singular value
        """
        s = self.squaredSv(v)
        return self.smoothedMin(s, self.smoothing)

    def svSquaredReciprocalSumSmooth(self, v):
        """smoothed sum of reciprocal of squared singular values of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed smoothed sum of reciprocal of squared singular values
        """
        s = self.squaredSv(v)
        return -sum(1/(s+1/self.smoothing))

    def svSquaredReciprocalSum(self, v):
        """sum of reciprocal of squared singular values of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed sum of reciprocal of squared singular values
        """
        s = self.squaredSv(v)
        return sum(1/(s))

    def squaredSv(self, v):
        """squared singular values of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed squared singular values
        """
        Kv = self.dotKernelMap(v)
        s = np.square(Kv.real) + np.square(Kv.imag)
        assert np.all(s >= 0), "squared singular value is negative?"
        return s

######################## Deprecated Functions ####################

    def nonsmoothMinBlurSquared(self, v):
        print('nonsmoothMinBlurSquared is deprecated, use minSvSquared')

    def smoothedMinBlurSquared(self, v):
        print('smoothedMinBlurSquared is deprecated, use minSvSquaredSmooth')

    def sumRecBlur(self, v):
        print('sumRecBlur is deprecated, use svSquaredReciprocalSumSmooth')

    def sumRecBlurNonsmooth(self, v):
        print('sumRecBlurNonsmooth is deprecated, use svSquaredReciprocalSum')

######################## Objective Gradients ####################

    def gradAverageSvSquared(self, v):
        """gradient of average squared singular value of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed gradient of average squared singular value
        """
        return 2*self.KTK.dot(v)

    def gradient(self, gradf, v):
        """gradient of a function of the squared singular values.

        The objective is defined by f(s), s_j=|Kv|_j^2

        Args:
            gradf: the gradient of f w.r.t. s
            v: the illumination vector

        Returns:
            computed gradient of f w.r.t. v
        """
        # todo: can be sped up in precomputed case
        gradobj = np.zeros(v.size)
        Kv = self.dotKernelMap(v)
        s = np.square(Kv.real) + np.square(Kv.imag)
        gradfs = gradf(s)
        for i in range(0,v.size):
            colK = self.getKernelMapCol(i)
            # todo: investigate imag and read-only
            gradobj[i] = self.gradDotHelper(colK.real.astype(np.float32), colK.imag.astype(np.float32), Kv.real.astype(np.float32), Kv.imag.astype(np.float32), gradfs.astype(np.float32))
        return gradobj

    def gradMinSvSquared(self, v):
        """gradient of min squared singular value of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed gradient of min squared singular value
        """
        gradf = lambda s : self.gradSmoothMin(s)
        return self.gradient(gradf, v)

    def gradSvSquaredReciprocalSum(self, v):
        """gradient of sum of reciprocal of squared singular value of kernel with illumination v

        Args:
            v: the illumination vector

        Returns:
            computed gradient of sum of reciprocal of squared singular value
        """
        gradf = lambda s : self.gradRecSum(s)
        return self.gradient(gradf, v)

    def gradSmoothObj(self, v):
        print('gradSmoothObj is deprecated, use gradMinSvSquared')

    def gradSumObj(self, v):
        print('gradSumObj is deprecated, use gradMinSvSquared')


######################## Helper Functions ####################

    def dotBlurMap(self, v):
        print('dotBlurMap is deprecated, use dotKernelMap')

    def dotKernelMap(self, v):
        """ multiplication of illumination vector and kernal map column by column
        Args:
            v: the illumination vector

        Returns:
            K.dot(v)
        """
        # todo: can improve for precomputed?

        if self.precomputed:
            return self.kernelMap.dot(v)
        else:
            result = v[0]*self.getKernelMapCol(0)
            for i in range(1, v.size):
                result = result + v[i]*self.getKernelMapCol(i)
        return result

    @staticmethod
    def smoothedMin(x, smoothing):
        """ smoothed minimum function via logsumexp
        Args:
            x: vector over which we are approximating minimum element
            smoothing: smoothing factor (larger -> better approximation to min, but less smooth function)

        Returns:
            approximated minimum element of x
        """
        return -scipy.misc.logsumexp(-x*smoothing) / smoothing

    @staticmethod
    def gradSmoothMin(x, smoothing):
        """ gradient of smoothed minimum function
        Args:
            x: vector over which we are approximating minimum element
            smoothing: smoothing factor (larger -> better approximation to min, but less smooth function)

        Returns:
            gradient of smoothed min
        """
        n = x.size
        t = np.amin(x) # avoiding overflow
        expx = np.exp(-smoothing*(x - t))
        return expx / (np.ones(n).dot(expx))


    def gradRecSum(self, x):
        """ gradient of smoothed sum reciprocal function
            f(x) = sum(1/x)

        Args:
            x: input argument

        Returns:
            gradient of smoothed sum reciprocal
        """
        return 1/(x+1/self.smoothing)**2

    @staticmethod
    #@jit('float32(float32[:], float32[:], float32[:], float32[:], float32[:])', nopython=True, nogil=True)
    def gradDotHelper(b_real, b_imag, Bv_real, Bv_imag, gradm):
        MN = gradm.size
        result = 0
        for i in range(0,MN):
            x = 2*(b_real[i]*Bv_real[i] + b_imag[i]*Bv_imag[i])
            result = result + x*gradm[i]
        return result

###### DEPRECIATED ######
    # def gradAbsSquared(self, v):
    #     partial_v = np.zeros([self.MN, v.size]) #to do replace with (mn, pl)
    #     self.gradAbsChainRule(partial_v, v, self.blurMap.real, self.blurMap.imag, 0, 0) #replace with functions
    #     return partial_v

    # @staticmethod
    # @jit('void(double[:,:], double[:], double[:,:], double[:,:], double, double)', nopython=True, nogil=True)
    # def gradAbsChainRule(partials, v, Br, Bi, Brv, Biv):
    #     MN, n = partials.shape
    #     for i in range(0,MN): # each row of B
    #         Brv = 0
    #         Biv = 0
    #         for j in range(0,n):
    #             Brv = Brv + Br[i,j]*v[j]
    #             Biv = Biv + Bi[i,j]*v[j]
    #         for j in range(0,n):
    #             partials[i,j] = 2*(Br[i,j] * Brv +  Bi[i,j] * Biv)


##### NOT FAST #######
    # def gradSmoothObjFast(self, v):
    #     gradobj = np.zeros(v.size)
    #     Bv = self.blurMap.dot(v)
    #     Bvsq = np.square(Bv.real) + np.square(Bv.imag)
    #     gradm = self.gradSmoothMin(Bvsq, self.scaling)
    #     self.gradFastHelper(gradobj, self.blurMap.real, self.blurMap.imag, Bv.real, Bv.imag, gradm)
    #     return gradobj

    # def gradSumObjFast(self, v):
    #     gradobj = np.zeros(v.size)
    #     Bv = self.blurMap.dot(v)
    #     Bvsq = np.square(Bv.real) + np.square(Bv.imag)
    #     grads = self.gradRecSum(Bvsq)
    #     self.gradFastHelper(gradobj, self.blurMap.real, self.blurMap.imag, Bv.real, Bv.imag, grads)
    #     return gradobj

    # @staticmethod
    # @jit('void(double[:], double[:,:], double[:,:], double[:], double[:], double[:])', nopython=True, nogil=True)
    # def gradFastHelper(gradobj, B_real, B_imag, Bv_real, Bv_imag, gradm):
    #     for j in range(0,gradobj.size):
    #         MN = gradm.size
    #         result = 0
    #         for i in range(0,MN):
    #             x = 2*(B_real[i,j]*Bv_real[i] + B_imag[i,j]*Bv_imag[i])
    #             result = result + x*gradm[i]
    #         gradobj[j] = result





###### DEPRECIATED ######
    # def gradSmoothObjOld(self, v):
    #     dx = self.gradAbsSquared(v)
    #     Bv = self.blurMap.dot(v)
    #     Bvsq = np.square(Bv.real) + np.square(Bv.imag)
    #     gradm = self.gradSmoothMin(Bvsq, self.scaling)
    #     gradobj = np.zeros(v.size)
    #     self.gradDotChainRule(gradobj, dx, gradm)
    #     return gradobj

###### DEPRECIATED ######
    # @staticmethod
    # @jit('void(double[:], double[:,:], double[:])', nopython=True, nogil=True)
    # def gradDotChainRule(gradobj, dx, gradm):
    #     MN,n = dx.shape
    #     for k in range(0,n):
    #         for i in range(0, MN):
    #             gradobj[k] = gradobj[k] + dx[i,k] * gradm[i]


#### NOT FASTER ###
    #import logging
    #numba.codegen.debug.logger.setLevel(logging.INFO) # numba logging bug http://stackoverflow.com/questions/19112584/huge-errors-trying-numba
    # from http://stackoverflow.com/questions/20195435/how-to-write-a-fast-log-sum-exp-in-cython-and-weave/20242943
    # @staticmethod
    # @jit('double(double[:])')
    # def logsumexp(a):
    #     result = 0.0
    #     largest_in_a = 0.0
    #     for i in range(a.shape[0]): # numba is slow when using max or np.max, so re-implementing
    #         if (a[i] > largest_in_a):
    #             largest_in_a = a[i]
    #     for i in range(a.shape[0]):
    #         result += np.exp(a[i] - largest_in_a)
    #     return np.log(result) + largest_in_a




###### DEPRECIATED ######
    # @staticmethod
    # # note: currently does not optimize for system aperature by cropping
    # def generateBlurKernelMap(n,M,N):
    #     imgSize = np.array([M,N])
    #     posListLinear = blurkernel.genLinearBlurKernelMapPosList(imgSize,n,1)
    #     blurKernelMap = blurkernel.positionListToBlurKernelMap(imgSize,posListLinear)
    #     blurKernelMap_f = blurkernel.Ft(blurKernelMap)
    #     kernelMap = blurKernelMap_f.reshape(np.prod(imgSize),np.size(blurKernelMap_f,2))
    #     return kernelMap

    @staticmethod
    def gradtest(func, gradfunc, n):
        """ function that tests the accuracy of a gradient
        print an approximate gradient and gradient computed with gradfunc. They should be very close.

        Args:
            func: function whose gradient we are testing
            gradfunc: gradient of the function
            n: size of inputs to f
        """
        x0 = np.random.rand(n)
        x0 = n*0.3*x0/sum(x0)
        eps = 1e-4
        d = np.random.rand(n)
        d = d / sum(d)
        estimate = (func(x0 + eps*d) - func(x0 - eps*d))/(2*eps)
        print(estimate, gradfunc(x0).dot(d))
