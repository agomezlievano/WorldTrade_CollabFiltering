# density_generators.py

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Useful utility for dataframes
def ends(df, x=5):
    """
    Similar to pd.DataFrame.head() or pd.DataFrame.tail(),
    but it appends both head and tail.
    """
    return df.head(x).append(df.tail(x))
setattr(pd.DataFrame,'ends',ends)
setattr(pd.Series,'ends',ends)


# Implied Rating Class Object
class ImpliedRating(BaseEstimator, TransformerMixin):
    """
    The class creates several types of "densities". A "density" is an
    estimate of the value $X_{c,p}$ of a matrix. The methodologies mimic
    the problem of collaborative filtering, in which one weeks to predict
    the rating of "items" by "users", and generate recommendations 
    based on the estimates. Applied to economic diversification, we 
    refer to these estimates as "densities".

    Parameters
    ----------
    method : array of strings (default = 'item-based')
        Which collaborative filtering method to use. Options:
        ['item-based', 'user-based', 'joint-based', 
        'athey', 'dotembedding', 'svd']
    
    similarity : string (default=None) 
        If method is 'item-based' or 'user-based', this parameter 
        needs to be chosed. Options:
        ['pearson', 'spearman', 
        ]
            
    max_iters : integer (default=100)
    
    doprint : boolean, optional (default=False)
        Prints different results of the estimation steps.

    

    Attributes
    ----------
    Lest_ : array-like, shape (N, T)
        This is the estimate of L.
    
    loss_ : float
        The root-square of the mean square error of the observed elements.
        
    iters_ : int
        Number of iterations it took the algorithm to get the desired
        precision given by the parameter 'epsilon'.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1,2,np.nan, 0],[np.nan,2,2,1],[3,0,1,3],[0,0,2,np.nan]])
    >>> observedset = np.nonzero(~np.isnan(data))
    >>> my_mcnnm = MatrixCompletion_NNM(setOk=observedset, lamb=2.5, epsilon=10**(-6), doprint=False)
    >>> print(data)
    [[  1.   2.  nan   0.]
    [ nan   2.   2.   1.]
    [  3.   0.   1.   3.]
    [  0.   0.   2.  nan]]
    >>> print(my_mcnnm.fit(data))
    MatrixCompletion_NNM(copy=True, doprint=False, epsilon=1e-06, lamb=2.5,
               max_iters=100, printbatch=10,
               setOk=(array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=int64), 
               array([0, 1, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=int64)))
    >>> print(my_mcnnm.transform(data))
    [[ 0.98082397  2.00915981  3.49022202  0.01813745]
     [ 1.55601129  1.33063721  2.38547663  0.97025289]
     [ 3.00567286  0.33134019  0.80761928  3.00948113]
     [ 0.0419737   0.87320627  1.48553996 -0.39839039]]
    >>> print(my_mcnnm.fit_transform(data))
    [[ 0.98082397  2.00915981  3.49022202  0.01813745]
     [ 1.55601129  1.33063721  2.38547663  0.97025289]
     [ 3.00567286  0.33134019  0.80761928  3.00948113]
     [ 0.0419737   0.87320627  1.48553996 -0.39839039]]
    >>> print(my_mcnnm.transform([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]))
    [[ 0.98082397  2.00915981  3.49022202  0.01813745]
     [ 1.55601129  1.33063721  2.38547663  0.97025289]
     [ 3.00567286  0.33134019  0.80761928  3.00948113]
     [ 0.0419737   0.87320627  1.48553996 -0.39839039]]
    
    Notes
    -----
    
    
    """
    

    def __init__(self, setOk=None, #missing_values=np.nan, 
                 lamb=0, epsilon=0.001, max_iters=100, 
                 doprint=False, printbatch=10, copy=True):
        self.setOk = setOk 
        #self.missing_values = missing_values 
        self.lamb = lamb 
        self.epsilon = epsilon 
        self.max_iters = max_iters 
        self.doprint = doprint 
        self.printbatch = printbatch
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the estimator to the matrix X (which is 
        really the matrix Y in Athey et al.'s paper).
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : Ignored
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
            
        Returns
        -------
        self : object
            Returns self.
        """
        
        # First, I should check that the missing values are right
        #Ynew = np.zeros_like(self.Ymat)
        #Ynew = self.missing_values
        #Ynew[self.setOk] = self.Ymat[self.setOk]

        assert (len(self.setOk) > 0), "setOk should be a non-empty array"
        assert (self.lamb > 0), "lamb is the lambda parameter which should be larger than zero"

        N, T = X.shape

        # Initialize L to the observed (non-missing) values of Y given by the set setOk
        Lprev = PO_operator(X, self.setOk)

        # Initialization of error with a highvalue and the iteration
        error = N*T*10**3
        iteration = 0

        while((error > self.epsilon) and (iteration < self.max_iters)):
            Lnext = shrink(PO_operator(X, self.setOk) + 
                           POcomp_operator(Lprev, self.setOk), lamb = self.lamb)

            # Updating values
            Lprev = Lnext.copy()
            error = mcnnm_loss(X, Lprev, self.setOk)
            iteration = iteration + 1

            if(self.doprint and (iteration%self.printbatch==0 or iteration==1)):
                print("Iteration {}\t Current loss: {}".format(iteration, error))

        if(self.doprint):
            print("")
            print("Final values:")
            print("Iteration {}\t Current loss: {}".format(iteration, error))
            print("")
            print(X)
            print(np.round(Lnext, 2))
        
        self.iters_ = iteration
        self.loss_ = error
        self.Lest_ = Lnext
        
        # Return the transformer
        return self

    def transform(self, X):
        """ 
        Actually returning the estimated matrix, in which we have 
        imputed the missing values of X (matrix Y).
        
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (N, T)
            The input data to complete.
            
        Returns
        -------
        X_transformed : array, shape (N, T)
            The array the completed matrix.
        """
        try:
            getattr(self, "Lest_")
        except AttributeError:
            raise RuntimeError("You must estimate the model before transforming the data!")
        
        return self.Lest_
