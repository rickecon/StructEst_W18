
# coding: utf-8

# In[13]:


# distributions.py

# This file contains the distributions:
# Gamma: GA(x; alpha, beta)
# Generalized Gamma: GG(x; alpha, beta, m)
# Generalized Beta 2: GB2(x; a, b, p, q)
# Normal: N(x; mu, sigma)

# NB: not solving for the exception where xvals = 0

import scipy.special as spc
import numpy as np

def gamma_pdf(xvals, alpha, beta):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the gamma pdf with shape alpha and scale
    beta. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the gamma distributed random
             variable
    alpha  = scalar > 0, shape parameter of the gamma distribution
    beta   = scalar > 0, scale parameter of the gamma distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, gamma PDF values for alpha and beta
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''        
    pdf_vals = (1/((beta**alpha) * spc.gamma(alpha))) * (xvals**(alpha-1)) * np.exp(-xvals/beta)
    
    return pdf_vals

def gengamma_pdf(xvals, alpha, beta, m):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the generalized gamma pdf with shape alpha, 
    scale beta and parameter m
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the generalized gamma distributed random
             variable
    alpha  = scalar > 0, shape parameter of the generalized gamma distribution
    beta   = scalar > 0, scale parameter of the generalized gamma distribution
    m      = scalar > 0, parameter of the generalized gamma distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, generalized gamma PDF values for alpha, beta and m
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''        
    pdf_vals    = (m/((beta**alpha) * spc.gamma(alpha/m)
                     )) * (xvals**(alpha-1)) * np.exp(-(xvals/beta)**m)
    
    return pdf_vals

def genbeta2_pdf(xvals, a, b, p, q):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the generalized beta 2 pdf with parameters
    a, b, p, q
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the generalized beta 2 distributed random
             variable
    a      = scalar > 0, parameter of the generalized beta 2 distribution
    b      = scalar > 0, parameter of the generalized beta 2 distribution
    p      = scalar > 0, parameter of the generalized beta 2 distribution
    q      = scalar > 0, parameter of the generalized beta 2 distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, generalized beta 2 PDF values for a, b, p, q
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''        
    pdf_vals    = (a * (xvals**(a * p - 1)))/((b**(a * p)
                                             ) * spc.beta(p,q) * ((1 + (xvals/b)**a)**(p + q)))
    
    return pdf_vals

def norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''

    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))))
    
    return pdf_vals

