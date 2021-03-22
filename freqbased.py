import pandas as pd
from .builtin import quantile, iqr

"""
Functions:
    1. min_freq
    2. max_freq
    3. mean_freq
    4. median_freq
    5. variance_freq
    6. stdev_freq
    7. range_freq
    8. mean_abs_dev_freq
    9. coef_of_var_freq
    10. outliers_freq
    11. outlier_c_freq"""
#Minimum
def min_freq(colData,**kwargs):
    
    """
    Return the minimum frequency count of the values in the requested axis.
    
                If you want the *index* of the minimum, use ``idxmin``. This is
                the equivalent of the ``numpy.ndarray`` method ``argmin``.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on

    axis : {index (0)}
        Axis for the function to be applied on.
    skipna : bool, default True
        Exclude NA/null values when computing the result.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    **kwargs
        Additional keyword arguments to be passed to the function.
    
    Returns
    -------
    scalar or Series (if level specified)
   
    Examples
    --------
     >>>s=pd.Series([2,3,3,4,3,3,2,1])
     >>ft.min_freq()
     1
"""
    freq=pd.Series.value_counts(colData)
    return pd.Series.min(freq,**kwargs)
    
#Maximum
def max_freq(colData,**kwargs):
    freq=pd.Series.value_counts(colData)
    return pd.Series.max(freq,**kwargs)
    """Return the maximum frequency count of the values in the requested axis.
    
                If you want the *index* of the maximum, use ``idxmax``. This is
                the equivalent of the ``numpy.ndarray`` method ``argmax``.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on

    axis : {index (0)}
        Axis for the function to be applied on.
    skipna : bool, default True
        Exclude NA/null values when computing the result.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    **kwargs
        Additional keyword arguments to be passed to the function.
    
    Returns:
    scalar or Series (if level specified)
   
    Examples:
     >>>s=pd.Series([2,3,3,4,3,3,2,1])
     >>ft.max_freq(s)
     4"""


#Mean
def mean_freq(colData,**kwargs):
    freq = pd.Series.value_counts(colData)
    return pd.Series.mean(freq,**kwargs)
    """Return the mean frequency count of the values in the requested axis.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    axis : {index (0)}
        Axis for the function to be applied on.
    skipna : bool, default True
        Exclude NA/null values when computing the result.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    **kwargs
        Additional keyword arguments to be passed to the function.
    
    Returns:
    scalar or Series (if level specified)
    
    Examples:
     >>>a=pd.Series([1,2,3,4,5,4,3,2,2,1]
     >>ft.mean_freq(a)
     2
     
    """
    


#Median
def median_freq(colData,**kwargs):
    """
    Return the median of the frequency values of the input series 
    
    Args:
    axis : {index (0)}
        Axis for the function to be applied on.
    skipna : bool, default True
        Exclude NA/null values when computing the result.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    **kwargs
        Additional keyword arguments to be passed to the function.
    
    Returns:
    scalar or Series (if level specified)
    
    Example:
    >>>>a=pd.Series([1,2,3,4,5,4,3,2,2,1]) 
    >>ft.median_freq(a)
    2
"""
   
    freq = pd.Series.value_counts(colData)
    return pd.Series.median(freq,**kwargs)

#Variance
def variance_freq(colData, **kwargs):
    """

    Return unbiased variance over frequency values of input series.
    
    Normalized by N-1 by default. This can be changed using the ddof argument
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    axis : {index (0)}
    skipna : bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    
    Returns:
    scalar or Series (if level specified)
    
    Example:
    >>a=pd.Series([1,2,3,4,5,4,3,2,2,1]) 
    >>ft.variance_freq(a)
    0.5
    
    
"""
    freq = pd.Series.value_counts(colData)
    return pd.Series.var(freq,**kwargs)

#Standard Deviation
def stdev_freq(colData,**kwargs):
    """Return sample standard deviation over frequency values of input series.
    
    Normalized by N-1 by default. This can be changed using the ddof argument
    
    Args:
    colData (array_like, 1D) : Pandas Series of Data or Dataframe Column for function to be applied on
    axis : {index (0)}
    skipna : bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    
    Returns:
    scalar or Series (if level specified)
    
    Example:
    >>a=pd.Series([1,2,3,4,5,4,3,2,2,1]) 
    >>ft.stdev_freq(a)
    0.707107

    """
    freq = pd.Series.value_counts(colData)
    return pd.Series.std(freq,**kwargs)

#Range
def range_freq(colData):
    """
    Range of Frequency
    Returns the difference between the highest frequency value and lowest frequency value in the input series object.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    range : int
        range = Maximum value - Minimum value
   
    Example:
    >>>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>>ft.range_freq(a)
    2"""
    freq = pd.Series.value_counts(colData)
    return pd.Series.max(freq)-pd.Series.min(freq)


def mean_abs_dev_freq(colData):
    """Return the mean absolute deviation of the values for the requested axis.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    axis : {index (0)}
        Axis for the function to be applied on.
    skipna : bool, default True
        Exclude NA/null values when computing the result.
    level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.
    numeric_only : bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
    **kwargs
        Additional keyword arguments to be passed to the function.
    
    Returns:
    scalar or Series (if level specified)
    
    Example:
    >>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>ft.mean_abs_dev_freq(colData)
    0.4
    """
    
    freq = pd.Series.value_counts(colData)
    return pd.Series.mad(freq)





# Coefficient Of Variation
def coef_of_var_freq(colData):
    """Coefficient of Variance of Frequency
    Returns the coefficient of variance of frequency values from the data in input series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Coefficient of Variance: float
        coefficient of variance = standard deviation/mean
    
    Example:
    >>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>ft.coef_of_var(a)
    0.353553
    """
    freq=pd.Series.value_counts(colData)
    return pd.Series.std(freq)/pd.Series.mean(freq)

def outliers_freq(colData):
    """Frequency Outliers
    Returns list of outliers of frequency values from input series.
    Outliers are determined using the quartiles of the data. Upper and lower bounds are calculated which are 1.5*IQR higher and lower than the 3rd     and 1st quartiles respectively. Data values higher than the upper bound or lower than the lower bound are considered outliers.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    list of outliers : list
    
    Example:
    >>>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>>ft.outliers_freq(c)
    []
    """
    freq = pd.Series.value_counts(colData)
    lowerbound = quantile(freq,0.25) -(1.5* iqr(freq))
    upperbound = quantile(freq, 0.75)+(1.5*iqr(freq))
    result=[]
    for x in freq:
        if (x<lowerbound) or (x>upperbound):
            result.append(x)
        else:
            pass
    return result

#Number of Outliers
def outlier_c_freq(colData):
    """
    Number of Outliers of Frequency
    Returns the number of outliers in frequency values of input series.  
    Outliers are determined using the quartiles of the data. Upper and lower bounds are calculated which are 1.5*IQR higher and lower than the 3rd     and 1st quartiles respectively. Data     values higher than the upper bound or lower than the lower bound are considered outliers.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Number of Outliers : int
   
    Example:
    >>>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>>ft.outlier_c_freq(a)
    0
    
    """
    freq = pd.Series.value_counts(colData)
    return len(outliers_freq(colData))
