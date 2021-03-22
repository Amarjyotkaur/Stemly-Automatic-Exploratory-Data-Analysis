import builtins
import pandas as pd
import statistics
import numpy as np
from collections import Counter
"""FUNCTIONS:
    
    1. minimum
    2. maximum
    3. mean
    4. median
    5. variance
    6. std_dev
    7. size
    8. type
    9. uniq_values
    10. cardinality_c
    11. cardinality_r
    12. count_c
    13. most_least_frequent
    14. count_bins
    15. sum
    16. count_r
    17. mode
    18. range
    19. null_c
    20. null_r
    21. quantile
    22. iqr
    23. kurtosis
    24. skewness
    25. mean_abs_dev
    26. memory
    27. outlier
    28. outlir_c
    29. zero_c
    30. coef_of_var
    31. missing_int_c
    32. missing_int_r
    33. att_class

    """

#Minimum
min = pd.Series.min

#Maximum
max = pd.Series.max

#Mean
mean = pd.Series.mean

#Median
median = pd.Series.median

#Variance
variance = pd.Series.var

#Standard Deviation
std_dev = pd.Series.std

#Size
def size(colData):
    
    """Size of Column Data
        Returns the numbr of elements in a column of data.

        Args:
        colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on

        Returns:
        element_count : int
            Number of elements along the axis.

        Example
        >>>ft.size([4,5,3,2])
        4
        """
    return len(colData)


#Data Type
def type(colData):
    return colData.dtype
type.__doc__=pd.Series.dtype.__doc__

#List of Unique Values
uniq_values = pd.Series.unique  # returns array of unique values

#Distinct Count
def cardinality_c(colData):
    """Cardinality Count
        Return number of unique values of Series object.

        Args:
        colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on

        Returns:
        element_count : int
            Number of unique values in input Series Object.

        Example:
        >>>ft.cardinality_c([2,3,4,3,2,1])
        4
        """
    return len(pd.Series.unique(colData))


#Cardinality ratio
def cardinality_r(colData):
    """Cardinality Ratio
        Returns ratio of the number of unique values to the total number of values in Series object.

        Args:
        colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on

        Returns:
        Ratio : float
            No. of unique values/total number of values

        Example:
        >>>ft.cardinality_c([2,3,4,3,2,1])
        0.6666666666666666
        """
    #output = len(pd.Series.unique(*args,**kwargs))/pd.Series.size(*args,**kwargs)
    output = cardinality_c(colData)/ size(colData)
    return output


#Count
count_c = pd.Series.value_counts #outputs table of counts for each value in series

#3 Most and Least Frequent Values:
def most_least_frequent(colData):
    """3 Most and Least Frequent Values
    Returns list of top 3 and bottom 3 most frequently occuring values in input series
    
    Args:
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
     List of Top 3 values: list
     List of Bottom 3 values: list
     
    Example: 
    >>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>ft.most_least_frequent(a)
    'Top 3 Frequent Values:', [2, 1, 3], 'Bottom 3 Frequent Values:', [3, 4, 5]
     
    """
    ls=[item for items, c in Counter(colData).most_common() for item in [items] * c]
    ls = uniq_values(pd.Series(ls))
    return "Top 3:", ls[:3].tolist(), "Bottom 3:",ls[-3:].tolist()


#Bin Count
def count_bins(colData,**kwargs):
    """
    Bin Count
    Returns a Series containing counts of values segmented in 10 bins for both numerical and categorical data.
    Catgeorical Data are segmented by the starting letter of each string,following the alphabetical order
    
    The resulting object will be in descending order so that the
    first element is the most frequently-occurring element.
    Excludes NA values by default.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    normalize : bool, default False
        If True then the object returned will contain the relative
        frequencies of the unique values.
    sort : bool, default True
        Sort by frequencies.
    ascending : bool, default False
        Sort in ascending order.
    bins : int, optional
        Rather than count values, group them into half-open bins,
        a convenience for ``pd.cut``, only works with numeric data.
    dropna : bool, default True
        Don't include counts of NaN.
        
    
    Example:
    >>d =pd.Series(["cat","monkeys", "a","dog","monkey","dog","cat","dog"])
    >>ft.count_bins(d)
    
    
    >>b=pd.Series([1.25,3.44,5.55,1.23,8.43,9.76])
    >>ft.count_bins(b)
    Bins Count:  
 [['a', 'b'], 1, ['c', 'd', 'e'], 5, ['f', 'g', 'h'], 0, ['i', 'j', 'k'], 0, ['l', 'm', 'n'], 2, ['o', 'p', 'q'], 0, ['r', 's'], 0, ['t', 'u'], 0, ['v', 'w'], 0, ['x', 'y', 'z'], 0] 
    
Bins Count:  
 (1.2200000000000002, 2.083]    2
(8.907, 9.76]                  1
(8.054, 8.907]                 1
(5.495, 6.348]                 1
(2.936, 3.789]                 1
(7.201, 8.054]                 0
(6.348, 7.201]                 0
(4.642, 5.495]                 0
(3.789, 4.642]                 0
(2.083, 2.936]                 0
dtype: int64 
    """
    if colData.dtypes!=str and colData.dtypes!=object:
        return pd.Series.value_counts(colData, bins=10, **kwargs)
    else:
        result={"a":0,"c":0,"e":0,"g":0,"j":0,"m":0,"p":0,"s":0,"u":0,"x":0}
        a = list(builtins.range(ord("a"),ord("a")+2))
        bin1 = [chr(s1) for s1 in a]
        c = list(builtins.range(ord("c"),ord("c")+2))
        bin2 = [chr(s1) for s1 in c]
        e = list(builtins.range(ord("e"),ord("e")+2))
        bin3 = [chr(s1) for s1 in e]
        g = list(builtins.range(ord("g"),ord("g")+3))
        bin4 = [chr(s1) for s1 in g]
        j = list(builtins.range(ord("j"),ord("j")+3))
        bin5 = [chr(s1) for s1 in j]
        m = list(builtins.range(ord("m"),ord("m")+3))
        bin6 = [chr(s1) for s1 in m]
        p = list(builtins.range(ord("p"),ord("p")+3))
        bin7 = [chr(s1) for s1 in p]
        s = list(builtins.range(ord("s"),ord("s")+2))
        bin8 = [chr(s1) for s1 in s]
        u = list(builtins.range(ord("u"),ord("u")+3))
        bin9 = [chr(s1) for s1 in u]
        x = list(builtins.range(ord("x"),ord("x")+3))
        bin10 = [chr(s1) for s1 in x]

        for st in colData:
            st=st.lower()
            if st[0] in bin1:
                result["a"]+=1
            elif st[0] in bin2:
                result["c"]+=1
            elif st[0] in bin3:
                result["e"]+=1
            elif st[0] in bin4:
                result["g"]+=1
            elif st[0] in bin5:
                result["j"]+=1
            elif st[0] in bin6:
                result["m"]+=1
            elif st[0] in bin7:
                result["p"]+=1
            elif st[0] in bin8:
                result["s"]+=1
            elif st[0] in bin9:
                result["u"]+=1
            elif st[0] in bin10:
                result["x"]+=1
        return result

    
        
'''       #colData.sort()
        for st in colData:
            st.lower()
        bin1=["a","b"]
        bin2=["c","d","e"]
        bin3=["f","g","h"]
        bin4=["i","j","k"]
        bin5=["l","m","n"]
        bin6=["o","p","q"]
        bin7=["r","s"]
        bin8=["t","u"]
        bin9=["v","w"]
        bin10=["x","y","z"]
        bins=[bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10]
        count1,count2,count3,count4,count5,count6,count7,count8,count9,count10=[],[],[],[],[],[],[],[],[],[]
        counts=[count1,count2,count3,count4,count5,count6,count7,count8,count9,count10]
        for st in colData:
            if st[0] in bin1:
                count1.append(str)
            elif st[0] in bin2:
                count2.append(str)
            elif st[0] in bin3:
                count3.append(str)
            elif st[0] in bin4:
                count4.append(str)
            elif st[0] in bin5:
                count5.append(str)
            elif st[0] in bin6:
                count6.append(str)
            elif st[0] in bin7:
                count7.append(str)
            elif st[0] in bin8:
                count8.append(str)
            elif st[0] in bin9:
                count9.append(str)
            elif st[0] in bin10:
                count10.append(str)
        result=[]
        for i in builtins.range(10):
            result.append(bins[i])
            result.append(len(counts[i]))
        return pd.Series(result)
        #for i in range(10):
            #print(bins[i],len(counts[i]))'''


       

        
#sum = pd.Series.sum
def sum(colData):
    """sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs)
    Return the sum of the values for the requested axis, for input series with numerical data only.
    
                This is equivalent to the method ``numpy.sum``.
    
    Parameters
    ----------
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
    min_count : int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.
    
        .. versionadded:: 0.22.0
    
           Added with the default being 0. This means the sum of an all-NA
           or empty Series is 0, and the product of an all-NA or empty
           Series is 1.
    **kwargs
        Additional keyword arguments to be passed to the function.
    
    Returns
    -------
    scalar or Series (if level specified)
    
   
    Examples
    --------
    >>> idx = pd.MultiIndex.from_arrays([
    ...     ['warm', 'warm', 'cold', 'cold'],
    ...     ['dog', 'falcon', 'fish', 'spider']],
    ...     names=['blooded', 'animal'])
    >>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)
    >>> s
    blooded  animal
    warm     dog       4
             falcon    2
    cold     fish      0
             spider    8
    Name: legs, dtype: int64
    
    >>> s.sum()
    14
    
    Sum using level names, as well as indices.
    
    >>> s.sum(level='blooded')
    blooded
    warm    6
    cold    8
    Name: legs, dtype: int64
    
    >>> s.sum(level=0)
    blooded
    warm    6
    cold    8
    Name: legs, dtype: int64
    
    By default, the sum of an empty or all-NA Series is ``0``.
    
    >>> pd.Series([]).sum()  # min_count=0 is the default
    0.0
    
    This can be controlled with the ``min_count`` parameter. For example, if
    you'd like the sum of an empty series to be NaN, pass ``min_count=1``.
    
    >>> pd.Series([]).sum(min_count=1)
    nan
    
    Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
    empty series identically.
    
    >>> pd.Series([np.nan]).sum()
    0.0
    
    >>> pd.Series([np.nan]).sum(min_count=1)
    nan
"""
    if colData.dtypes in [int, float]:
        return pd.Series.sum(colData)


#Frequency
def count_r(colData,**kwargs):
    """
    Frequency of values in series object
    Returns a Series containing count of unique values as a percentage of the total number of values.
    
    The resulting object will be in descending order so that the
    first element is the most frequently-occurring element.
    Excludes NA values by default.
    
    Parameters
    ----------
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on

    normalize : bool, default False
        If True then the object returned will contain the relative
        frequencies of the unique values.
    sort : bool, default True
        Sort by frequencies.
    ascending : bool, default False
        Sort in ascending order.
    bins : int, optional
        Rather than count values, group them into half-open bins,
        a convenience for ``pd.cut``, only works with numeric data.
    dropna : bool, default True
        Don't include counts of NaN.
    
    Returns
    -------
    Series of percentage values for each unique value in series.
     For each value, frequency = count of value/total number of values in series *100%
    
    Examples
    --------
    >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
    >>> index.value_counts()
    3.0    20%
    4.0    10%
    2.0    10%
    1.0    10%
    dtype: int64
    
    """
    return pd.Series.value_counts(colData,**kwargs)/size(colData)*100

#Mode
mode=pd.Series.mode


#Range
def range(colData):
    """
    Range of Data
    Returns the difference between the highest value and lowest value in the input series object.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    range : float
        range = Maximum value - Minimum value
   
   Example:
   >>>a=pd.Series([1,2,3,4,5,6])
   >>>ft.range(a)
   5
    
"""
    return max(colData)-min(colData)

#No. of Null/Missing values
def null_c(colData):
    """
    Number of Missing Values
    Returns the number of missing values in input series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    count : int
        Total number of missing values (NA, nan, None) in the series
    
    Example:
    >>>a=pd.Series([1,2,3,None,4])
    >>>ft.null_c(a)
    1
    
    """
    return pd.Series.isnull(colData).sum()


#Ratio of Null/Missing Values
def null_r(colData):
    """
    Ratio of Missing Values
    Returns the ratio of the number of missing values to the total number of values in input series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    ratio : float
        ratio = number of missing values/total number of values in input series
    
    Example:
    >>>a=pd.Series([1,2,3,None,4])
    >>>ft.null_r(a)
    0.2"""
    
    return null_c(colData)/size(colData)


#Quantiles
quantile = pd.Series.quantile
#change so that categrical can work on it too

#Inter Quartile Range
def iqr(colData):
    """
    Interquartile Range
    Returns the inter quartile range (difference between the 75th and 25th percentiles) of the values in the input series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Interquartile Range: float
        interquarile range = Q3 - Q1
    Example:
    >>>a=pd.Series([1,2,3,4,5,4,3,2,2,1])
    >>>ft.iqr(a)
    1.75
     
    """
    return quantile(colData,0.75)-quantile(colData,0.25)


#Kurtosis
kurtosis = pd.Series.kurtosis

#Skewness
skewness = pd.Series.skew

#Mean Absolute Deviation
mean_abs_dev = pd.Series.mad 

#Memory Size
memory = pd.Series.memory_usage

#Outlier
def outlier(colData):
    """Outliers
    Returns outlier values from input series.
    Outliers are determined using the quartiles of the data. Upper and lower bounds are calculated which are 1.5*IQR higher and lower than the 3rd and 1st quartiles respectively. Data values higher than the upper bound or lower than the lower bound are considered outliers.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    list of outliers : list
    
    Example:
    >>>c= pd.Series([20.5,23.6,30.8,27.2,28.3,22.1,25.7,0,2.1,27.3,N100.2])
    >>>ft.outliers(c)
    [2.1, 100.2]
    """
    lowerbound = quantile(colData,0.25) -(1.5* iqr(colData))
    upperbound = quantile(colData, 0.75)+(1.5*iqr(colData))
    result=[]
    for x in colData:
        if (x<lowerbound) or (x>upperbound):
            result.append(x)
        else:
            pass
    return result
#insert docs


#Number of Outliers
def outlier_c(colData):
    """
    Number of Outliers
    Returns the number of outliers in input series.  
    Outliers are determined using the quartiles of the data. Upper and lower bounds are calculated which are 1.5*IQR higher and lower than the 3rd and 1st quartiles respectively. Data     values higher than the upper bound or lower than the lower bound are considered outliers.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Number of Outliers : int
   
    Example:
    >>c = pd.Series([20.5,23.6,30.8,27.2,28.3,22.1,25.7,0,2.1,27.3,None,100.2]) 
    >>ft.outlier_c(c)
    3
    
    """
    return len(outlier(colData))


#Zero Count
def zero_c(colData):
    """Zero Count
    Returns number of Zero values in the input series of data.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Count of Zeros: int
    
    Example:
    >>c = pd.Series([20.5,23.6,30.8,27.2,28.3,22.1,25.7,0,2.1,27.3,None,100.2]) 
    >>ft.zero_c(c)
    1
    """
    return size(colData)-np.count_nonzero(colData)


# Coefficient Of Variance
def coef_of_var(colData):
    """Coefficient of Variance
    Returns the coefficient of variance of the data in input series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Coefficient of Variance: float
        coefficient of variance = standard deviation/mean
    
    Example:
    >>b=pd.Series([1.25,3.44,5.55,1.23,8.43,9.76]) 
    >>ft.coef_of_var(b)
    0.7316997205653936
    """
    return std_dev(colData)/mean(colData)


def missing_int_c(colData): 
    """Missing Integer Count
    Returns number of missing integers in the input series of integer data within range of the maximum and minimum value in the series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Count of Missing Values: int
    
    Example:
    >>c = pd.Series([2,5,8,1]) 
    >>ft.missing_int_c(c)
    4
    """
    if colData.dtype == int:
        num_records = len(uniq_values(colData))
        result = max(colData)-min(colData)-num_records+1
        return result
        

        

def missing_int_r(colData):
    """Missing Integer Ratio
    Returns ratio of missing integers in the input series of integer data within range of the maximum and minimum value in the series.
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Ratio of Missing Values: float
    
    Example:
    >>c = pd.Series([2,5,8,1]) 
    >>ft.missing_int_r(c)
    0.571429
    """
    if colData.dtype==int:
        return missing_int_c(colData)/(range(colData))
   


from visions.functional import infer_series_type  
from visions.functional import detect_series_type
from visions.typesets.complete_set import CompleteSet
def att_class(colData):
    """Attribute Class
    Returns the class of data in the input series based on the Visions package
    
    Args:
    colData (array_like, 1D):Pandas Series of Data or Dataframe Column for function to be applied on
    
    Returns:
    Attribute Class: class 'visions.types.type.VisionsBaseTypeMeta'
    
    Example:
    >>c = pd.Series([2,5,8,1]) 
    >>ft.att_class(c)
    Integer
    
    """
    typeset = CompleteSet()
    return detect_series_type(colData, typeset)