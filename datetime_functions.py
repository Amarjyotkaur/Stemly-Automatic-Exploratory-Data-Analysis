import pandas as pd
import datetime
import numpy as np
import features as ft
from pandas.tseries.offsets import *
from datetime import timedelta, date
from visions.functional import infer_series_type
from visions.typesets import StandardSet
'''Date Time Functions

Meta Attributes:
1. time_from _min
2. extract_unit
3. interval_ls

Loader Function:
4. load_date_attributes
5. fix_date

'''
#Func to convert date time to time count from minimum, outputs each data point as a time count (in input unit) from the minimum datetime.
def time_from_min(colData, units):
    """Time From Minimum
    Returns series of converted DateTimes into number of seconds/days/business days from beginning DateTime in input series (depending on input unit)
    Args:
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of DateTime data for function to be applied on
    
    Returns:
     Series of converted DateTimes into integer number of seconds/days/business days from beginning DateTime in input series (depending on input unit)
    
    Example: 
    >>data = pd.Series([1965-01-02,1965-01-04,1965-01-05,1965-01-08,1965-01-09,1965-01-10,1965-01-12])
    >>ft.time_from_min(data, 'days')
    0      0
    1      2
    2      3
    3      6
    4      7
    5      8
    6     10
 
     
    """
    ls=[]
    #colData=pd.to_datetime(colData)
    #colData = pd.to_datetime(colData.apply(lambda x: x.date()))
    min = ft.min(colData)
    max = ft.max(colData)
    for i in colData:
        interval = i - min
        seconds = interval.total_seconds()
        bdays = len(pd.bdate_range(min, i))
        #minutes = divmod(seconds,60)[0]
        #hours = divmod(seconds,3600)[0]
        days = int(divmod(seconds, 86400)[0])
        if units =="seconds":
            ls.append(seconds)
        elif units =="days":
            ls.append(days)
        elif units == "bdays":
            ls.append(bdays)
        else:
            pass
    return pd.Series(ls)



def extract_unit(colData, unit):
    """Extract Unit
    Returns series of extracted input unit (dayofm, dayofw, dayofy,hour, minute, second, month or year) from input series of DateTime data.
    Args:
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of DateTime data for function to be applied on
    
    Returns:
     Series of extracted unit values in integer
     
    Example: 
    >>data = pd.Series([1965-01-02,1965-01-04,1965-01-05,1965-01-08,1965-01-09,1965-01-10,1965-01-12])
    >>ft.extract_unit(data, 'dayofy')
    0      2
    1      4
    2      5
    3      8
    4      9
    5     10
    6     12

     
    """
    if unit == "dayofm": # day of the month
        return colData.dt.day
    elif unit =="dayofw":
        return colData.dt.weekday
    elif unit =="dayofy":
        return colData.dt.dayofyear
    elif unit =="hour":
        return colData.dt.hour
    elif unit == "minute":
        return colData.dt.minute
    elif unit == "second":
        return colData.dt.second
    elif unit =="month":
        return colData.dt.month
    elif unit == "year":
        return colData.dt.year
    else:
        return "incorrect unit"



def interval_ls(colData):  
    """Interval List
    Returns series of interval values between DateTime records in input series, in units of days.
    Args:
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of DateTime data for function to be applied on
    
    Returns:
     Series of integer interval values between DateTime records in input series, in units of days.
     
    Example: 
    >>data = pd.Series([1965-01-02,1965-01-04,1965-01-05,1965-01-08,1965-01-09,1965-01-10,1965-01-12])
    >>ft.interval_ls(colData)
    0    2 days
    1    1 days
    2    3 days
    3    1 days
    4    1 days
    5    2 days
    6    3 days

     
    """
    ls=[]
    for i in range(len(colData)-1):
        lag = colData[i+1]-colData[i]
        ls.append(lag)
    return pd.Series(ls)





def load_date_attributes(dataframe):
    """Date Time Meta Attributes Loader
    Returns list of dataframes containing meta attributes of DateTime columns from input dataframe
    Args:
     dataframe : Pandas Dataframe that includes DateTime column(s) for function to be applied on
    
    Returns:
     ls : list of dataframes for each DateTime column in the input dataset.
     
    Example: 
    *please refer to Date Time Demo Notebook for example of load_date_attributes output list of dataframes*

     
    """
    typeset = StandardSet()
    datetime_cols = [x for x in dataframe.columns if str(infer_series_type(dataframe[x], typeset)) == 'DateTime']
    ls=[]
    print(datetime_cols)
    for col in datetime_cols:
        dataframe[col] = pd.to_datetime(dataframe[col])
        dataframe[col] = pd.to_datetime(dataframe[col].apply(lambda x: x.date()))
        
        for i in range(len(dataframe[col])):
            dataframe[col][i] = fix_date(  dataframe[col][i])
            
        data = {col:dataframe[col],"Year":extract_unit(dataframe[col],"year"),"Month":extract_unit(dataframe[col],"month"),"Day of Month":extract_unit(dataframe[col],"dayofm"),"Day of Week":extract_unit(dataframe[col],"dayofw"),"Day of Year":extract_unit(dataframe[col],"dayofy"),"Hour":extract_unit(dataframe[col],"hour"),"Minute": extract_unit(dataframe[col],"minute"), "Second":extract_unit(dataframe[col],"second"),"Day Count"+str(col):time_from_min(dataframe[col],"days"), "Business Day Count":time_from_min(dataframe[col],"bdays"), "Interval Lag":interval_ls(dataframe[col])}

        
       
        new_df = pd.DataFrame(data)
        ls.append(new_df)
    return ls


def fix_date(x):
    """Fix Date - Helper function for loader function
    Returns DateTime value with corrected year for input DateTime records that are later than 2040.
    Args:
     x : datetime value
    
    Returns:
     Corrected datetime64 value
     
    Example: 
    >> ft.fix_date(datetime.datetime(2050, 1, 2, 0, 0)))
    1950-01-02 00:00:00
     
    """
    
    if x.year > 2040:
        year = x.year - 100
    elif x.year<1920:
        year = x.year +100
    else:
        year = x.year
    return datetime.datetime(year,x.month,x.day, x.hour, x.minute, x.second)