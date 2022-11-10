# %% [markdown]
# # Project 1: Processing health and activity data [40 marks]
# 
# ---
# 
# Make sure you read the instructions in `README.md` before starting! In particular, make sure your code is well-commented, with sensible structure, and easy to read throughout your notebook.
# 
# ---
# 
# The MMASH dataset [1, 2] is is a dataset of health- and activity-related measurements taken on 22 different people, over a continuous period of 24 hours, using wearable devices.
# 
# In this project, we have provided you with some of this data for **10** of those individuals. In the `dataset` folder, you will find:
# 
# - a file `subject_info.txt` which summarises the age (in years), height (in cm), and weight (in kg) of all 10 study participants,
# - 10 folders named `subject_X`, which each contain two files:
#     - `heartbeats.txt` contains data on all individual heartbeats detected over the 24-hour study period,
#     - `actigraph.txt` contains heart rate and other activity data measured with another device, over the same 24-hour period.
# 
# The tasks below will guide you through using your Python skills to process some of this data. Note that the data was reformatted slightly for the purpose of the assignment (to make your life a bit easier!), but the values are all the original ones from the real dataset.
# 
# ### Getting stuck
# 
# Tasks 3 to 8 follow directly from each other. There is a `testing` folder provided for you with `.npy` files and a supplementary `actigraph.txt` dataset. The `.npy` files are NumPy arrays, which you can load directly using `np.load()`, containing an example of what the data should look like after each task. You will be able to use this example data to keep working on the later tasks, even if you get stuck on an earlier task. Look out for the 💾 instructions under each task.
# 
# These were produced using the data for another person which is not part of the 10 you have in your dataset.
# 
# 
# ### References
# 
# [1] Rossi, A., Da Pozzo, E., Menicagli, D., Tremolanti, C., Priami, C., Sirbu, A., Clifton, D., Martini, C., & Morelli, D. (2020). Multilevel Monitoring of Activity and Sleep in Healthy People (version 1.0.0). PhysioNet. https://doi.org/10.13026/cerq-fc86
# 
# [2] Rossi, A., Da Pozzo, E., Menicagli, D., Tremolanti, C., Priami, C., Sirbu, A., Clifton, D., Martini, C., & Morelli, D. (2020). A Public Dataset of 24-h Multi-Levels Psycho-Physiological Responses in Young Healthy Adults. Data, 5(4), 91. https://doi.org/10.3390/data5040091.
# 
# ---
# ## Task 1: Reading the subject information
# 
# The file `subject_info.txt` in your `dataset` folder summarises the age (in years), height (in cm), and weight (in kg) of all 10 study participants.
# 
# ---
# 🚩 ***Task 1:*** Write a function `read_subject_info()` which reads in the information in `subject_info.txt`, and returns two outputs:
# 
# - a list `headers` containing the four column headers as strings, read from the first line in the file;
# - a NumPy array `info` containing the numerical information for each person (i.e. it should have 10 rows and 4 columns).
# 
# **Important:** the height of each subject should be given in **metres** in your `info` array.
# 
# **[3 marks]**

# %%
import numpy as np
def read_subject_info():
  '''
  Read subject_info.txt, stored title and content 
  output: headers(list); info(array)
  '''

  with open('dataset/subject_info.txt') as f:
    #read the file, skip title 
    #set the type to floating point
    info = np.loadtxt('dataset/subject_info.txt',skiprows =1 ,delimiter=',',dtype=float)
    #Read the headings, separated by commas
    headers = f.readline().strip('\n').split(',')
    #Divide your height by 100 to convert it into meters 
    for i in range(len(info)):
          info[i][2] = round(info[i][2]/100,2) #keep two decimal places
  return headers,info


# %% [markdown]
# ---
# ## Task 2: Charting the Body Mass Index (BMI) for all participants
# 
# The Body Mass Index (BMI) can be used to indicate whether someone is at a healthy body weight. [The NHS website](https://www.nhs.uk/common-health-questions/lifestyle/what-is-the-body-mass-index-bmi/) describes it as follows:
# 
# > The body mass index (BMI) is a measure that uses your height and weight to work out if your weight is healthy.
# >
# > The BMI calculation divides an adult's weight in kilograms by their height in metres, squared. For example, a BMI of $25$ means $25 \text{kg/m}^2$.
# >
# > For most adults, an ideal BMI is in the $18.5$ to $24.9$ range.
# 
# This means that the BMI is calculated as follows:
# 
# $$
# \text{BMI} = \frac{\text{weight}}{\text{height}^2}.
# $$
# 
# ---
# 🚩 ***Task 2:*** Write a function `bmi_chart(info)` which takes as input the `info` array returned by `read_subject_info()`, produces a visualisation showing all subjects' heights and weights on a graph, and clearly indicates whether they are within the "healthy weight" range as described above (i.e. their BMI is in the $18.5$ to $24.9$ range).
# 
# Your function should not return anything, but calling it with `bmi_chart(info)` must be sufficient to display the visualisation.
# 
# You should choose carefully how to lay out your plot so that it is easy to interpret and understand.
# 
# **[4 marks]**

# %%
import matplotlib.pyplot as plt
import numpy as np
headers,info = read_subject_info()
def bmi_chart(info):
    '''
    The value of bmi was calculated and the image was drawn. 
    The points in different colors represented whether the BMI was up to the standard
    input: info(array)
    '''
    # Store the value of BMI
    bmi = []
    #store the position (18.5<BMI<24.9)
    position = []
    # According to the formula, get the BMI and keep two decimal places
    for i in range(0,10):
        bmi.append(round(info[i][1]/(info[i][2]*info[i][2]),1))
    # The height, weight, and BMI data were made into charts
    # Find the points that exceed the standard，record the location
    for i in range(10):
        if bmi[i] >24.9 or bmi[i]<18.5: 
            position.append(i)
    weight = info[:,1]
    height = info[:,2]
    # Define image size 10*5
    # Draw the image with green for health and red for unhealthy
    plt.figure(figsize=(10,5),dpi=200)
    plt.scatter(height,weight,label = ' health',color = 'g')
    plt.scatter(height[position],weight[position],label = ' unhealth',color = 'r')
    # x axis coordinate, y axis coordinate, image title
    # Add label on the upper left side of the table
    plt.xlabel('Height(m)',fontsize = 15) 
    plt.ylabel('Weight(kg)',fontsize = 15) 
    plt.title('BMI scatter plot')
    plt.legend(loc='upper left')
    plt.show()
info = read_subject_info()[1]
bmi_chart(info)
    
    

# %% [markdown]
# ---
# ## Task 3: Instantaneous heart rate data
# 
# For each subject, the file `heartbeats.txt` contains data on all individual heartbeats detected over the 24-hour study period. Specifically, the two columns record the time at which each heartbeat was detected, and the interval (in seconds) between the current heartbeat and the previous one.
# 
# ### Handling timestamp data
# 
# For the next tasks, you will use NumPy's `datetime64[s]` and `timedelta64[s]` object types, respectively used to represent times (as if read on a clock) and time intervals. You should [consult the relevant documentation](https://numpy.org/doc/stable/reference/arrays.datetime.html#datetimes-and-timedeltas).
# 
# Here are a few illustrative examples:

# %%
import numpy as np
import matplotlib.pyplot as plt

# Create two datetime objects and a vector of dates
start_time = np.datetime64('2022-10-20 12:00:00')
end_time = np.datetime64('2022-11-10 12:00:00')
time_vector = np.array(['2022-10-20', '2022-10-23', '2022-10-28'], dtype='datetime64[s]')
print(time_vector)

# Get time interval between the two times
time_elapsed = end_time - start_time
print(time_elapsed)
print(type(time_elapsed))

# Divide by the duration of 1 second to get number of seconds (as a number object)
seconds_elapsed = time_elapsed / np.timedelta64(1, 's')
print(seconds_elapsed)
print(type(time_elapsed))

# Divide by the duration of 1 day to get number of days
days_elapsed = time_elapsed / np.timedelta64(1, 'D')
print(days_elapsed)

# Create a range of datetimes spaced by 1 day
step = np.timedelta64(1, 'D')
days = np.arange(start_time, end_time + step, step)

# Plot something using days as the x-axis
fig, ax = plt.subplots(figsize=(12, 4))
value = np.random.randint(1, 11, size=len(days))
ax.plot(days, value, 'ro-')
ax.set(ylim=[0, 11], xlabel='Date', ylabel='Value')
plt.show()

# %% [markdown]
# ---
# 🚩 ***Task 3a:*** Write a function `read_heartbeat_data(subject)` which takes as input an integer `subject` between 1 and 10, reads the data in `heartbeats.txt` for the given `subject`, and returns it as two NumPy vectors:
# 
# - `times`, containing the recorded times of each heartbeat (as `datetime64[s]` objects),
# - `intervals`, containing the recorded intervals between heartbeats (in seconds, as `float` numbers).
# 
# **[3 marks]**

# %%
import numpy as np
def read_heartbeat_data(subject):
    '''
    read heartbeat.txt and store time,intervals
    input:subject(int)
    output:intervals(arrat-float);times(array-datetime64[s])
    '''
    # Read the contents of the file, skipping the first line header
    file = np.loadtxt(f'dataset/subject_{subject}/heartbeats.txt',dtype=str,delimiter=',',skiprows=1)
    # Store interval information of type floating point
    intervals = np.array(file[:,2],dtype='float')
    # Store time information of type datetime64
    times = np.array(file[:,1], dtype='datetime64[s]')
    return intervals,times

# %% [markdown]
# ---
# 🚩 ***Task 3b:*** Write a function `hr_from_intervals(intervals)` which takes as input a NumPy vector containing heartbeat interval data (such as that returned by `read_heartbeat_data()`), and returns a NumPy vector of the same length, containing the instantaneous heart rates, in **beats per minute (BPM)**, calculated from the intervals between heartbeats. You should return the heart rates as floating-point numbers.
# 
# For instance, an interval of 1 second between heartbeats should correspond to a heart rate of 60 BPM.
# 
# **[2 marks]**

# %%
def hr_from_intervals(intervals):
    '''
    Calculate heart rate per second
    input:intervals(array-float)
    output: hr_row(array-float)
    '''
    # Based on the interval, 60 per interval
    hr_raw = 60/intervals # 60/intervals = per second
    return hr_raw
    

# %% [markdown]
# ---
# ## Task 4: Data cleaning
# 
# There are gaps and measurement errors in the heartbeat data provided by the device. These errors will likely appear as outliers in the data, which we will now try to remove.
# 
# One possible method is to remove data points which correspond to values above and below certain **percentiles** of the data. Removing the data below the $p$th percentile means removing the $p\%$ lowest values in the dataset. (Note that, for instance, the 50th percentile is the median.)
# 
# ---
# 🚩 ***Task 4a:*** Write a function `clean_data(times_raw, hr_raw, prc_low, prc_high)` which takes 4 inputs:
# 
# - `times_raw` is the NumPy array of timestamps returned by `read_heartbeat_data()`,
# - `hr_raw` is the NumPy array of computed heart rate values returned by `hr_from_intervals()`,
# - `prc_low` and `prc_high` are two numbers such that $0\leq$ `prc_low` $<$ `prc_high` $\leq 100$.
# 
# Your function should return two NumPy arrays of the same length, `times` and `hr`, which are the original arrays `times_raw` and `hr_raw` where all the measurements (heart rate and associated time stamp) below the `prc_low`th percentile and above the `prc_high`th percentile of the heart rate data have been removed.
# 
# You may wish to make use of NumPy functionality to calculate percentiles.
# 
# **[4 marks]**

# %%
def clean_data(times_raw, hr_raw, prc_low, prc_high):
    '''
    Deleting extreme Values
    input:time_row(array-datetime64[s]);hr_row(array-float); prc_low(int), prc_high(int)
    output:times(array-datetime64[s]);hr(array-float)
    '''
    # Calculate the outliers to be eliminated
    low = np.percentile(sorted(hr_raw),prc_low) 
    high = np.percentile(sorted(hr_raw),prc_high)
    times = [] 
    hr = [] 
    # If the data is larger than the small outlier range and smaller than the high outlier range, retain
    for i in range(len(hr_raw)): # Go through the hr_row
        if (hr_raw[i] >=low) and (hr_raw[i]<= high): # satisfied the condition: low outliers<=hr_row<=high outliers
            # Store the content that meets the criteria
            hr.append(hr_raw[i])
            times.append(times_raw[i])
    times = np.array(times)
    return times,hr

# %% [markdown]
# ---
# 🚩 ***Task 4b:*** Write a function `evaluate_cleaning(subject)`, which takes as input an integer `subject` between 1 and 10 indicating the subject number, and plots the following two histograms for that subject:
# 
# - a histogram of the raw heart rate data,
# - a histogram of the heart rate data after cleaning with `clean_data()`, where the bottom 1% and the top 1% of the values have been removed.
# 
# Your histograms should use a logarithmic scale on the y-axis, and be clearly labelled. You should consider carefully how to lay out the histogram to best present the information.
# 
# Your function `evaluate_cleaning()` should call the functions `read_heartbeat_data()`, `hr_from_intervals()`, and `clean_data()` you wrote above, in order to obtain the raw and cleaned heart rate data for a given `subject`.
# 
# Then, use your function to display the histograms of the raw and cleaned data for Subject 3. Given that heart rates in adults can typically range from about 40 to 160 beats per minute, and given your histograms, explain why this is a suitable method to remove likely measurement errors in the heart rate data.
# 
# **[3 marks]**
# 
# ---
# 
# 💾 *If you are stuck on Task 3 or on the task above, you can load the data provided in the `testing` folder to produce your histograms, by running the following commands:*
# 
# ```python
# times_raw = np.load('testing/times_raw.npy')
# hr_raw = np.load('testing/hr_raw.npy')
# times = np.load('testing/times.npy')
# hr = np.load('testing/hr.npy')
# ```

# %%
def evaluate_cleaning(subject):
  '''
  Draw the image after removing the extreme, and compare
  input:subject(int)
  '''
  # Call the function and get the hr_raw,hr
  intervals,times_raw= read_heartbeat_data(subject)
  hr_raw = hr_from_intervals(intervals)
  hr = clean_data(times_raw,hr_raw,1,99)[1]
  # Calculating the Number of bins:(start-end)/2
  num = int((hr[-1]-hr[1])/6)
  # Make a picture with the y axis as log
  
  plt.hist(hr_raw,bins=num,rwidth=0.9)
  plt.yscale('log')
  plt.title('hr_row')
  plt.xlabel('heart_rate')
  plt.ylabel('log of frequency')
  plt.show()
  
  plt.hist(hr,bins=num,rwidth=0.8)
  plt.yscale('log')
  plt.title('hr')
  plt.ylabel('log of frequency')
  plt.xlabel('cleaned heart_rate')
  plt.show()

evaluate_cleaning(3)
    

# %% [markdown]
# *Use this Markdown cell to write your explanation for Task 4.*
# After cleaning up the data, more concentrated between 40 and 160

# %% [markdown]
# ---
# ## Task 5: Interpolating the data
# 
# Although the device detecting heartbeats was able to measure intervals between beats with millisecond precision, the recorded timestamps could only record the second at which a heartbeat occurred. This means that there are not only time gaps in the data (due to the device missing heartbeats), but also several heartbeats usually recorded in the same second.
# 
# For example, this is an excerpt from Subject 7's data, showing a 9-second time gap between `09:19:57` and `09:20:06`, as well as 3 different heartbeats detected at `09:20:06`:
# 
# ```
# 59,2022-07-21 09:19:56,1.033
# 60,2022-07-21 09:19:57,0.942
# 61,2022-07-21 09:20:06,0.307
# 62,2022-07-21 09:20:06,0.439
# 63,2022-07-21 09:20:06,0.297
# 64,2022-07-21 09:20:07,0.427
# ```
# 
# The goal of this next task is to **interpolate** the recorded data, in order to produce a new dataset containing values of the heart rate at regular time intervals. We will use **linear interpolation**, with the help of SciPy's `interp1d()` function (from the `interpolate` module) which we saw in Week 5.
# 
# ---
# 🚩 ***Task 5a:*** The `interp1d()` function from SciPy can only be used with numeric data, and not timestamps. Two functions are provided for you below.
# 
# - Explain, in your own words, what both functions do and how.
# - Write a few lines of test code which clearly demonstrate how the functions work.
# 
# **[2 marks]**

# %%
def datetime_to_seconds(times):
    return (times - times[0]) / np.timedelta64(1, 's')
    
def seconds_to_datetime(seconds_elapsed, start_time):
    return seconds_elapsed * np.timedelta64(1, 's') + start_time



# Demonstrating usage
intervals,times_raw = read_heartbeat_data(1)
hr_raw = hr_from_intervals(intervals)
times,hr = clean_data(times_raw, hr_raw, 1, 99)
print(datetime_to_seconds(times))
print(seconds_to_datetime(seconds_elapsed, start_time))


# %% [markdown]
# *Use this Markdown cell to explain how the functions `datetime_to_seconds()` and `seconds_to_datetime()` work.*
# 1. Convert the timestamp format to a calculable number of seconds
# 2. Convert the number of seconds format to timestamp format

# %% [markdown]
# ---
# 🚩 ***Task 5b:*** Write a function `generate_interpolated_hr(times, hr, time_delta)` which takes as inputs:
# 
# - two NumPy vectors `times` and `hr` such as those returned by `clean_data()`,
# - a `timedelta64[s]` object representing a time interval in seconds,
# 
# and returns two new NumPy vectors, `times_interp` and `hr_interp`, such that:
# 
# - `times_interp` contains regularly spaced `datetime64[s]` timestamps, starting at `times[0]`, ending on or less than `time_delta` seconds before `times[-1]`, and with an interval of `time_delta` between consecutive times.
# - `hr_interp` contains the heart rate data obtained using **linear interpolation** and evaluated at each time in `times_interp`, for example with the help of the `interp1d()` function from `scipy.interpolate`.
# 
# For example, if `times` starts at `10:20:00` and ends at `10:20:09` with a `time_delta` of two seconds, then your `times_interp` vector should contain `10:20:00`, `10:20:02`, `10:20:04`, `10:20:06`, `10:20:08`, and `hr_interp` should consist of the corresponding interpolated heart rate values at each of those times.
# 
# **[4 marks]**

# %%
# URL: http://zhuanlan.zhihu.com/p/557155437
# Lines X-Y: dawufeixingqi
# Accessed on 7 Oct 2022.
from scipy.interpolate import interp1d
def generate_interpolated_hr(times, hr, time_delta):
     '''
     Returns the timestamp and heart rate for the specified interval
     input:times(array-datetime64[s]);hr(array-float);time_delta(timedelta64[s])
     output:times_interp(array-datetime64[s]);hr_interp(array-float)
     '''
     # Convert to digital format
     times_float = datetime_to_seconds(times)
     timedelta_float = time_delta / np.timedelta64(1,'s')
     #Record the start time and end time
     start = times_float[0]
     end = times_float[-1]
     # Build the complete time set
     times_interp = np.arange(start,end,timedelta_float)
     # Call interp1d method, linear fitting, interpolation
     interp = interp1d(times_float, hr)
     hr_interp = interp(times_interp)
     times_interp = np.array(times_interp+np.array(times[0],dtype=float),dtype='datetime64[s]')
     return times_interp,hr_interp

# %% [markdown]
# ---
# 🚩 ***Task 5c:*** Check your `generate_interpolated_hr()` function by generating interpolated heart rate data for Subject 1 for just the first 100 measurements (after cleaning). You should generate interpolated data with a time interval of 5 seconds. Plot the data points, as well as your interpolated data, and discuss in the Markdown cell below whether your plot is what you expected, and why.
# 
# **[2 marks]**
# 
# ---
# 💾 *If you are stuck on Task 4, you can use the cleaned data provided in the `testing` folder to check your code, by running the following commands:*
# 
# ```python
# times = np.load('testing/times.npy')
# hr = np.load('testing/hr.npy')
# ```

# %%
# Call the function to get the times_interp,hr_interp
intervals,times_raw = read_heartbeat_data(1)
hr_raw = hr_from_intervals(intervals)
times,hr = clean_data(times_raw, hr_raw, 1, 99)
# Get the first 100 data
x = times[0:100]
y = hr[0:100]
times_interp,hr_interp = generate_interpolated_hr(x, y, np.timedelta64(5,'s'))
#Draw a plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x,y,'*',label = 'raw')
ax.plot(times_interp,hr_interp, 'k.',label = 'new')
plt.legend(loc='upper right',fontsize = 15)
plt.xlabel('Time')
plt.ylabel('BPM')
plt.show()

# %% [markdown]
# *Use this Markdown cell for discussing your interpolation results.*
# The result of interpolation is ideal and there are more overlapping points
# 

# %% [markdown]
# ---
# ## Task 6: Smoothing the data with a rolling average
# 
# A rolling average is simply an average of the heart rate data, calculated over a given window of time. For example:
# 
# - The 20-second rolling average of the heart rate at a time `10:20:00` is the average heart rate over the 20 seconds leading up to that time, i.e. the average of all the heart rates between `10:19:41` and `10:20:00` (inclusive). If we have measurements of the heart rate every 5 seconds, then this would be the average of the heart rates measured at `10:19:45`, `10:19:50`, `10:19:55`, and `10:20:00`.
# - We can similarly calculate the 20-second rolling average at the next measurement time, `10:20:05`, as the average heart rate over the 20-second period from `10:19:46` to `10:20:05` (inclusive).
# 
# The rolling average essentially smoothes out the sudden jumps in the measured (or interpolated) heart rate data, allowing us to see the longer-term variations more clearly.
# 
# ---
# 🚩 ***Task 6:*** Write a function `rolling_average()` which takes as inputs:
# 
# - two NumPy vectors `times` and `hr` such as those returned by `clean_data()`,
# - a `timedelta64[s]` object `time_delta` representing a time interval in seconds,
# - a `timedelta64[s]` object `window`, representing the window duration in seconds (with `window` assumed to be an integer multiple of `time_delta`),
# 
# and returns a NumPy vector `hr_rolling` containing values for the rolling average of the heart rate over time, with the given window size.
# 
# Your `rolling_average()` function should call `generate_interpolated_hr()` to generate regularly-spaced heart rate data with a time interval `time_delta`, before computing and returning the averaged heart rate data.
# 
# Note that `hr_rolling` will be shorter than the length of your interpolated heart rate data, because you can only start computing rolling averages after one window of time has elapsed. (For instance, if your data starts at `10:20:00`, with a 30-second window, the first value of the rolling average you can obtain is at `10:20:29`.)
# 
# **[4 marks]**

# %%
def rolling_average(times, hr, time_delta, window):
    '''
    Calculate the moving average to smooth the data
    input:times(array-datetime64[s]);hr(array-float);time_delta(array-timedelta64),window(array-timedelta64)
    output:hr_rolling(array-float)
    '''
    # Call the function to get the times_interp,hr_interp
    times_interp,hr_interp = generate_interpolated_hr(times, hr, time_delta)
    hr_rolling= []
    n = int(window/time_delta) # calculate the rolling time
    for i in range(n,len(times_interp)): 
          a = np.sum(hr_interp[i-n+1:i+1])/n # Calculate the average value
          hr_rolling.append(a)
            
    
    return hr_rolling

    

# %% [markdown]
# ---
# ## Task 7: Putting it all together
# 
# You should now have a series of functions which allow you to:
# 
# - read data on measured heartbeart-to-heartbeat intervals for a given subject,
# - transform this data into heart rate measurements and clean out the outliers,
# - interpolate the data to generate measurements at regular time intervals,
# - compute a rolling average of the heart rate data over time, to smooth out the data.
# 
# For each subject, there is another file `actigraph.txt`, containing activity data recorded by a separate device. In particular, this data provides another independent measurement of the subjects' heart rate. We can use this to check our work.
# 
# ---
# 🚩 ***Task 7:*** Write a function `display_heart_rate(subject)` which takes as input an integer `subject` between 1 and 10, and produces one single graph, containing two plots on the same set of axes:
# 
# - a plot of the heart rate data found in `actigraph.txt` over time,
# - a plot of the smoothed heart rate data computed by you from the data in `heartbeats.txt`, using interpolated measurements of the heart rate every 3 seconds, and a 30-second window size for the averaging.
# 
# Your plot should show good agreement between the two sets of data. Instead of showing the full 24 hours of data, you should choose a period of time over which to plot the heart rate (say, approximately 1 hour), in order to better visualise the results.
# 
# Show an example by using your function to display the results for 3 subjects of your choice.
# 
# **[4 marks]**
# 
# ---
# 💾 *If you are stuck on Task 5 or 6, you can use the actigraph heart rate data provided in the `testing` folder in `actigraph.txt`, and compare this to the smoothed heart rate data provided in the `testing` folder, which you can load by running the following command:*
# 
# ```python
# hr_rolling = np.load('testing/hr_rolling.npy')
# ```

# %%
import pandas as pd
def display_heart_rate(subject):
    '''
    Show the heart rate chart for the specified time period
    input:subject(int)
    '''
    
    
    # Create a Chart
    fig, ax = plt.subplots(1,2,figsize = (18,6))
    
    # Read the actigraph data and save it to heart_rate_data
    file = np.loadtxt(f'dataset/subject_{subject}/actigraph.txt',dtype=str,delimiter=',',skiprows=1)
    heart_rate_data = np.array(file[:,2],dtype = 'float')
    time = np.array(file[:,1],dtype='datetime64[s]')
    #point,hr_pointUsed to store time，BPM that meet the conditions
    data = pd.to_datetime(time[0]).date()
    start_time = np.datetime64(data) + np.timedelta64(14,'h')
    end_time = np.datetime64(data) + np.timedelta64(15,'h')
    point1 = []
    hr_point1 = []

    for i in range(0,len(time)):
        if time[i]>=start_time and time[i]<=end_time:#Check whether it is within the specified time
            point1.append(time[i])
            hr_point1.append(heart_rate_data[i])
        elif time[i]> end_time:#over the end time, the loop ends
            break
    
    #obtain smoothed heart rate data
    intervals,times_raw = read_heartbeat_data(subject)
    hr_raw = hr_from_intervals(intervals)
    times,hr = clean_data(times_raw, hr_raw, 1, 99)
    times_interp = generate_interpolated_hr(times, hr, np.timedelta64(3,'s'))[0]
    hr_rolling = rolling_average(times, hr, np.timedelta64(3,'s'), np.timedelta64(30,'s'))
    times_rolling = times_interp[10:] #window/ time_delta
    #point,hr_pointUsed to store time，BPM that meet the conditions
    point2 = []
    hr_point2 = []
    #Check whether the current time range is within the specified time range
    for i in range(0,len(times_rolling)):
        if times_rolling[i]>=start_time and times_rolling[i]<=end_time:#Check whether it is within the specified time
            point2.append(times_rolling[i])
            hr_point2.append(hr_rolling[i])
        elif times_rolling[i] > end_time:#over the end time, the loop ends
            break
    
    
    #picture
    ax[0].plot(point1,hr_point1,'-',label = 'heart rate')
    ax[1].plot(point2,hr_point2,'-',label='smooth heart rate',color = 'r')
    plt.legend(loc='upper left',fontsize = 10)
    plt.xlabel('time')
    plt.ylabel('BPM')
    plt.show()

# subject1
display_heart_rate(2)

#subject2
display_heart_rate(3)

#subject3
display_heart_rate(4)   

# %% [markdown]
# ---
# ## Task 8: relating to other data
# 
# The data in `actigraph.txt` also contains the following columns:
# 
# - `Steps` indicates the number of steps detected per second (using a pedometer).
# - `Inclinometer Standing`/`Sitting`/`Lying` indicates the position of the subject, automatically detected by the device.
# - `Inclinometer Off` indicates when the device didn't record a position.
# 
# In particular, the `Inclinometer ...` columns record either `0` or `1`, and they are mutually exclusive over each row. This means that, for example, a subject can't be recorded simultaneously sitting and standing.
# 
# ---
# 🚩 ***Task 8:*** Using the results of your data processing work in previous tasks, can you relate some of this additional data (and/or some of the data in `subject_info.txt`) to the heart rate estimates that you have obtained?
# 
# You are free to choose how you complete this task. You will be assessed on the correctness of your code and analysis, the quality of your code (readability, commenting/documentation, structure), and the presentation of your results.
# 
# Note that you do not have to use **all** of the extra data to obtain full marks.
# 
# **[5 marks]**
# 
# ---
# 💾 *If you are using `hr_rolling.npy` and the actigraph data in the `testing` folder, this is the information for this person:*
# 
# | Weight | Height | Age |
# |:-:|:-:|:-:|
# | 85 | 180 | 27 |

# %%
import pandas as pd
import numpy as np
def determine_healthy(subject):
    '''
    Calculate the proportion of the day in each state
    '''
    #Reading the file
    file = np.loadtxt(f'dataset/subject_{subject}/actigraph.txt',dtype=str,delimiter=',',skiprows=1)
    # Whether the storage machine works
    inclinometer_Off = np.array(file[:,4],dtype='int')
    # Storage standing posture
    standing = np.array(file[:,5],dtype='int')
    num_standing = 0
    # Store the sitting state
    heart = np.array(file[:,2],dtype='float')
    rate = []

    
    for i in range(0,len(inclinometer_Off)):
        if(standing[i]==1): # In a standing position
            num_standing = num_standing +1
            rate.append(heart[i]) # Record your heart rate while standing
    # Get the ratio in each state
    
    standing_rate = np.sum(rate)/len(rate)  # Calculate the average heart rate while standing
    
    #获取smoothed heart rate data
    intervals,times_raw = read_heartbeat_data(subject)
    hr_raw = hr_from_intervals(intervals)
    times,hr = clean_data(times_raw, hr_raw, 1, 99)
    times_interp = generate_interpolated_hr(times, hr, np.timedelta64(3,'s'))[0]
    hr_rolling = rolling_average(times, hr, np.timedelta64(3,'s'), np.timedelta64(30,'s'))
    ave = np.average(hr_rolling)#Get the average heart rate
    print(standing_rate,ave)
    


# Traverse the data of 10 people
for i in range(1,11):
    determine_healthy(i)
    
bmi =[]
read_subject_info()
for i in range(0,10):
        bmi.append(round(info[i][1]/(info[i][2]*info[i][2]),1))

print(bmi)


# The heart rate while standing is higher than average. 
# And, for the most part, people with a lower BMI had a higher heart rate while standing than people with a lower BMI


