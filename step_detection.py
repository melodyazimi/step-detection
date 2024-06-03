import pandas as pd
import os
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math


#Take in the file and make header names 
#path = working_directory + '/Downloads/acce 3.csv'
dataset = pd.read_csv('acce 3.csv', header = None)
dataset.columns = ["time", "x", "y", "z"]


#split columns into frequencies and timestamp 
timestamp = dataset.iloc[:, 0:1].values
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
z = dataset.iloc[:, 3:4].values

#Change the timestamp data from nanoseconds to seconds
start = timestamp[0][0]
#Place new time data into a new array
time = []
for i in range(0, len(timestamp)):
    temp = timestamp[i] - start 
    time_data = temp[0]
    sec = time_data/(1000000000)
    time.append(sec)
    
#Plot the 3 frequencies against time
plt.figure(figsize=(15, 4))
plt.plot(time, x)
plt.plot(time, y)
plt.plot(time, z)
plt.title("Unfiltered Step Signal Data with Noise")
plt.margins(0, 0.25)
plt.tight_layout()
plt.show()



#Convert signals to magnitude to remove orientation
#Take each value from x-array, y-array, and z-array, add them up to make one data point for magntiude
dataset["x_pow"] = dataset["x"]**2
dataset["y_pow"] = dataset["y"]**2
dataset["z_pow"] = dataset["z"]**2

#Take the square root
dataset["magnitude"] = dataset["x_pow"]+dataset["y_pow"]+dataset["z_pow"]
dataset["magnitude"] = np.sqrt(np.abs(dataset["magnitude"]))

#This will all be added to one array
mag = dataset["magnitude"]


#Plot the new data without direction (just magnitude)
plt.figure(figsize=(15, 4))

plt.plot(time, mag)
plt.title("Magnitude Step Signal Data")
plt.margins(0, .05)

plt.tight_layout()
plt.show()



# Design lowpass filter.
sampling_rate = 100

#Calculate the Nyquist frequency
nyq = 0.5 * sampling_rate

#Set order of the filter
order = 3

#Set cutoff frequency for lowpass filter
cutoff = 3 #Hz

#Design filter
low = cutoff / nyq
b, a = scipy.signal.butter(order, low, btype='low')

#Apply the lowpass filter to the magnitude data
filt_mag = scipy.signal.filtfilt(b, a, mag)



plt.plot(mag, '.-', alpha=.5, label="Noisy ECG data")

for cutoff in [.03, .05, .06, .08, .1]:
    b, a = scipy.signal.butter(order, cutoff)
    filtered = scipy.signal.filtfilt(b, a, mag)
    label = f"{int(cutoff*nyq):d} Hz cutoff"
    plt.plot(filtered, label=label)

plt.legend()
plt.axis([350, 500, None, None])
plt.title("Effect of Different Cutoff Values")
plt.show()




#Plot the filtered step signal data 
plt.figure(figsize=(25, 5))
plt.plot(time, filt_mag)
plt.title("Filtered Step Signal Data")
plt.margins(0, .05)
plt.tight_layout()
plt.show()



#Step counting
#Detect peaks on the filtered magnitude data and add a threshold not all peaks are considered steps 
#Define threshold so not all peaks are considered a step. 
threshold = np.mean(filt_mag)
peaks, heights = scipy.signal.find_peaks(filt_mag, height = threshold)

plt.figure(figsize=(20, 5))
plt.plot(filt_mag)
plt.plot(peaks, filt_mag[peaks], "x")
plt.title("Find Peaks")
plt.margins(0, .05)

plt.tight_layout()
plt.show()

#Go through all the data points in the peak heights data
step_count = []
for i in heights.values():
    for j in i:
        step_count.append(j)  
      

    
#Print Step count        
print("The total stepcount is = ", len(step_count))
