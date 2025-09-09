#! C:\Users\Vicon-OEM\AppData\Local\Programs\Python\Python311\python.exe
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:25:54 2025

@author: daniel.gregory@shrinenet.org
"""
import pandas as pd
import matplotlib.pyplot as plt
from get_AutoEvents import (get_Events as gev,
                            segments as gs)

#import Vicon Nexus Subroutines
from viconnexusapi import ViconNexus
vicon = ViconNexus.ViconNexus()

# get subject and file info
SubjectName = vicon.GetSubjectNames()[0]
FilePath, FileName = vicon.GetTrialName()

print(f'File path is: {FilePath}')
print(f'File name is: {FileName}')

fnp = FilePath + FileName

filenamepath = fnp.replace("\\","/")
print(f'File namepath is: {filenamepath}')

### ---------------------------------- Plotting (if we want?) -----------------
fig, ax = plt.subplots(2, 1)
filename = filenamepath.split('\\')[-1]
fig.suptitle(filename)

# figure out auto-detected events
# get segment position data and time/frame info into a dataframe using segment labels
labels = ['foot','toe','pelvis_4X4']
jerk_threshold = 10
seg_df, time_df, vfr = gs.get_Segments(filenamepath, FilePath, labels, jerk_threshold)

# get relative positions of the foot/toe segments
reference_segment = 'pelvis_4X4_x'
rel_df = gs.get_relPosition(seg_df, reference_segment)

# get absolute and relative segment velocities and summed velocities
mrk_vel, rel_vel, flip_direction = gs.get_segVelocity(seg_df, rel_df, vfr)

### ---------------------------------- Foot Off Events ------------------------
# potential tuning parameters
# - samples_perCycle
# - percent_peakHeight

# extract foot off events and show plots
events_off, samples_perCycle = gev.segment_FootOff(mrk_vel, rel_df, time_df, vfr, ax[0], flip_direction)

### ---------------------------------- Foot On Events -------------------------
# other potential tuning parameters
# - 

# direction = True
minThresh = 0.15
ffq = 0 # filter frequency for foot 'segment' data
k = 1 # shift

# extract foot off events and show plots
# print(f'Video Frame Rate is: {vfr}')
walk_direction = 'x'
events_onn = gev.segment_FootOn(mrk_vel, rel_df, time_df, vfr, ax[1], minThresh, ffq, k, samples_perCycle, flip_direction, walk_direction)

### --------------------------------- Combine & sort auto events --------------
events_auto = pd.concat([events_off, events_onn]).sort_values(by='Times', ascending=True)

### --------------------------------- Save events to c3d file -----------------
# format of adding events?
# Input
#   subject     = string, name of an existing subject
#   context     = string, name of the context. Valid values are: Left, Right, General
#   event      = string, name of the event type. Valid values are: Foot Strike, Foot Off, General
#   frame      = integer value, trial frame number as displayed in the application time bar
#   offset     = double value, offset (in seconds) from the beginning of the frame to the event occurrence
#                The value should be in the range of 0.00 to 1/FrameRate

# Usage Example:

#  vicon.CreateAnEvent( 'Patricia', 'Foot Strike', 'Left', 137, 0.0 )
for index, row in events_auto.iterrows():
    subject, context, event, frame, offset = SubjectName, row['Contexts'], row['Labels'], row['Frames'], 0.00
    vicon.CreateAnEvent(subject, context, event, frame, offset)

print('This seems to work')