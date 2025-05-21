# -*- coding: utf-8 -*-

# Py3_readC3D.py

"""
Created on Mon Mar 11 14:01:52 2024
@author: dgregory
"""

"""
Py3_readC3D_v2.py 

% AUTHOR(S) AND VERSION-HISTORY
% Matlab Ver. 1.0 Creation              (Alan Morris, Toronto, October 1998) [originally named "getc3d.m"]
% Matlab Ver. 2.0 Revision              (Jaap Harlaar, Amsterdam, april 2002)
% Matlab Ver. 3.0 Revision              (Ajay Seth, Stanford University, August 2007)
% R      Ver. 1.0 Creation              ()
% Python Ver. 1.0 Matlab Conversion     (Dan Gregory, Shriners Children's New England 2024)
% Python Ver. 2.0 R Conversion          (Dan Gregory, Bradley Nelson, Shriners Children's New England/Texas 2025)
 
This script was initally written by Dan Gregory PhD as a conversion from v3 of the
readC3D.m matlab file that was originally written by the above authors. Version
2 of this code was combined with a translation of the R readC3D code by the below
authors:
    Bruce A. MacWilliams, Shriners Children's and University of Utah
    Michael H. Schwartz, Gillette Children's Specialty Hospital and University of Minnesota
"""

############################## import libraries ###############################

import re
import Py3_DECbytes2PCbytes
import ReadC3DParameters
import numpy as np
import tkinter as tk
import pandas as pd
from tkinter import messagebox
from datetime import datetime

############################## readC3D functions ##############################
# Function to make names unique when marker names are repeated
def make_names_unique(num_markers, marker_names): 
    new_names = [] 
    if len(marker_names) == 0:
        for n in range(num_markers):
            new_names.append('Marker_' + str(n + 1))
    elif num_markers != len(marker_names):
        for n in range(len(marker_names)+1,num_markers+1):
            new_names.append('Marker_' + str(n))
        # print(new_names)

    marker_names = marker_names + new_names
    
    newMarker_names = []
    seen = {} 
    for name in marker_names: 
        if name not in seen: 
            seen[name] = 0 
            newMarker_names.append(name) 
        else: 
            seen[name] += 1 
            unique_name = f"{name}_{seen[name]}" 
            while unique_name in seen: 
                seen[name] += 1 
                unique_name = f"{name}_{seen[name]}" 
            seen[unique_name] = 0 
            newMarker_names.append(unique_name) 
    return newMarker_names

def get_C3Ddata(filenamepath, getData_dict=None):
    if getData_dict is None:
        # return all if no input is given
        GetVideoData            = True              # DEFAULT: True, Logical, If TRUE then return MarkerData dataframe
        GetAnalogData           = True              # DEFAULT: True, Logical, If TRUE then return AnalogData dataframe
        GetForceData            = True              # DEFAULT: True, Logical, If TRUE then return transformed ForceData dictionary, with each dictionary containing the individual force plate dataframe
        ForcePlateZero          = True              # DEFAULT: True, Logical, If TRUE then use FPZero parameter to determine force plate DC offsets, if FALSE then no offsets
        GetGaitCycleEvents      = True              # DEFAULT: True, Logical, If TRUE then returns GaitCycleEvents dataframe
        ForceDataFrames         = 'analog'          # DEFAULT: True, String, default 'analog', other option is 'video'
        ZeroForceBaseline       = True              # DEFAULT: True, Logical, If TRUE then sets all force plate data below *threshold to zero
        
    if getData_dict:
        GetVideoData            = getData_dict['GetMarkerData']             
        GetAnalogData           = getData_dict['GetAnalogData']            
        GetForceData            = getData_dict['GetForceData']             
        ForcePlateZero          = getData_dict['RemoveOffset']  
        GetGaitCycleEvents      = getData_dict['GetGaitCycleEvents'] 
        ForceDataFrames         = getData_dict['ForceDataFramesFormat']
        ZeroForceBaseline       = getData_dict['ZeroForceBaseline']
        
    # initiating data extraction types
    start_time = datetime.now()
    try:
        # format when used in folderselection and c3d files are looped in extraction process
        C3DFileName             = filenamepath.name # REQUIRED, String, Full path and name of the C3D file to be loaded
    except:
        # format when used to specify only filenamepath explicitly
        C3DFileName = filenamepath.split('\\')[-1]
        
    MiscData                = {}
    Header                  = {}    
    ParameterGroup          = {}
    CameraInfo              = []
    ResidualError           = []
    FirstReturnFrame        = 0
    LastReturnFrame         = 0
    ForceData               = {}                # ForceData is set to dictionary instead of dataframe because where there are more than one force plate used, the dictionary will store a dataframe for each force plate
    MarkerNames             = []
    VideoFrameRate          = 0
    AnalogFrameRate         = 0
    AnalogUnits             = []
    MarkerData              = pd.DataFrame()    # initialize empty dataframe 
    AnalogData              = pd.DataFrame()    # initialize empty dataframe 
    GaitCycleEvents         = pd.DataFrame()    # initialize empty dataframe 
   
    ############################ OPEN FILE ########################################
    # open c3d file for read access in binary format
    with open(filenamepath, 'rb') as file:
        # Word 1: reading record number of parameter section
        NrecordFirstParameterblock = np.fromfile(file, dtype='int8', count=1)
        
        # key should equal 80
        key = np.fromfile(file, dtype='int8', count=1)
        
        if key[0]!= 80:
            tk.messagebox.showerror('C3D ERROR', 'File does not comply with .c3d format! \n Application error')
            file.close()
            quit()
            
        # set file position indicator, jumps to processor type - field
        # Matlab code originally has +3 at the end of this line of code making the fileoffset = 515
        # removing the +3 sets the value to 512 which is the size of a data section and will move to the beginning of the block (not where we want to start)
        # fileoffset = 512*(NrecordFirstParameterblock)[0]-1)+3
        FirstParameterByteNumber = 512*(NrecordFirstParameterblock[0]-1)
        # print(fileoffset)
        
        # move to specific postiion in file that is fileoffset[0] bytes from beginning of file, which is the beginning of the next block at position 512
        file.seek(FirstParameterByteNumber,0)
        
        
        # Matlab code originally had -83 at the end of this code which when starting at position 515 gives '84'
        # Original code had proctypes of 1(intel), 2(DEC), 3(MIPS); new code shifts to position 4 of proctype (position 515 in file) which is interpreted as below
        # proctype: 84(INTEL-PC); 85(DEC-VAX); 86(MIPS-SUN/SGI)
        # proctype = np.fromfile(file, dtype='int8', count=1)-83
        proctype = np.fromfile(file, dtype='int8', count=4)
        # print(proctype)
        
        if proctype[3]==84:
            filetype            = 'PC'
            
        elif proctype[3]==85:
            filetype            = 'DEC'
           
        elif proctype[3]==86:
            file.close()
            print('File access type is MIPS-SGI binary format which is not currently supported. ')
            # Need data place holders to export
            finish_time         = datetime.now()
            extraction_duration = finish_time - start_time
            filetype            = 'MIPS-SGI'
            return ParameterGroup, Header, MarkerData, AnalogData, ForceData, GaitCycleEvents, MiscData, extraction_duration, filetype, C3DFileName
        print(f'File type: {filetype}')
       
    ############################ READ HEADERS #####################################
        
    ###############################################################################
    
        # move to specific postiion in file that is 2 bytes from beginning of file
        file.seek(2,0) 
       
        # Word 2: Number of 3D points (markers), unsigned 2 byte integer
        NumberOfMarkers = int(np.fromfile(file, dtype='int16', count=1))
        
        # Flag for 3D data inclusion
        if NumberOfMarkers == 0:
            # if no marker data, set GetVideoData to false
            GetVideoData = False
            tk.messagebox.showerror('C3D ERROR', f'Marker data requested but not found in file {filenamepath}')

        # Word 3: number of channels collected in the duration between two camera frames for all analog channels collected
        NanalogChannelsPerVideoFrame = np.fromfile(file, dtype='int16', count=1)
        
        # Word 4: number of first video frame in c3d file
        # C3D User Guide cautions against using header information for number of frames.
        # These entries have a 2 byte upper limit which can cause issues, read here to 
        # report what is actually in header, but use values from POINT parameters for coding
        # FirstC3DFrameNumber = np.fromfile(file, dtype='int16', count=1)
        data = np.frombuffer(file.read(2), dtype=np.uint16)
        FirstC3DFrameNumber = data[0]
        
        # Word 5: Number of last video frame in c3d file; max value 65535
        # LastC3DFrameNumber = np.fromfile(file, dtype='int16', count=1)
        data = np.frombuffer(file.read(2), dtype=np.uint16)
        LastC3DFrameNumber = data[0]
        
        # Calculate number of frames in c3d header info
        # Should always exist even if wrong - if max detected, set to -1
        if LastC3DFrameNumber == 65535:
            HeaderC3DFrames = -1
        else:
            HeaderC3DFrames = LastC3DFrameNumber - FirstC3DFrameNumber
                
        # Word 6: maximum interpolation gap allowed (in frames)
        MaxInterpolationGap = np.fromfile(file, dtype='int16', count=1)
        
        # Word 7, 8: floating-point scale factor to convert 3D-integers to ref system units
        # If negative, 3D data is already in REAL*4 format - from R code
        if filetype == 'DEC': # DEC type format
            VideoScaleFactor = Py3_DECbytes2PCbytes.DEC2PC(np.fromfile(file, dtype=np.uint8, count=4))
        else:
            VideoScaleFactor = np.fromfile(file, dtype='float32', count=1)
        
        # Word 9: starting record number for 3D point and analog data
        FirstDataRecordNumber = np.fromfile(file, dtype='int16', count=1 )
        
        # Word 10: number of samples collected in the duration between two camera frames for a single analog channel
        AnalogToVideoRate = np.fromfile(file, dtype='int16', count=1)
        
        
        if NanalogChannelsPerVideoFrame[0] > 0 and AnalogToVideoRate[0] > 0:
            # GetAnalogData = True
            # added the "try" set here, the original code is the "except" line of code
            try:
                NanalogChannels=[NanalogChannelsPerVideoFrame[0]/AnalogToVideoRate[0]]
            except:
                NanalogChannels=[NanalogChannelsPerVideoFrame/AnalogToVideoRate]
        else:
            GetAnalogData           = False
            NanalogChannels         = [0]
            if GetAnalogData:
                # messagebox.showinfo("WARNING", "Analog data requested but no analog data detected in file")
                print('WARNING: Analog data requested but no analog data detected in file')

        
        # Word 11, 12: (REAL*4) Video frame rate (Hz), - not necessarily an integer - 
        if filetype == 'DEC': # DEC type format
            VideoFrameRate = Py3_DECbytes2PCbytes.DEC2PC(np.fromfile(file, dtype=np.uint8, count=4))
        else:
            VideoFrameRate = np.fromfile(file, dtype='float32', count=1)
       
        # 3D & analog data format (Real or Integer) designated by sign of VideoScaleFactor (aka  POINT:SCALE)
        DataFormat = None
        
        if VideoScaleFactor < 0:
            Real = True
            DataFormat = 'Real'
            VideoByteLength = NumberOfMarkers*16                # 16 bytes per marker =  x,y,z,residual&contribution
            AnalogSampleLength = NanalogChannels*4              # Assumes 4 bytes per data point
            VideoScaleFactor = 1                                # Set to 1 for 'No Scaling'
        else:
            Real = False
            DataFormat = 'Integer'
            VideoByteLength = NumberOfMarkers*8                 # 8 bytes per marker =  x,y,z,residual&contribution
            AnalogSampleLength = NanalogChannels*2              # Assumes 2 bytes per data point
        
        # Calculate parameters
        AnalogFrameRate = VideoFrameRate*AnalogToVideoRate
        FirstDataByteNumber = 512*(FirstDataRecordNumber[0] - 1)
        ParameterByteSize = FirstDataByteNumber - FirstParameterByteNumber
        if ParameterByteSize <= 0:
            messagebox.showinfo("C3D ERROR", f"Invalid ParameterByteSize for file: {filenamepath}")
        
        
    ############################ READ EVENTS ######################################
        
    ###############################################################################
    
        # Word 13-149 unused, go to word 150
        file.seek(149*2,0)
        
        # Word 150: Look 298 bytes from beginning of file for Event Flag (should be 12345) to indicate presence of event data 
        EventFlag = np.fromfile(file, dtype='int16', count=1)
        HeaderEventSwitches     = []
        HeaderEventLabels       = []
        Nevents                 = None
        
        if EventFlag[0] == 12345:
            # Word 151: Number of defined time events -- max 18
            Nevents = np.fromfile(file, dtype=np.uint16, count=1)
            
            # below is equivalent to file.seek(2,1) - look 2 bytes (or 1 position) ahead - in seek, the second input when == 1 indicate "from current position"
            file.seek(152*2,0)
            
            # Initialize HeaderEventTimes array
            HeaderEventTimes = np.zeros(Nevents[0], dtype=np.float64)
            
            if Nevents[0]>0:
                # Word 152: not used
                # Word 153-188 (REAL*4) Event times (in seconds) -- max 18
                
                if filetype == 'DEC':
                    for et in range(Nevents[0]):
                        # read 4 bytes from the file
                        dec_bytes = file.read(4)
                        # convert the byte data using DEC function
                        HeaderEventTimes[et] = Py3_DECbytes2PCbytes.DEC2PC(dec_bytes)
                        
                else:
                    HeaderEventTimes = np.frombuffer(file.read(Nevents[0] * 4), dtype=np.float32)
                    
                # Word 189-198 (BYTE*1) Event display switches (0=OFF, 1=ON)
                # This section of code was not commented in the matlab script - unsure if the R code directly translates here
                file.seek(188*2,0) # seek from beginning of file 188*2 bytes
                
                Bytes = np.frombuffer(file.read(Nevents[0]), dtype=np.uint8)
                HeaderEventSwitches = np.where(Bytes == 1, 'ON', 'OFF')
                    
                # Word 199-234 Event labels. 4 characters for each event
                file.seek(198*2,0)
                # need function to split charachters
                def split_every_n_chars(s, n): 
                    return [s[i:i+n] for i in range(0, len(s), n)]
                
                byte_data = file.read(4 * Nevents[0])
                char_string = byte_data.decode('latin1')
                
                HeaderEventLabels = split_every_n_chars(char_string, 4)
                
    ############################ READ PARAMETERGROUP DATA #####################
        
    ###########################################################################
    ParameterGroup = ReadC3DParameters.GetParameters(filenamepath, file, NrecordFirstParameterblock[0], FirstParameterByteNumber, filetype)
    
    ############################ READ DATA ####################################
    
    ###########################################################################
    #### ---- Get Trial Parameters ----
    # Only Call optional group subroutines if that group is present in the file
    
    # Camera_Rate <- GetParameterData(Parameters,  "TRIAL", "CAMERA_RATE")
    
    #### ---- Get Point Parameters ---- 
    # Get point names and parameters to scale stored 3D values
    # Even if there are no markers there is still a POINT group with valid USED and FRAMES parameters
    
    # Number of points used, these can be markers, modeled markers, or variables
    # This is the same as NumberOfMarkers from header so don't actually need to read parameter
    # NumberOfC3DPoints <- GetParameterData(Parameters, "POINT", "USED")
  
    # Calculate number of frames in C3D from header info, these should always exist
    # even if they are wrong. If max is detected set to -1
    
    if LastC3DFrameNumber == 65535:
        HeaderC3DFrames = -1 
    else:
        HeaderC3DFrames = LastC3DFrameNumber - FirstC3DFrameNumber + 1
    
    # C3D User Guide recommends using POINT:FRAMES for number of frames, not header info
    # PointFrames will be -1 if >2^16 frames
    # Keep warning on for this one as it should always be there:
    try:
        PointFrames = ParameterGroup['POINT']['FRAMES']['data'][0]
    except:
        PointFrames = HeaderC3DFrames
    
    if not np.isnan(PointFrames):
        # Account for signed integer possibility from PointFrames
        if PointFrames < -1:
            NumberOfC3DFrames = PointFrames + 65536
        else:
            NumberOfC3DFrames = PointFrames
    else:
        # Use the header value though this should never happen unless > 2^16 frames
        # in which case both HeaderC3DFrames and PointFrames = -1
        NumberOfC3DFrames = HeaderC3DFrames
        
    #################### NEW to fix ANALOG:OFFSET issues ######################
    # Update LastC3DFrame when LastC3DFrame = 65535
    if LastC3DFrameNumber - FirstC3DFrameNumber + 1 < NumberOfC3DFrames:
        LastC3DFrameNumber = NumberOfC3DFrames + FirstC3DFrameNumber - 1
    
    # If HeaderC3DFrames == -1 this indicates overflow of number of frames for 
    # integer storage. This will also be reflected by the header variable 
    # LastC3DFrameNumber == 65535
    
    # Handle files with number of frames exceed integer storage as read in the header.
    # This is system dependent: 
    # Vicon writes two parameters for all files  TRIAL:ACTUAL_START_FIELD and 
    # TRIAL:ACTUAL_END_FIELD. For files with less than 2^16 frames, these return 
    # the same values as the header values read here as FirstC3DFrameNumber and 
    # LastC3DFrameNumber. 
    # C-Motion stores the total frame count as a single floating-point value in
    # the parameter POINT:LONG_FRAMES. 
    
    # TRIAL:ACTUAL_START_FIELD and TRIAL:ACTUAL_END_FIELD  return two signed 
    # integers in a vector c(low_int, high_int), need to combine to long integer.
    # According to Edi Cramp, this is a Vicon issue, these are not written correctly 
    # as scalar so need to convert. BUT, as these are read in the header, there is no 
    # reason to read these again from teh parameter section unless the frames 
    # exceed integer storage 2^16 in which case the header will return 
    # LastC3DFrameNumber = 65535. 
    # In theory the integer representing the higher bytes could also be negative,
    # but that is not realistic so we will not attempt to convert.     
    
    if NumberOfC3DFrames == -1:
        # First address the Vicon method
        # Not realistic that the first frame could be > 2^16 but check anyway ...
        # Account for possible NA return
        
        # ACTUAL_START_FIELD
        ints = ParameterGroup['TRIAL']['ACTUAL_START_FIELD']['data']
        
        if not np.isnan(ints[0]):
            # low_int value can be negative since it is stored as a signed int
            if ints[0] < 0:
                ints[0] = ints[0] + 65536
            
            # Combine unsigned low_int and high_int into long int
            FirstC3DFrameNumber = ints[0] + 65536 * ints[1]
            
        # ACTUAL_END_FIELD
        ints = ParameterGroup['TRIAL']['ACTUAL_END_FIELD']['data']
        
        if not np.isnan(ints[0]):
            # low_int value can be negative since it is stored as a signed int
            if ints[0] < 0:
                ints[0] = ints[0] + 65536
            
            # Combine unsigned low_int and high_int into long int
            LastC3DFrameNumber = ints[0] + 65536 * ints[1]
            NumberOfC3DFrames = LastC3DFrameNumber - FirstC3DFrameNumber + 1
            
        # Next check for LONG_FRAMES option which stores the number of frames, not start/end
        # Account for possible NA return
        # Also, fix to account for two integer POINT:LONG_FRAMES 
        try:
            longframes = ParameterGroup['POINT']['LONG_FRAMES']['data']
        
            if not np.isnan(longframes):
                # Check for two-integer POINT:LONG_FRAMES
                if len(longframes) == 2:
                    NumberOfC3DFrames = longframes[0] + 65536 * longframes[1]
                    
                if longframes[0] > NumberOfC3DFrames:
                    NumberOfC3DFrames = longframes[0]
        except:
            print('Looking for LONG_FRAMES to establish NumberOfC3DFrames but no LONG_FRAMES parameter key found. NumberOfC3DFrames value will remain as: LastC3DFrameNumber - FirstC3DFrameNumber + 1')
        
    # It is possible that some system storage for >2^16 frames has not been detected 
    # and NumberOfC3DFrames still does not have a real value if none of the above 
    # parameters were found so warn and use 2^16 frames:
    if NumberOfC3DFrames == -1:
        NumberOfC3DFrames = LastC3DFrameNumber - FirstC3DFrameNumber + 1
        tk.messagebox.showerror('C3D ERROR', f'Header frame length is maximum and no parameters found for file: {filenamepath}\n Data will be trimmed to 65536 frames')
    
    # If a file is corrupt there still may not be a value so exit
    if np.isnan(PointFrames) & HeaderC3DFrames == -1:
        tk.messagebox.showerror('C3D ERROR', f'Number of frames not read for file: {filenamepath}')
        finish_time         = datetime.now()
        extraction_duration = finish_time - start_time
        return ParameterGroup, MarkerData, AnalogData, ForceData, GaitCycleEvents, MiscData, extraction_duration, filetype, C3DFileName
    
    # ensure number of c3d frames is in integer format for downstream calculations
    try:
        NumberOfC3DFrames = int(NumberOfC3DFrames)
    except:
        NumberOfC3DFrames = int(NumberOfC3DFrames[0])
    
    
    # From C3D Manual: Traditionally, all integers used in the parameter section were 
    # stored as one's complement signed integers with a range of –32767 to +32767 
    # and all bytes were one's complement signed bytes with a range of –127 to +127. 
    # However, some parameters may use unsigned integers to store data that will never 
    # have a negative value. There is no flag to indicate that a C3D file uses 
    # unsigned integers in the parameter section.
    
    # Pretty sure these are now an obsolete conditions
    # So if NumberOfC3DFrames is negative we need to adjust
    # if (NumberOfC3DFrames < 0) {NumberOfC3DFrames <- NumberOfC3DFrames + 65536}
    
    # Account for 2 byte limitation in end frame value from header if there are more than 65,535 frames
    # if (NumberOfC3DFrames > LastC3DFrameNumber) {
    #   LastC3DFrameNumber <- NumberOfC3DFrames + FirstC3DFrameNumber - 1
    # }

    # Verify that the (optional) user requested start frame number is valid for this data set
    if (FirstReturnFrame > 0 and FirstReturnFrame < FirstC3DFrameNumber) or ((FirstReturnFrame > LastC3DFrameNumber)):
        print(f'ReadC3D:Trial Parameters: Specified FirstReturnFrame is invalid! Valid frame range is {FirstC3DFrameNumber} to {LastC3DFrameNumber}')
        finish_time         = datetime.now()
        extraction_duration = finish_time - start_time
        return ParameterGroup, MarkerData, AnalogData, ForceData, GaitCycleEvents, MiscData, extraction_duration, filetype, C3DFileName
    
    elif FirstReturnFrame == 0:
        # Default: Set the FirstC3DFrameNumber to the first valid video frame number
        FirstReturnFrame = FirstC3DFrameNumber
        
    # Verify that the (optional) end frame number is valid for this data set
    if (LastReturnFrame > 0 and LastReturnFrame > LastC3DFrameNumber):
        print(f'ReadC3D:Trial Parameters: Specified LastReturnFrame is invalid! Valid frame range is {FirstC3DFrameNumber} to {LastC3DFrameNumber}')
        finish_time         = datetime.now()
        extraction_duration = finish_time - start_time
        return ParameterGroup, MarkerData, AnalogData, ForceData, GaitCycleEvents, MiscData, extraction_duration, filetype, C3DFileName
    
    elif LastReturnFrame == 0:
        # Default: Set the LastC3DFrameNumber to the last valid video frame number
        LastReturnFrame = LastC3DFrameNumber
        
    # Verify the start frame is before or equal to the end frame 
    if FirstReturnFrame > LastReturnFrame:
        print(f'ReadC3D:Trial Parameters: Specified FirstReturnFrame is after specified LastReturnFrame! Valid frame range is {FirstC3DFrameNumber} to {LastC3DFrameNumber}')
        finish_time         = datetime.now()
        extraction_duration = finish_time - start_time
        return ParameterGroup, MarkerData, AnalogData, ForceData, GaitCycleEvents, MiscData, extraction_duration, filetype, C3DFileName
    
    # Get other point parameters only if there is any marker data
    if GetVideoData:
        # Read Point:Labels parameter group
        
        # ------------------- IMPORTANT R CONVERSION COMMENT ------------------
        # ---------------------------------------------------------------------
        
        # R code gets initial MarkerNames data here, then handles extra LABELSx 
        # data extraction separately by searching for additional 'LABELS2' etc.
        
        # The below python code should be able to handle any  number of 'LABELSx'
        # key:value pairs without the extra search
        
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
    
        # Likely a Vicon issue as there is no need to store this way, but Vicon files
        # limit the point labels and descriptions parameters to 255 entries.
        # If there are more points then these are stored in LABELS2, LABELS3, etc. 
        # and DISCRIPTIONS2, DISCRIPTIONS3, etc. Arrays are from 1:255 so not 256. 
        # Find the number of label groups in Parameters
        
        MNlist      = []
        MarkerNames = []
        
        for key in ParameterGroup['POINT'].keys():
            # recursively search for 'LABELS', 'LABELS2', etc.
            if 'LABELS' in key:
                    MNlist.append(ParameterGroup['POINT'][key]['data'])
        
        mnDims = ParameterGroup['POINT']['LABELS']['dimensions']
        max_length = int(mnDims[0]) # max marker label length
        mnList_length = int(mnDims[1])
        
        MNames = []
        if type(MNlist) == list:
            for markerlist in MNlist:
                # break the string by given dimensions
                for i in range(0, max_length * mnList_length, max_length):
                    MNames.append(markerlist[i:i+int(mnDims[0])])
        else:
            for i in range(0, max_length * mnList_length, max_length):
                MNames.append(MNlist[i:i+int(mnDims[0])])
        
        # need to remove any trailing or leading spaces if they remain        
        MarkerNames = [name.strip() for name in MNames if name != '']
        
        # If marker names list is still not the correct length or if more than
        # one marker has the same name, append genaric names to list and/or 
        # create unique names
        if NumberOfMarkers > len(MarkerNames):
            MarkerNames = make_names_unique(NumberOfMarkers, MarkerNames)
            
        # Just to be sure, trim to length of NumberOfMarkers
        MarkerNames = MarkerNames[:NumberOfMarkers]
   
    try:
        markerUnits = ParameterGroup['POINT']['UNITS']['data']
    except:
        markerUnits = 'None'
    
    if re.search('mm', markerUnits):
        MarkerUnitsScaleFactor = 1
    elif re.search('cm', markerUnits):
        MarkerUnitsScaleFactor = 10
    elif re.search('m', markerUnits):
        MarkerUnitsScaleFactor = (1000)
    else:
        MarkerUnitsScaleFactor = -1
        
    ###### Analog Parameters ########
    # if force data is requested, analog data is required
    if GetForceData and not GetAnalogData:
        GetAnalogData = True
        
    if GetAnalogData:
        try:
            Analog_format = ParameterGroup['ANALOG']['FORMAT']['data']
        except:
            Analog_format ='SIGNED'
        
        try:
            Analog_Offset = ParameterGroup['ANALOG']['OFFSET']['data']
        except:
            Analog_Offset = None
        
        # MAC Systems labels this as "OFFSETS"
        if Analog_Offset is None:
            try:
                Analog_Offset = ParameterGroup['ANALOG']['OFFSETS']['data']
            except:
                # if no offsets in file, specify as zeros, length of NanalogChannels
                Analog_Offset = np.zeros(int(NanalogChannels[0]))
        
        # Fix analog offsets for unsigned storage
        if Analog_format == 'UNSIGNED' or any(x < 0 for x in Analog_Offset):
            # Get BITS
            try:
                AB = ParameterGroup['ANALOG']['BITS']['data'][0]
            except:
                # If no BIT parameter then assume 16 as this is default standard
                AB = 16
                
            def bitwise_and(value, mask):
                return value & mask
    
            Analog_Offset = [bitwise_and(x, 0xFF) if AB == 8 else
                                bitwise_and(x, 0xFFFF) if AB == 16 else
                                bitwise_and(x, 0xFFFFFFFF) if AB == 32 else
                                bitwise_and(x, 0xFFFFFFFFFFFFFFFF) if AB == 64 else
                                x for x in Analog_Offset]
   
        # Trim to number of analog channels, these can sometimes have more entries than channels
        Analog_Offset = Analog_Offset[:int(NanalogChannels[0])]
        # print(f'Analog Offset is: {Analog_Offset}')
        
        # Channel scale factors
        try:
            AnalogScaleMultiplier = ParameterGroup['ANALOG']['SCALE']['data']
        except:
            # if no scale multiplier in file, specify as array of ones, length of NanalogChannels
            AnalogScaleMultiplier = np.zeros(int(NanalogChannels[0])) + 1
        
        # Trim to number of analog channels, these can sometimes have more entries than channels
        AnalogScaleMultiplier = AnalogScaleMultiplier[:int(NanalogChannels[0])]
        
        # General Scale Factor
        try:
            AnalogGeneralScale = ParameterGroup['ANALOG']['GEN_SCALE']['data']
        except:
            # if no general scale in file, specify as '1'
            AnalogGeneralScale = [1]
        
        # Units
        try:
            AUnits = ParameterGroup['ANALOG']['UNITS']['data']
             # need to split string into individual strings by spaces and remove any
            # trailing or leading spaces if they remain
            AnalogUnits = [unit.strip() for unit in AUnits.split()]
            
            # Trim to number of analog channels
            AnalogUnits = AnalogUnits[:int(NanalogChannels[0])]
        except:
            AnalogUnits = ['none' for unit in range(int(NanalogChannels[0])) ]
        
        # Want our units to be standardized to Volts and Nmm, if Nm scale by 1000 by channel scale
        if 'Nmm' in AnalogUnits and 'V' in AnalogUnits:
            AnalogScaleMultiplier = 1*AnalogScaleMultiplier         # already in desired units
        elif 'Nm' in AnalogUnits and 'mV' in AnalogUnits:
            AnalogScaleMultiplier = 1000*AnalogScaleMultiplier     # scale up by 1k
            # update AnalogUnits list to match new scales of Nmm and V
            AnalogUnits = ['Nmm' if unit == 'Nm' else 'V' if unit == 'mV' else unit for unit in AnalogUnits]
        else:
            AnalogScaleMultiplier = 1*AnalogScaleMultiplier
        
        # ------------------- IMPORTANT R CONVERSION COMMENT ------------------
        # ---------------------------------------------------------------------
        # The above check is ensuring that the units are Volts and Nmm. If they 
        # are mV and Nm, in addition to changing the scale factor, we should be
        # updating the list of AnalogUnits to reflect this change, yes???    
        
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        
        # First, last, and number of analog frames to be returned    
        # FirstReadADC = (FirstReturnFrame - 1) * AnalogToVideoRate + 1
        # LastReadADC = LastReturnFrame * AnalogToVideoRate
        # NumberofADCFrames = NumberOfC3DFrames * AnalogToVideoRate

       
    #### ---- Read All Data ---- 
    # Instead of only reading marker or analog, read everything all at once since the 
    # data are interwoven and the entire block has to be read anyway. This should be 
    # faster. By everything I mean everything. Even if user passes start and stop return frames
    # read in all data because force plate offset ranges may be outside of start/stop frames
    # and needed for zeroing.
    
    # Data Structure
    # For 1 Video Frame:
    #   All Markers 4 bytes for X, Y, Z, R if Real or 2 bytes if int
    #   All Analog data for that frame
    #   Each analog sub-frame of data:
    #     4 bytes for each analog channel if Real or 2 bytes if int
    FirstReadADC        = (FirstReturnFrame - 1) * AnalogToVideoRate + 1
    LastReadADC         = LastReturnFrame * AnalogToVideoRate
    NumberofADCFrames   = NumberOfC3DFrames * AnalogToVideoRate
    # Check to make sure the passed values will work
    if GetVideoData:
        if FirstReturnFrame == 0:
            print("Warning: ReadC3D: Read All Data: FirstReturnFrame = 0")
        if LastReturnFrame == 0:
            print("Warning: ReadC3D: Read All Data: LastReturnFrame = 0")
        if NumberOfMarkers == 0:
            print("Warning: ReadC3D: Read All Data: NumberOfMarkers = 0")
            
    if GetAnalogData:
        if FirstReadADC == 0:
            print("Warning: ReadC3D: Read All Data: FirstReadADC = 0")
        if LastReadADC == 0:
            print("Warning: ReadC3D: Read All Data: LastReadADC = 0")
        if NanalogChannels == 0:
            print("Warning: ReadC3D: Read All Data: NumberOfAnalogChannels = 0")
            
    # Set pointer to start of video data
    with open(filenamepath, 'rb') as file:
        file.seek(FirstDataByteNumber, 0)

        # Read in all bytes for all frames as real values
        # readBin has an object property, should try to see if you can read a matrix directly into it
        VideoColumns = int(4 * NumberOfMarkers + NanalogChannelsPerVideoFrame)
        if Real:
            if filetype == "DEC":
                # If DEC/Real need to step through and do the byte conversions
                # Read raw vector
                VidBytes  = file.read(4* VideoColumns * NumberOfC3DFrames)
                # Byte reordering
                dec_index   = [2,3,0,1]
                # Convert data to matrix
                # VideoBytes  = np.array(VideoBytes)
                VidBytes = np.frombuffer(VidBytes, dtype=np.uint8)
                VidBytes  = VidBytes.reshape((4, VideoColumns * NumberOfC3DFrames))
                # Use indicies to swap
                VidBytes  = VidBytes[dec_index, :].flatten()
                # Now compute floats, divide by four to account for exponent adjustment
                # The struct.unpack function is used to interpret bytes as packed binary data.
                ########################## NOTE ###############################
                # len(VideoBytes)//4 calculates the number of 4-byte chunks in 
                # the VideoBytes array, because each float is represented by 4 bytes.
                #
                # The f-string f'{len(VideoBytes)//4}f' dynamically creates a 
                # format string for unpacking. If len(VideoBytes)//4 equals 10, 
                # the format string becomes '10f', meaning “interpret the data 
                # as 10 floating point numbers.”
                #
                # The struct.unpack function unpacks the binary data in VideoBytes 
                # according to the given format string. The format string '10f' 
                # indicates that the data should be read as 10 floats. 
                # The result is a tuple of float numbers.
                # VideoBytes = np.array(struct.unpack(f'{len(VideoBytes)//4}f', VideoBytes)) / 4
                VideoBytes = np.frombuffer(VidBytes, dtype=np.float32, count=(VideoColumns * NumberOfC3DFrames))
                VideoBytes /= 4
                # Equivalent to subracting 01 from last byte
                # VideoBytes  = VideoBytes[:VideoColumns * NumberOfC3DFrames * 4]
                # VideoBytes  = VideoBytes.view(dtype = np.float32).reshape(VideoColumns*NumberOfC3DFrames)
                # VideoBytes  = VideoBytes/4
            else:
                # PC/Real data format
                # videoData   = file.read(4*VideoColumns*NumberOfC3DFrames)
                # VideoBytes  = np.frombuffer(videoData, dtype=np.float32)
                VideoBytes = np.fromfile(file, dtype=np.float32, count=(VideoColumns * NumberOfC3DFrames))
                
        else:
            # Integer format
            # videoData   = file.read(2 * VideoColumns * NumberOfC3DFrames)
            # VideoBytes  = np.frombuffer(videoData, dtype=np.int16)
            VideoBytes = np.fromfile(file, dtype=np.int16, count=(VideoColumns * NumberOfC3DFrames))
    
    # Reshape VideoBytes into two matrices
    # 1) VideoData: All marker and caclulated 3D variables
    # 2) ADCData: All analog channels
    # Start by making a matrix with rows of frame numbers and columns will be 3D data 
    # and every analog having NumberOfAnalogChannelsPerFrame columns
    VideoMatrix = VideoBytes.reshape(NumberOfC3DFrames, VideoColumns, order = "C")
    
    # Now split this matrix of everything into Marker and ADC matrices
    
    ### Create Video Data Frame
    # Video data ih columns of NumberofMarkers by rows of NumberofC3DFrames
    if GetVideoData:
        MarkerData_matrix = VideoMatrix[:NumberOfC3DFrames, :4*NumberOfMarkers] * VideoScaleFactor * MarkerUnitsScaleFactor
        
        # ------------------- IMPORTANT R CONVERSION COMMENT ------------------
        # ---------------------------------------------------------------------
        
        # The matrix() function in R uses column-major order (Fortran-style) by 
        # default when reshaping or creating matrices.
        # This means that it fills the matrix column by column.
        # In Python, to match this behavior, the order='F' parameter should be used
        # in the reshape function when converting R code to Python using NumPy.
        
        MarkerData_matrix = MarkerData_matrix.reshape((4*NumberOfC3DFrames, NumberOfMarkers), order='F')
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        
        # Make MarkerData matrix into a data frame with ID columns for XYZR coordinates and video frame numbers
        Coord = np.concatenate([np.repeat('X', NumberOfC3DFrames),
                                np.repeat('Y', NumberOfC3DFrames),
                                np.repeat('Z', NumberOfC3DFrames),
                                np.repeat('R', NumberOfC3DFrames)])
        Frame = np.tile(np.arange(FirstC3DFrameNumber,LastC3DFrameNumber+1), 4)
        # Apply marker names to columns, remove all residual rows, and filter to optionally requested frame range
        MarkerData              = pd.DataFrame(MarkerData_matrix, columns=MarkerNames)
        MarkerData['Coord']     = Coord
        MarkerData['Frame']     = Frame
        
        # Filter the DataFrame for the Frame between FirstReturnFrame and LastReturnFrame
        MarkerData = MarkerData[(MarkerData['Frame'] >= FirstReturnFrame) & (MarkerData['Frame'] <= LastReturnFrame)]
        
        # ---------------------------- COMMENT AND NOTE -----------------------
        # ---------------------------------------------------------------------
        
        # Unsure if we care to remove the residual data from the dataframe
        # Likely won't change the speed of data extraction and will minimally 
        # impact data storage capacity. 
        # I vote to just leave it there. All data in the Pandas dataframe will
        # be easy to extract
        
        # Need to determine the likelihood that First and LastReturnFrames will
        # be different from First and LastC3DFrameNumber. According to above code,
        # First and LastReturnFrame numbers will always be the same as First and
        # LastC3DFrameNumber except when there are issues, in which case the 
        # code is exited with a return of "ParameterGroup". This makes me think
        # the line of code below to 'trim' the data to First and LastReturnFrame
        # is never going to perform any additional trimming because the Frame
        # list defined above used the First and LastC3DFrameNumber to define its
        # values. 
        
        # code above seems to take into account if there are descrepancies, 
        # but FirstReturnFrame is set to zero initially
        
        #Filter rows where 'Coord' is not 'R' and is between FirstReturnFrame and LastReturnFrame
        
    ############## Create Analog Data Frame ##################
    # if force data is requested, analog data is required
    if GetForceData and not GetAnalogData:
        GetAnalogData = True
        
    if GetAnalogData:
        # Analogs are stored in colunms of 0:analogchannel repeated AnalogToVideoRateTimes
        # Use the remaining columns of the VideoMatrix which contain ADC channel data
        ADCdata = VideoMatrix[0:NumberOfC3DFrames, (4*NumberOfMarkers):VideoColumns]
        # Reshape into 3D array compiant with dimensions and data storage order
        ADCdata = ADCdata.reshape(NumberOfC3DFrames, int(NanalogChannels[0]), AnalogToVideoRate[0], order='F')
        # Permutate so that sub-frames are first dimension
        ADCdata = np.transpose(ADCdata,(2,0,1))
        # Reshape back to matrix so that sub-frames are now repeated through video frames
        ADCdata = np.reshape(ADCdata,(int(NumberofADCFrames[0]), int(NanalogChannels[0])), order='F')
        # Scale and offset: Offset values are not scaled so need to be applied first then scaled
        Analog_Offset   = np.array(Analog_Offset).reshape(-1,1)
        AnalogScaleMultiplier = np.array(AnalogScaleMultiplier).reshape(-1,1)
        ADCdata         = (ADCdata.T - Analog_Offset) * AnalogScaleMultiplier
        ADCdata         = (ADCdata.T) * int(AnalogGeneralScale[0])
        
        # Make ADCData into a data frame with ID 3 ID columns: ADC frames (each observation), Video frames, Sub-frames 
        # Set the column names as stored in ANALOG:LABELS parameter
        ANlist = ParameterGroup['ANALOG']['LABELS']['data']
        
        anDims = ParameterGroup['ANALOG']['LABELS']['dimensions']
        max_length = int(anDims[0]) # max marker label length
        anList_length = int(anDims[1])
        
        ANames = []
        for i in range(0, max_length * anList_length, max_length):
            ANames.append(ANlist[i:i+int(anDims[0])])
        
        # need to remove any trailing or leading spaces if they remain        
        ANames = [name.strip() for name in ANames if name != '']
        
        # Trim to number of analog channels
        # ANames = ANames[:int(NanalogChannels[0])]
        
        # If analog names list is still not the correct length or if more than
        # one channel has the same name, append genaric names to list and/or 
        # create unique names
        if int(NanalogChannels[0]) > len(ANames):
            ANames = make_names_unique(int(NanalogChannels[0]), ANames)
        
        # Just in case, trim to NanalogChannels length
        ANames = ANames[:int(NanalogChannels[0])]
        
        # replace AnalogNames where a period is present, replace with underscore
        # ANames = [name.replace('.','_') for name in ANames]
        
        # add frame column names
        AnalogNames = ['AnalogFrames', 'VideoFrames','SubFrames'] + ANames
        
        # Create frame number data
        ############################# NOTE WARNING ############################
        '''
        NOTE: the data type of NumberOfC3DFrames and AnalogToVideoRate apprears
        to be int.16, leading to a RunTimeWarning of overflow for scalar multiply
        that is too large for the int.16 data type. 
        
        Wrapping each variable in the int() function SHOULD resolve this issue
        but this may need to be modified if issues arise in the assignment of
        AnalogFrames, VideoFrames, and SubFrames to the AnalogData dataframe.
        '''
        #######################################################################
        AnalogFrames = FirstReadADC + pd.Series(range(NumberOfC3DFrames * int(AnalogToVideoRate[0])))
        VideoFrames = np.repeat(np.arange(FirstC3DFrameNumber, LastC3DFrameNumber + 1), AnalogToVideoRate, axis=0)
        SubFrames = np.tile(np.arange(1, AnalogToVideoRate + 1), NumberOfC3DFrames)
        
        AnalogData = pd.DataFrame(np.hstack((np.column_stack((AnalogFrames, VideoFrames, SubFrames)), ADCdata)), columns=AnalogNames)
        
        # import numpy as np
        # import matplotlib.pyplot as plt
    ########################## Force Plate Data ###################################
    # Express moments as CoPx, CoPy and Tz, create separate data frames for each plate
    # Only run this if asked for and there is analog data
    
    if GetForceData and GetAnalogData:
        # Make sure there are force plates used
        FP_Used = None
        try:
            FP_Used = ParameterGroup['FORCE_PLATFORM']['USED']['data'][0]
                   
        except:
            # FP_Used is None:
            # messagebox.showinfo('C3D WARNING',f'Force Plate Data: NO FORCE DATA FOUND IN C3D FILE {filenamepath}')
            print('WARNING: force data requested but no force data detected in file')
            GetForceData = False
        
        if GetForceData:
            # Initialize variable for force plate name storage
            # FP_Name = str(FP_Used)
            
            # Determine what format the force plate output was written to the c3d file. Valid types are 1-4.
            FP_Type = ParameterGroup['FORCE_PLATFORM']['TYPE']['data']
                        
            # Read in the location of the origins of the force plates (these are offsets from FP centers, not lab origins)
            # In "Research Methods in Biomechanics", these are referred to as the "Plate Reference System" (PRS) orgin values
            FP_Origin_dims = ParameterGroup['FORCE_PLATFORM']['ORIGIN']['dimensions']
            FP_Origin = np.array(ParameterGroup['FORCE_PLATFORM']['ORIGIN']['data']).reshape(int(FP_Origin_dims[0]),FP_Used)
            
            # Because the AMTI force plate manual has the wrong sense of the origin vector,
            # must verify that the Z value is negative. If not, negate the vector
            FP_Origin[2, :] = -np.abs(FP_Origin[2, :])
            
            # Read in the frame numbers associated with determining "zero" values (start zero, end zero)
            # The FORCE_PLATFORM:ZERO parameter is an array that normally contains two non-zero integer values. 
            # These specify the range of 3D data frame numbers that may be used to provide a baseline for the 
            # force platform measurements. The default array values are is 1,10 although some applications may 
            # specify a range of 0,10 which should be interpreted as a range of 1 to 10 since the C3D file does 
            # not have a 3D frame number 0, the first frame in a C3D file is normally frame 1.
            FP_Zero = ParameterGroup['FORCE_PLATFORM']['ZERO']['data']
                        
            # Determine which analog channels correspond to the force plate output
            FP_Channel_dims = ParameterGroup['FORCE_PLATFORM']['CHANNEL']['dimensions']
            FP_Channel = np.array(ParameterGroup['FORCE_PLATFORM']['CHANNEL']['data']).reshape(int(FP_Channel_dims[1]),int(FP_Channel_dims[0])) - 1
                       
            # Determine if Offset Range is wanted and if so valid, set Offset to True or False
            # Offset is True if all the following conditions are met:
            Offset = ((FP_Zero[0] <= FP_Zero[1]) and 
                      (FP_Zero[0] > 0) and 
                      (FP_Zero[1] > 0) and 
                      (FP_Zero[1] <= NumberOfC3DFrames)) and ForcePlateZero
            # Make output array for force plate data: N force plates, 
            # analog frame rows x 6 columns (Fx, Fy, Fz, CoPx, CoPy, Tz)
            for FP in range(FP_Used):
                # Repackage Analog for each force plate using FP channels, remove 0 channels and offset by the 3 ID columns
                
                # Shifting 3 column indexes to account for the frames columns added to AnalogData above
                FPData = AnalogData.iloc[:,FP_Channel[FP,:] + 3]
                # -------------------------------------------------------------
                # Plot analog trajectories, if troubleshooting
                # for name in AnalogNames:
                #     AnalogData[name].plot(y=name, legend=name)
                #     plt.show()
                
                # First calculate offsets for each channel if specified
                # Do this on the raw data before transformations
                # Fix zero frame if set to 0
                if FP_Zero[0] == 0:
                    FP_Zero[0] = 1
                
                # Check if user wants the baseline offset to be removed.
                # FP_Zero array specifies starting and ending range of frames to zero values
                # Here we are assuming that if user specifies frames 1-10 to zero, but first 
                # frame of data in C3D is say 50, we will still use the first 10 frames: 50-60
                if Offset:
                    # If frame range is valid, remove "DC" noise. eg. Valid: FP_Zero = (1, 10)
                    # If frame range is invalid, leave the signal alone. eg. FP_Zero = (10, 1)
                    # (Vicon specifies this definition of Invalid frame ranges; C3d standard
                    # requires FP_Zero = (0, 0) to prevent baseline removal)
                    
                    # Find offsets and Z-range
                    # Compute range of ADCData to use as zero
                    ADCZeroStart    = int((FP_Zero[0] - 1) * AnalogToVideoRate)
                    ADCZeroEnd      = int(FP_Zero[1] * AnalogToVideoRate)
                    # Dataframe for just zero range
                    ZeroData        = FPData.iloc[ADCZeroStart:ADCZeroEnd, :].dropna()
                    ZeroMean        = ZeroData.mean()
                    
                    # Remove any baseline offset from the force plate data
                    # Having determined the mean and range of the noise, remove it
                    # Column wise substraction of the ZeroMean value 
                    FPData          = FPData.sub(ZeroMean, axis=1)
                
                # Type 1 force plates
                if FP_Type[FP] == 1:
                    # Type 1 force plate output is Fx,Fy,Fz,COP_x,COP_y,Tz
                    # Transformations from transducer outputs have occurred prior to C3D storage so no 
                    # additional manipulations other than possibly flipping reaction signs and zeroing are needed
          
                    # Standardize names
                    FPData.columns  = ['Fx', 'Fy', 'Fz', 'CoPx', 'CoPy', 'Tz']
                    # Force plate data is already transformed
                    FPDataT         = FPData
                    
                # Type 2 force plates
                elif FP_Type[FP] == 2:
                    # FP output data in form(Fx,Fy,Fz,Mx,My,Mz)
                    # For a type 2 force plate, FP_Origin is the vector from the FP origin
                    # to the geometric center of the FP surface in the FP coordinate system
                    # Type 2 plates are made by AMTI, Bertec, ...
                    
                    # Standardize names
                    FPData.columns  = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
                    
                    # Add columns for standardized output
                    # Calculate center of pressure and free moment
                    # Reference "Research Methods in Biomechanics - Forces and Their Measurements" chapter for equations
                    
                    # Center of pressure and free moment
                    Cx              = -(FPData['My'] - FP_Origin[2, FP] * FPData['Fx']) / FPData['Fz'] - FP_Origin[0, FP]
                    CoPx            = Cx.fillna(0) # fill nan values with zero
                    
                    Cy              = (FPData['Mx'] + FP_Origin[2, FP] * FPData['Fy']) / FPData['Fz'] - FP_Origin[1, FP]
                    CoPy            = Cy.fillna(0)
                    
                    freeMoment      = FPData['Mz'] + FPData['Fx'] * (CoPy + FP_Origin[1, FP]) - FPData['Fy'] * (CoPx + FP_Origin[0, FP])
                    Tz              = freeMoment.fillna(0)
                    
                    # Ensure using a copy, not a view of the force plate dataframe
                    # Add calculated data to main force dataframe
                    FPData          = FPData.copy()
                    FPData['CoPx']  = CoPx
                    FPData['CoPy']  = CoPy
                    FPData['Tz']    = Tz
                    
                    # Select columns of standardized output
                    FPDataT = FPData[['Fx', 'Fy', 'Fz', 'CoPx', 'CoPy', 'Tz']]
                    
                
                # Type 3 force plates
                elif FP_Type[FP] == 3:
                    # Kistler plate with 8 channels of output data
                    # Standardize columns names
                    FPData.columns  = ['Fx12', 'Fx34', 'Fy14', 'Fy23', 'Fz1', 'Fz2', 'Fz3', 'Fz4']
                    
                    Fx              = FPData['Fx12'] + FPData['Fx34']
                    Fy              = FPData['Fy14'] + FPData['Fy23']
                    Fz              = FPData['Fz1'] + FPData['Fz2'] + FPData['Fz3'] + FPData['Fz4']
                    
                    Mx              = FP_Origin[1, FP] * (FPData['Fz1'] + FPData['Fz2'] - FPData['Fz3'] - FPData['Fz4']) + FP_Origin[2, FP] * Fy
                    My              = FP_Origin[0, FP] * (-FPData['Fz1'] + FPData['Fz2'] + FPData['Fz3'] - FPData['Fz4']) - FP_Origin[2, FP] * Fx
                    Mz              = FP_Origin[1, FP] * (FPData['Fx34'] - FPData['Fx12']) + FP_Origin[0, FP] * (FPData['Fy14'] - FPData['Fy23'])
                    CoPx            = -My / Fz
                    CoPy            = Mx / Fz
                    Tz              = Mz - Fy * CoPx + Fx * CoPy
                    
                    # Ensure using a copy, not a view of the force plate dataframe
                    # Add calculated data to main force dataframe
                    FPData          = FPData.copy()
                    FPData['Fx']    = Fx
                    FPData['Fy']    = Fy
                    FPData['Fz']    = Fz
                    FPData['CoPx']  = CoPx
                    FPData['CoPy']  = CoPy
                    FPData['Tz']    = Tz
                    
                    # Select columns of standardized output
                    FPDataT = FPData[['Fx', 'Fy', 'Fz', 'CoPx', 'CoPy', 'Tz']]
                
                # Type 4 force plates
                elif FP_Type[FP] == 4:
                    # Same as Type 2 force plate EXCEPT that parameter CAL_MATRIX is provided to
                    # convert from volts to form (Fx,Fy,Fz,Mx,My,Mz)
                                                        
                    # For a type 4 force plate, FP_Origin is the vector from the FP origin
                    # to the geometric center of the FP surface in the FP coordinate system
                            
                    # The calibration matrix, CAL_MATRIX, is used to convert volts to
                    # force and moment units. (ANALOG_SCALE parameter should only have taken
                    # the output signals to volts.)
                    
                    # Stash this GetParameterData in this FP loop even though redundant 
                    # parameter reads will happen if more than one FP with type 4, FP Type 4s will be rare
                    FP_cm = ParameterGroup['FORCE_PLATFORM']['CAL_MATRIX']['data']
                    
                    # Dimensions of FP_CalMatrix are (6,6,FP_Used)
                    # Convert to matrix and multiply by calibration matrix transpose (sensitivity matrix)
                    fpdims = ParameterGroup['FORCE_PLATFORM']['CAL_MATRIX']['dimensions']
                    FP_CM_array = np.array(FP_cm)
                    FPdims = [int(dim) for dim in fpdims]
                    FP_CM = FP_CM_array.reshape(tuple(FPdims))
                    FPmat = np.dot(FPData.to_numpy(), FP_CM[:, :, FP].T)
                    
                    # Transpose the force plate matrix and extract the rows of force and moment data
                    Fx, Fy, Fz, Mx, My, Mz = FPmat.T
                    
                    # Calculate center of pressure and free moment
                    Cx          = -(My - FP_Origin[2, FP] * Fx) / Fz - FP_Origin[0, FP]
                    CoPx        = np.nan_to_num(Cx, nan=0.0) # fill nan values with zero
                    Cy          = (Mx + FP_Origin[2, FP] * Fy) / Fz - FP_Origin[1, FP]
                    CoPy        = np.nan_to_num(Cy, nan=0.0)
                    freeMoment  = Mz + Fx * (CoPy + FP_Origin[1, FP]) - Fy * (CoPx + FP_Origin[0, FP])
                    Tz          = np.nan_to_num(freeMoment, nan=0.0)
                    
                    
                    # Load current forces, center of pressure, and free moment into FPDataT array
                    FPDataT = pd.DataFrame({'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'CoPx': CoPx, 'CoPy': CoPy, 'Tz': Tz})
                
                # Regardless of DC offset requested, zero all data below calculated Fz threshold
                # Calculate the range of the vertical force signal--"peak-to-peak signal"
                FzSignalRange = abs(FPDataT['Fz'].max() - FPDataT['Fz'].min())
                
                # Zero any data less than 0.5% of range
                # In case force plate has no loading and therefore no real signal range, 
                # set an artificial threshold of 1N
                if ZeroForceBaseline:
                    FzCutOff = max(0.005 * FzSignalRange, 1)
                
                # Calculate mean vertical force to adjust force directions if flipped (Reaction Force if mean(Fz) < 0)
                # Check if the scaled FP output is a reaction force or an action force, set ReactionFactor accordingly
                
                # The np.sign function in NumPy returns the sign of each element in an array. Specifically, it works as follows:
                # If an element is positive, np.sign returns 1.
                # If an element is negative, np.sign returns -1.
                # If an element is zero, np.sign returns 0.
                ReactionFactor = -np.sign(FPDataT['Fz'].mean())
                # Augment FP data to include analog frames, video frames, and sub-frames
                # These were already made in ADCData so just copy
                # Also filter to First and Last Returned frames, flip components if reaction force was wrong sensed, and
                # apply zero threshold offsets
                
                # FPDataT[['Fx','Fy','Fz','Tz']] = FPDataT[['Fx','Fy','Fz','Tz']].apply(lambda x: x * ReactionFactor)
                analogSub_df = AnalogData[['AnalogFrames','VideoFrames','SubFrames']]
                FPDataT = pd.concat([analogSub_df, FPDataT], axis=1)
                
                filtered_df = FPDataT.loc[
                    (FPDataT['VideoFrames'] >= FirstReturnFrame) & (FPDataT['VideoFrames'] <= LastReturnFrame)
                    ].copy()
                
                # Perform the assignment in a vectorized manner
                filtered_df['Fx'] *= ReactionFactor
                filtered_df['Fy'] *= ReactionFactor
                filtered_df['Fz'] *= ReactionFactor
                filtered_df['Tz'] *= ReactionFactor
                
                # Zero each row if less than the cutoff threshold
                if ZeroForceBaseline:
                    filtered_df.loc[abs(filtered_df['Fz']) < FzCutOff, ['Fx', 'Fy', 'Fz', 'Tz']] = 0
                
                # update original dataframe
                FPDataT.update(filtered_df)
                    
                # Filter the DataFrame for the Frame between FirstReturnFrame and LastReturnFrame
                FPDataT = FPDataT[(FPDataT['VideoFrames'] >= FirstReturnFrame) & (FPDataT['VideoFrames'] <= LastReturnFrame)]
                
                # If shorter format requested then filter to just video frames and remove SubFrames column
                if ForceDataFrames == 'video':
                    FPDataT = FPDataT[['AnalogFrames','VideoFrames', 'Fx','Fy','Fz','Tz']]
                
                # Store each set of FP data in one list
                if FP == 0:
                    ForcePlate_Name = []
                    ForcePlate_Name.append('ForcePlate1')
                    ForceDataFrames  = [FPDataT]
                else:
                    ForcePlate_Name.append(f'ForcePlate{FP + 1}')
                    ForceDataFrames.append(FPDataT)
            
            # ForceData = {}
            for num in range(len(ForcePlate_Name)):
                ForceData.update({ForcePlate_Name[num]: ForceDataFrames[num]})
        
        # Cleanup
        # del FPData, FPDataT, VideoMatrix
    
    # Now filter Analog Data for First and Last Return Frame
    # Filter the DataFrame for the Frame between FirstReturnFrame and LastReturnFrame
    if GetAnalogData:
        AnalogData = AnalogData[(AnalogData['VideoFrames'] >= FirstReturnFrame) & (AnalogData['VideoFrames'] <= LastReturnFrame)]
    
    ########################## Gait Cycle Events ##############################
    # ---- Gait Cycle Events ----
    # Repackage parameter events into more user-friendly vectors
    # Check if any events exist before trying to read individual parameters
    
    ######################### Get number of gait events in file ###############
    if GetGaitCycleEvents:
        try:
            GaitCycleEventsUsed = ParameterGroup['EVENT']['USED']['data'][0]
        except:
            GaitCycleEventsUsed = 0
            # warning('ReadC3D:Events: No events found in EVENT group \n')
            # Report EVENT group doesn't exist in returned variable
            GaitCycleEvents = 'No Gait Cycle Events from EVENT:USED'
        
        if GaitCycleEventsUsed != 0:
            # Calculate Gait Cycle Events
            # Read other parameters into temp variables
            
            # Get list of event times       
            # Each time is stored as minutes in one field, second in next field
            # Need to assemble the event times by adding two fields together;
            EventTimes = ParameterGroup['EVENT']['TIMES']['data']
             
            #######################################################################
            # Get list of event descriptions
            EDlist = []
            EDsentence = []
            EventDescriptions = []
            for key in ParameterGroup['EVENT'].keys():
                if 'DESCRIPTIONS' in key:
                        EDlist.append(ParameterGroup['EVENT'][key]['data'])
            
            # Event descriptions are extracted as a single string with 
            # each sentence separated by numerous spaces, the sentences need to be 
            # extracted with a delimeter of two or more spaces - avoiding extracting
            # each word individually
            for idx in range(len(EDlist)):
                EDsentence.extend(EDlist[idx].split('  '))
                
            # Spaces will be extracted as individual list elements, therefore these 
            # must be removed manually
            EventDescriptions = [sentence.strip() for sentence in EDsentence if sentence != ""]
            #######################################################################
            # Get list of event labels
            LBlist = []
            LBsentence = []
            EventNames = []
            for key in ParameterGroup['EVENT'].keys():
                if 'LABELS' in key:
                        LBlist.append(ParameterGroup['EVENT'][key]['data'])
            
            # Same process as above
            for idx in range(len(LBlist)):
                LBsentence.extend(LBlist[idx].split('  '))
            
            EventNames = [label.strip() for label in LBsentence if label != ""]
            #######################################################################
            # Get list of event contexts
            CTlist = []
            CTsentence = []
            EventContexts = []
            for key in ParameterGroup['EVENT'].keys():
                if 'CONTEXTS' in key:
                        CTlist.append(ParameterGroup['EVENT'][key]['data'])
            
            # Same process as above
            for idx in range(len(CTlist)):
                CTsentence.extend(CTlist[idx].split('  '))
            
            EventContexts = [context.strip() for context in CTsentence if context != ""]
            #######################################################################
            # Get list of subjects
            SJlist = []
            SJsentence = []
            EventSubjects = []
            for key in ParameterGroup['EVENT'].keys():
                if 'SUBJECTS' in key:
                        SJlist.append(ParameterGroup['EVENT'][key]['data'])
            
            # Same process as above
            for idx in range(len(SJlist)):
                SJsentence.extend(SJlist[idx].split('  '))
            
            EventSubjects = [subject for subject in SJsentence if subject != ""]
            #######################################################################
            
            # Fix times and compress character labels into string arrays
            # Each time is stored as minutes in one field, second in next field
            # Need to assemble the event times by adding two fields together;
            # The first number and every other is the minute compenent of each event
            # extract the minute component and convert to seconds
            
            EventTime               = [0] * GaitCycleEventsUsed # pre-allocated list of zeros
            for eventIDX in range(0,int(len(EventTimes)/2)):
                EvMin               = 60 * EventTimes[eventIDX*2]
                EvSec               = EventTimes[eventIDX*2 + 1]
                EventTime[eventIDX] = round(EvMin+EvSec,5)
            
            # length of EventTimes list needs to be the length of GaitCycleEventsUsed
            EventTimes              = EventTime
    
            # Also store Event Frames
            # Need to add one frame interval as time = 0 is frame = 1
            EventFrames = [int(np.round(int(VideoFrameRate) * time)) + 1 for time in EventTimes]
            
            # Ensure all lists are of the same length
            # Find the length of the longest list
            lists = [EventContexts, EventNames, EventTimes, EventFrames, EventDescriptions, EventSubjects]
            max_length = max(len(lst) for lst in lists)
            
            # Pad the shorter lists with None (or any other dummy variable you prefer)
            padded_lists = [lst + [None] * (max_length - len(lst)) for lst in lists]
            
            GcEvents = {
                "Contexts":             padded_lists[0], # EventContexts
                "Labels":               padded_lists[1], # EventNames
                "Times":                padded_lists[2], # EventTimes
                "Frames":               padded_lists[3], # EventFrames
                "Descriptions":         padded_lists[4], # EventDescriptions
                "Subjects":             padded_lists[5]  # EventSubjects
                }
            
            GaitCycleEvents = pd.DataFrame(GcEvents).sort_values(by='Times')  
       
    MiscData = {
        'Camera Info':              CameraInfo,
        'Data Format':              DataFormat,
        'Residual Error':           ResidualError,
        'Video Byte Length':        VideoByteLength,
        'Analog Sample Length':     AnalogSampleLength,
        'Header Event Switches':    HeaderEventSwitches,
        'Header Event Labels':      HeaderEventLabels,
        }
    
    Header = {
        'C3D_File_Name':            C3DFileName,
        'First_Parameter_Record':   NrecordFirstParameterblock[0],
        'Number_of_Trajectories':   NumberOfMarkers,
        'Analog_Channels':          NanalogChannels[0],
        'First_Frame':              FirstC3DFrameNumber,
        'Last_Frame':               LastC3DFrameNumber,
        'Video_Sampling_Rate':      VideoFrameRate,
        'Analog_Sampling_Rate':     AnalogFrameRate[0],
        'Scale_Factor':             VideoScaleFactor,
        'Start_Record_Num':         FirstDataRecordNumber[0],
        'Max_Interpolation_Gap':    MaxInterpolationGap[0],
        'C3D_File_Format':          filetype,
        'C3D_Data_Format':          DataFormat,
        'Number_of_Events':         Nevents
        }
    
    ############################## FINISHING UP ###################################
    finish_time = datetime.now()
    extraction_duration = finish_time - start_time
    print('c3d file extraction duration: {}'.format(extraction_duration))
       
    C3D_dict = {}
    
    # By default, return parameter group, header etc.
    C3D_dict.update({
            'Parameter Group':      ParameterGroup,
            'Header':               Header,
            'MiscData':             MiscData,
            'Extraction Duration':  extraction_duration,
            'filetype':             filetype,
            'C3D File Name':        C3DFileName
            })
    
    if GetVideoData:
        C3D_dict.update({
            'Marker Data':          MarkerData
            })
        
    if GetAnalogData:
        C3D_dict.update({
            'Analog Data':          AnalogData
            })
        
    if GetForceData:
        C3D_dict.update({
            'Force Data':          ForceData
            })
    
    if GetGaitCycleEvents:
        C3D_dict.update({
            'Gait Cycle Events':   GaitCycleEvents
            }) 
    
    return C3D_dict

if __name__ == "__main__":
    get_C3Ddata()
    Py3_DECbytes2PCbytes.DEC2PC()
    ReadC3DParameters.GetParameters()
    
    
    