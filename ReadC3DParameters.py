# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:20:42 2024

@author: Vicon-OEM
"""
import struct
import numpy as np
import Py3_DECbytes2PCbytes


Parameters = {}
# Initialize the ParameterGroup dictionary
ParameterGroup = {}
GN_list = ['Not Specified']*128
ParamNumber = np.zeros(128, dtype=int)

def GetParameters(filenamepath, file, NrecordFirstParameterblock, FirstParameterByteNumber, proctype):
    # print(filenamepath)

    # open c3d file for read access in binary format
    with open(filenamepath, 'rb') as file:
    
        # NrecordFirstParameterblock determines starting point for parameter records
        # Parameter block starts at byte 512
        # proctype: 1(INTEL-PC); 2(DEC-VAX); 3(MIPS-SUN/SGI)
        
        # ignore first two bytes
        file.seek(FirstParameterByteNumber + 2, 0)
        
        # Below code is interpreted from the original matlab readc3d code
        # NparameterRecords = np.fromfile(file, dtype='int8', count=1)
        
        # Interpreted from the R readc3d code - line 123 'Parameter_Blocks
        # This reads 1 byte from the file and interprets it as an unsigned 8-bit integer (`np.uint8`).
        # `file.read(1)` reads the byte directly from the file object.
        # 3rd byte (byte 514) is the number of 512 byte blocks that the parameter section uses 
        NparameterRecords = np.frombuffer(file.read(1), dtype=np.uint8)[0]
        
        # Some files may not record the number of parameter blocks
        # Calculates number of parameter blocks using data start record
        if NparameterRecords == 0:
            # store current position
            Current_Position = file.tell()
            
            # Skip to byte 17 (word 9 = bytes 16, and 17)
            file.seek(16,0)
            
            # Word 9: Starting record number for 3D point and analog data, unsigned 2 byte integer
            # **Note: matlab code uses 'int8' dtype instead of unsigned integer  'uint8'
            # FirstDataRecordNumber = np.fromfile(file, dtype=np.uint8, count=2)
            data = np.frombuffer(file.read(2), dtype=np.uint16)
            FirstDataRecordNumber = data[0]
            
            NparameterRecords = FirstDataRecordNumber - NrecordFirstParameterblock
            
            # Set position in file back to previous position
            file.seek(Current_Position,0)
            
        # Read entire parameter section
        file.seek(FirstParameterByteNumber,0)
        Nbytes = 512*NparameterRecords
        
        # in R, the code to read the pb data extracts the data as a hexidecimal
        # representation of each byte
        # In Python, the data is extracted as a string with escape sequences
        # which need to be converted to hexidecimal - this is used for 
        # interpretation during coding and can be switched back to bytes object
        
        # extract data as a string with escape sequences
        bytesObject = file.read(Nbytes) 
        # convert to hexidecimal
        pb = [f'{byte:02x}' for byte in bytesObject] 
        
          # From the C3D User Manual (C3D.org):
          
          # The parameters are stored starting at byte 5 of the parameter section. 
          # The parameters are stored in random order providing flexibility when 
          # parameters need to be edited, deleted or added. 
          # Each parameter has a data type, optional dimensions, a name, a description, 
          # and belongs to a group. Each group defined in the parameter section also 
          # has a name and a description.
          
          # Read the GROUP parameters
          # Byte  1:                       # of characters n in group name
          # Byte  2:                       Group ID number (-1 to -127, always negative)
          # Bytes 3:3 + n - 1:             Group name character(1: n = #characters)
          # Byte  3 + n:                   Signed integer offset in bytes pointing to start of next group/parameter
          # Byte  3 + n + 2:               # characters m in the Group description
          # Byte  3 + n + 3:               Group description
          
          # Read the PARAMETERS
          # Byte 1:                       # of characters n in parameter name
          # Byte 2:                       Group ID number (1 to 127, always positive)
          # Byte 3:3 + n:                 Parameter name character(1: n = #characters)
          # Byte 3 + n:                   Signed integer offset in bytes pointing to start of next group/parameter
          # Byte 3 + n + 2:               Length of each data element
          #                                 -1 for character data
          #                                  1 for byte data
          #                                  2 for integer data
          #                                  4 for floating point data
          # Byte 3 + n + 3:               Number of dimensions d of the parameter, 0 if scalar
          # Byte 3 + n + 4:               Parameter dimensions, length d bytes
          # Byte 3 + n + 4 + d:           Parameter data length t bytes
          # Byte 3 + n + 4 + d + t:       # of characters m in parameter description
          # Byte 3 + n + 4 + d + t + 1:   Parameter description length m
          
          # There is no count stored for the number of parameters in each group and all 
          # group and parameter records can appear in any order. This means that it is 
          # permissible for a parameter to appear in the parameter section before the
          # group information and software accessing a C3D file should be prepared to deal 
          # with this situation.
          
          # Parameters are connected to groups by use of the group ID number. Group 
          # records have unique ID numbers within the file, which are stored as a negative 
          # value in byte 2. All parameters belonging to a group will store the same ID as 
          # a positive value, also in byte 2.
          
          # Always use the absolute value of the first byte to determine the length of the 
          # parameter name. This value may be set to a negative value to indicate that the 
          # data item has been marked as locked and should not be edited.
          # ............................................................................
          
          # Number of Groups and Parameters are not known, so have to search through all blocks
    
        byte = 4 # index 5, from R
        PR = 0
        GroupNumber = 1
        
        while GroupNumber != 0:
            
            # Byte 1: # characters in GROUP or PARAMETER name
            # This is actually signed because a negative number indicates "LOCKED"
            # as in do not change (value from 1-127)
            byte_val = pb[byte] # indexing the hexidecimal list
            byte_value = bytes.fromhex(''.join(byte_val)) # conversion back to bytes
            
            # NameLength = abs(np.int8(struct.unpack('b', byte_value)[0]))
            NameLength = abs(np.frombuffer(byte_value, dtype=np.int8)[0])
            # firt time through, the value of NameLength should be equal to 5
            # print(f'Name Length is: {NameLength}')
            
            if NameLength == 0:
                GroupNumber = 0
                GN = 0
            else:
                # Byte 2: # Group ID number, also signed 
                # Negative if group, positive if parameter
                byte = byte + 1
                
                # Group Number for indexing
                byte_value = bytes.fromhex(''.join(pb[byte]))
                GN = np.int8(struct.unpack('b', byte_value)[0])
            
            # print(f'Name length is: {NameLength}')
            # print(f'Group Number is: {GN}')
            # Group data is GN < 0
            if GN < 0:
                # Is Group if GN < 0
                
                # Read the parameter name of length 'NameLength' store in array for use in ParameterGroup dictionary
                # Extract name as hexidecimal 
                byte = byte + 1
                namebyteval_1 = pb[byte:byte + NameLength]
                GroupName = bytes.fromhex(''.join(namebyteval_1)).decode('utf-8')
                GN_list[abs(GN)-1] = GroupName
                ParameterGroup[GroupName] = {}
                
                # code to get 
                # 2 Bytes following name are the byte number of next group
                byte = byte + NameLength
                namebyteval_2 = pb[byte:byte + 2]
                val_2 = bytes.fromhex(''.join(namebyteval_2))
                offset2Next = np.frombuffer(val_2, dtype=np.int16)[0]
                
                if offset2Next == 0:
                    # The last parameter in the parameter section always has a pointer value 
                    # of 0x0000h to indicate that there are no more parameters
                    GroupNumber = 0
                    break
                
                NextParamByte = byte + offset2Next
                # Next byte is # characters in GROUP description, unsigned integer
                byte = byte + 1
                
                ######### ********** NOTE ************* ###########################
                # DescLength = int(pb[byte]) # Currently unused
                # R code has a line for assinging the group description that is unused
                # GroupDesc[GroupNumber, pn] <- rawToChar(pb[(byte):(byte + DescLength - 1)])
                ###################################################################
                
                # Advance to next group/parameter by number of bytes + offset
                byte = NextParamByte
            else:
                # Update group number list to reference in 
                gName = GN_list[GN-1]
                
                # print(f'Group Name List is: {GN_list}')
                # print(f'Group Name is: {gName}; Group Number is: {GN}')
                # Is parameter if GN >= 0
                byte = byte + 1
                namebyteval_3 = pb[byte:byte + NameLength]
                val_3 = bytes.fromhex(''.join(namebyteval_3))
                ParamName = bytes(val_3).decode('utf-8')
                
                # Keep count of the number of parameters in each group, use PN as counter 
                PN = ParamNumber[GN] + 1
                ParamNumber[GN] = PN
                
                # 2 bytes following name are the byte number of next group
                byte = byte + NameLength
                namebyteval_4 = pb[byte:byte + 2]
                val_4 = bytes.fromhex(''.join(namebyteval_4))
                offset2Next = np.frombuffer(val_4, dtype=np.int16)[0]
                                
                # This offset points to start of next parameter from this byte
                NextParamByte = byte + offset2Next
                
                if offset2Next == 0:
                    break
                
                # Update parameters dictionary after validating parameter exists
                PR += 1
                
                # error handling for when a group name does not exist
                try:
                    ParameterGroup[gName].update({ParamName: {}})
                except:
                    ParameterGroup['Not Specified'] = {}
                    ParameterGroup[gName].update({ParamName: {}})
                
                # print(f'Group Name List is: {GN_list}')
                # print(f'Group Name is: {gName}; Group Number is: {GN}')
                # print(f'ParameterName is: {ParamName}; Parameter Number is: {PN}\n\n')
                # update the ParamName dict with parameter number
                
                ParameterGroup[gName][ParamName].update({'Parameter_Number': PN})            
                
                # Type/length of data element
                byte = byte + 2
                
                try:
                    # if pb[byte] is a string which can be converted to signed integer
                    PT = np.int8(pb[byte])
                except:
                    # if pb[byte] is a string which needs to be converted to a bytes
                    # object first, then to a signed integer
                    val_5 = bytes.fromhex(''.join(pb[byte]))
                    PT = np.frombuffer(val_5, dtype=np.int8)[0]    
                
                # type of data: -1=char/1=byte/2=integer*2/4=real*4
                if PT not in [-1, 1, 2, 4]:
                    print(f'ReadC3DParameters - Main function: Type not read at parameter byte: {byte}')
                    return ParameterGroup
                else:
                    datatype = PT 
                    ParameterGroup[gName][ParamName].update({'datatype': datatype}) 
                    
                # number of dimensions in parameter
                # Parameter dimensions, length is given by numdim
                # First get the number of values by the dimensions
                # One dimensional
                # Data Length byte gives value in bytes (variables * byte size), not # variables
                byte = byte + 1
                numdim = np.int8(pb[byte])
                ParameterGroup[gName][ParamName].update({'NumDimensions': numdim})
                                                                                           
                byte = byte + 1
                
                # Keep count of number of values stored in each parameter
                NumVal = 0
                # if scaler - don't read next byte
                if numdim == 0:
                    # from line 332 R code ReadC3DParameters.R
                    # ParameterGroup[GroupName][ParamName].update({'dimensions': 0})
                    ParameterGroup[gName][ParamName].update({'dimensions': 0})
                      
                    NumVal = 1
                    DataLength = int(abs(datatype))
                    # print(f'Data length scalar: {DataLength}')
                    # print(f'NumVal from scalar: {NumVal}')
                
                # if vector
                elif numdim == 1:                    
                    ParameterGroup[gName][ParamName].update({'dimensions': 1})
                        
                    val_new = bytes.fromhex(''.join(pb[byte]))
                    NumVal = np.frombuffer(val_new, dtype=np.uint8)[0]
                    DataLength = int(NumVal * abs(int(datatype)))
                    # if DataLength == -116:
                    #     print('stopping here')
                    byte = byte + 1
                    # print(f'Data length vector: {DataLength}')
                    # print(f'NumVal from vector: {NumVal}')
                
                # if 3D or more
                else:
                    NumVal = 1
                    DimSize = np.zeros(numdim)
                    # DimSize.append(int(numdim))
                    for dim in range(0,numdim):
                        val_new = bytes.fromhex(''.join(pb[byte]))
                        DimSize[dim] = int(np.frombuffer(val_new, dtype=np.uint8)[0])
                        
                        NumVal = int(NumVal * DimSize[dim])
                        byte = byte + 1
                        # print(f'NumVal from 3D: {NumVal}')
                    
                    DataLength = int(NumVal * abs(datatype))
                    # print(f'Data length 3D: {DataLength}')

                    ParameterGroup[gName][ParamName].update({'dimensions': DimSize})
                    
                ######################### re: DataLength parameter ############
                """ 
                The DataLength parameter specifies the length of the data to be
                extracted from the c3d file by the number elements to be read
                from the file times the number of bytes per element. 
                
                If the data is 2 elements long and 2 bytes per element - as 
                indicated by the data type (i.e. 2-bytes per element is 
                dtype=np.int16). Where elements are 2-bytes each, DataLength
                must be divided by 2 where dtype=np.int16. If the data type is
                1-byte per element (i.e. dtype=np.int8), DataLength remains
                unchanged.                
                
                """
                # Read all parameter data into a vector regardless of dimensions, then re-dimension later
                Endbyte = byte + DataLength # had to ditch the "-1" for python indexing
                index_bytes = pb[byte:Endbyte]
                
                # If characters
                if datatype == -1:
                    # store strings exactly as in C3D, deal with white space and dimensions as needed
                    # convert to character string
                    val_new = bytes.fromhex(''.join(index_bytes))
                    char_string = bytes(val_new).decode('utf-8', errors='ignore')  # Ignore invalid characters
                    data = char_string.strip()
                    
                # If bytes
                elif datatype == 1:
                    # DataLength is preserved because dtype is np.int8 indicating index_bytes is read with each element 1-byte in size
                    val_new = bytes.fromhex(''.join(index_bytes))
                    data = list(np.frombuffer(val_new, dtype=np.uint8, count=DataLength))
                    
                # If integer
                elif datatype == 2:
                    val_6 = bytes.fromhex(''.join(index_bytes))
                    # According the the C3D manual 16-bit data samples must be 
                    # stored as signed integer numbers (the default) unless the 
                    # optional parameter ANALOG:FORMAT is set to UNSIGNED.
                    # try:
                    #     Analog_format = ParameterGroup['ANALOG']['FORMAT']['data']
                    
                    # except:
                    #     Analog_format ='Not Specified'
                        
                    # if Analog_format == 'UNSIGNED':
                    #     # DataLength divided by 2 because dtype is np.int16 indicating index_bytes is read with each element 2-bytes in size
                    #     data = list(np.frombuffer(val_6, dtype=np.uint16, count=int(DataLength/2)))
                    # else:
                    data = list(np.frombuffer(val_6, dtype=np.int16, count=int(DataLength/2)))
                   
                    
                # If float/real
                elif datatype == 4:
                    # DEC conversion, need to take one at time dimension temp variable
                    if proctype == 'DEC':
                        paramdat = np.zeros(NumVal)
                        for i in range(0,NumVal):
                            index_bytes = pb[byte:byte + 4]
                            val_new = bytes.fromhex(''.join(index_bytes))
                            paramdat[i] = Py3_DECbytes2PCbytes.DEC2PC(val_new)
                            byte = byte + 4
                        
                        data = list(paramdat)
                    
                    # PC Real
                    else: 
                        val_7 = bytes.fromhex(''.join(index_bytes))
                        data = list(np.frombuffer(val_7, dtype=np.float32, count=NumVal))
                        
                # else we're in trouble haha
                else:
                    raise ValueError('ReadC3DParameters - Main function: Parameter type not recognized')
                
                # data value output from one of each option above, starting at previous if statement
                ParameterGroup[gName][ParamName].update({'data': data})
                
                ##################### below code from R - not used ####################
                # Advance to next by number of bytes + offset    
                # byte = Endbyte + 1
                # Number of characters in the parameter description, unsigned
                # DescLength = int(pb[byte])
                # Do we care about these or just skip? I vote for skip ...
                # ParamDesc <- rawToChar(pb[(byte):(byte + DescLength - 1)])
                #######################################################################
                
                # Advance to next group/parameter by number of bytes + offset
                byte = NextParamByte
                # print(f'PG keys: {ParameterGroup.keys()}')
            
        # finish while-loop
        return ParameterGroup

if __name__ == "__main__":
    GetParameters()
    Py3_DECbytes2PCbytes.DEC2PC()
    