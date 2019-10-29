# import the library
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from collections import Counter
from statistics import mean
import time


# define the task1 function
def Task1(img_pixels,task1_img):
    #get the original image shape value
    img_shape = img.shape
    # max x coordinate value
    x_max = img_shape[0]
    # max y coordinate value
    y_max = img_shape[1]
    
    # double for loop to retrieve all the pixels in the image
    for x in range (0,x_max):
        for y in range(0,y_max):
            
            # get the original RGB band value
            px = img_pixels[x,y]
            # get the new band value through ths equation provided
            new_band = 0.299*px[0]+0.587*px[1]+0.114*px[2]
            # swap the original band value with this new_band
            # implement it in a copied original image matrix in order to avoid changing the origianl matrix
            task1_img[x,y]=new_band
    # return the updated image matrix
    return task1_img

# define a sub matrix to get the neighbours around one point/pixel
# x, y is the coordinate
# window_size is the neighbourhood size that we need to define
# the copied image matrix from original image
def get_most_freq_neighbour(x,y,window_size,image_matrix):
    
    # get the winsow_size matriax
    s = (window_size-1)//2
    
    # avoid out of index
    if y-s < 0:
        row_s = 0
    else:
        row_s = y-s
    if x-s < 0:
        col_s = 0
    else:
        col_s = x-s
    # get the subbmatrix based on current location
    sub_matrix = image_matrix[row_s:y+s+1 ,col_s:x+s+1]
    # change 3D to 1D
    oneD_matrix = sub_matrix.ravel()
    # cuonter the elements
    ele_counts = Counter(oneD_matrix)
    # get the most frequent one element
    most_freq = ele_counts.most_common(1)[0][0]
    # return the most frequent pixel value
    return most_freq

# define a function to deal with task2
# get_most_freq_neighbour function is used here
def Task2(window_size,task1_answer):
    # get a copied image matrix of task1
    task1_img = task1_answer
    image_matrix=np.copy(task1_img)
    # another copied image matrix of task1
    task2_matrix = np.copy(task1_img)
    # get the shape of matrix and (x,y)
    shape2 = task2_matrix.shape
    y2_max = shape2[0]
    x2_max = shape2[1]
    # retrieve task2_matrix
    for x in range (0,x2_max):
        for y in range(0,y2_max):
            # get the frequent neighbour
            most_freq_neigh = get_most_freq_neighbour(x,y,window_size,image_matrix)
            # swap the current pixel value with the most frequency neighbour
            task2_matrix[y,x] = most_freq_neigh
    # return task2_matrix
    return task2_matrix



# get a list that contains the coordinators of the point whch has the same value
# (x,y) is the coordinate
def get_same_value_list(x,y,window_size,image_J):
    
    # get the value of current point
    current_value = image_J[y,x][0]
    # get the winsow_size matriax
    s = (window_size-1)//2
    # get the subbmatrix based on current location
    if y-s < 0:
        row_s = 0
    else:
        row_s = y-s
    if x-s < 0:
        col_s = 0
    else:
        col_s = x-s
    # create the list used to store coordinate
    common_list = []
    # use two for loops to retrieve all the points
    for yy in range (row_s, y+s+1):
        for xx in range(col_s,x+s+1):
            # avoid out of index
            if yy >= image_J.shape[0]:
                yy = image_J.shape[0]-1
            if xx >= image_J.shape[1]:
                xx = image_J.shape[1]-1
            # if neighbour's value equal to current value
            if  image_J[yy,xx][0] == current_value :
                # append to common list
                common_list.append((yy,xx))
    # return this common list
    # the common list will be used in the calculate_averge_pixel function
    return common_list

# this function is used to calculate the avergae RGB band of one piexl point
# image_B is the ori
def calculate_averge_pixel(image_B,common_list):
    
    # create a list used to store RGB band value
    avg_outcome=[]
    # create RGB band list used to store the neighbour's RGB value
    R_average_list = []
    G_average_list = []
    B_average_list = []
    
    # for loop to append RGB band value to respective list
    for (y,x) in common_list:
        R_average_list.append(image_B[y,x][0])
        G_average_list.append(image_B[y,x][1])
        B_average_list.append(image_B[y,x][2])
    
    #average_list=np.array(average_list)
    # if R_average_list is not empty
    if R_average_list:
        # get the mean of the R band list
        R_avg_value = mean(R_average_list)
        # append the value to avg_outcome
        avg_outcome.append(R_avg_value)
    # if G_average_list is not empty
    if G_average_list:
        # get the mean of the G band list
        G_avg_value = mean(G_average_list)
        # append the value to avg_outcome
        avg_outcome.append(G_avg_value)
    # if B_average_list is not empty
    if B_average_list:
        # get the mean of the B band list
        B_avg_value = mean(B_average_list)
        # append the value to avg_outcome
        avg_outcome.append(B_avg_value)
    
    return avg_outcome

# combine get_same_value_list and calculate_averge_pixel functions
# swap all the original image pixel values
def Task3(task2_answer,img,w_size):
    # get the copied image matrix from original image
    image_B = np.copy(img)
    # copy task2 outcome
    image_J = np.copy(task2_answer)
    image_T3 = np.copy(img)
    # get the image shape and it's max coordinate values
    shape2 = image_J.shape
    y2_max = shape2[0] -1
    x2_max = shape2[1] - 1
    # for loop, retrieve the point in task2 outcome and original image
    for x in range (0,x2_max):
        for y in range(0,y2_max):
            # call get_same_value_list function, get the common list
            common_list = get_same_value_list(x,y,w_size,image_J)
            # call function, calculate the avergae RGB value
            avg_list = calculate_averge_pixel(image_B,common_list)
            R_avg_value = avg_list[0]
            G_avg_value = avg_list[1]
            B_avg_value = avg_list[2]
            #swap the RGB value in copied original image matrix
            image_T3[y,x][0] = R_avg_value
            image_T3[y,x][1] = G_avg_value
            image_T3[y,x][2] = B_avg_value
    # return matrix
    return image_T3



# get all the image files in the directory
for file in glob.glob("*.jpg"):
    # read the img as a matrix
    img = cv2.imread(file,1)
    #cv2.imshow("image",img)
    # get the matrix shape
    shape = img.shape
    print(f"Loading the image:{file}, Shape:{shape} \n")
    print (f"Start Time:{time.ctime()}")
    
    # copy the image matrix
    img_pixels = np.copy(img)
    task1_img = np.copy(img)
    #######################################
    ## Call task1 function
    task1_answer = Task1(img_pixels,task1_img)
    print(task1_answer)
    # write down the task1 image and store in local directory
    # define the name
    t1_file_name = f"task1_{file}.jpg"
    cv2.imwrite(t1_file_name,task1_answer)
    print("Task1 finished")
    
    #######################################
    ## TASK2
    # try three different window size
    for window_size in [3,9,21]:
        # call task2 function
        task2_answer = Task2(window_size,task1_answer)
        # define the filename
        t2_file_name = f"task2_{file}_{window_size}X{window_size}.jpg"
        # write down the image of task2
        cv2.imwrite(t2_file_name,task2_answer)
        print("Task2 finished")
        #######################################
        ## TASK3
        # call the task3 function
        task3_answer = Task3(task2_answer,img,window_size)
        # define the filename
        t3_file_name = f"task3_{file}_{window_size}X{window_size}.jpg"
        # write down the image of task3
        cv2.imwrite(t3_file_name,task3_answer)
        print("Task3 finished")
        print (f"Start Time:{time.ctime()}")
