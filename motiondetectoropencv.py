from datetime import datetime
import cv2
import pandas as pd
import time

first_frame = None
status_list = [None,None]
times = []
#Dataframe to store the time values during which object detection and movement appears
df = pd.DataFrame(columns=['Start','End'])

cam = cv2.VideoCapture("C:/Users/mojta/Desktop/pred.mp4")

#Iterate through frames and display the window
while cam.isOpened():
    check, frame = cam.read()

    #Status at beginning of the recording is zero as the object is not visisble
    status = 0

    #Converting each frame into gray scale image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Convert grayscale image to GaussianBlur
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    #This is used to store the first image/frame of the video
    if first_frame is None:
        first_frame = gray
        continue

    #Calculates the difference between the first frame and another frames
    delta_frame = cv2.absdiff(first_frame,gray)

    #Giving a threshold value, such that it will convert the difference value with less than 30 to black
    #If it is greater than 30, then it will convert those pixels to white
    _,thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=3)

    #Defining the contour area i.e., borders
    cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Removes noises and shadows, i.e., it will keep only that part white, which has area greater than 10000 pixels
    for cont in cnts:
        if cv2.contourArea(cont) < 20000:
            continue
        #Change in status when the object is being detected
        status = 1
        #creates a rectangular box around the object in the frame
        (x, y, w, h) = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    #List of status for every frame
    status_list.append(status)
    status_list = status_list[-2:]

    #Record datetime in a list when change occurs
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    #Opening all types of frames/images
    cv2.imshow("Grey Scale",gray)
    cv2.imshow("Delta", delta_frame)
    cv2.imshow("Threshold", thresh_delta)
    cv2.imshow("Colored frame",frame)

    last_frame_num = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    #Generate a new frame after every 1 millisecond
    key = cv2.waitKey(1)
    #If entered 'q' on keyboard, breaks out of loop, and window gets destroyed
    if key == ord('q'):
        if status==1:
            times.append(datetime.now())
        break

#Store time values in a Dataframe
for i in range(0,len(times),2):
    df = df.append({'Start':times[i],'End':times[i+1]}, ignore_index=True)

#Write the dataframe to a CSV file
df.to_csv("Times.csv")

cam.release()

#Closes all the windows
cv2.destroyAllWindows