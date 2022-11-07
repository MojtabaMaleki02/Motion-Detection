import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import PyQt5.QtWidgets as PyWidget
import PyQt5.QtGui as PyGui
from datetime import datetime
import cv2
import pandas as pd
import time
from moviepy.editor import *
import numpy as np
from tkinter import filedialog
import sys
import PySimpleGUI as sg
import subprocess

layout = [
            [sg.T("")],
            [sg.Text("Choose a folder: "),sg.Input(key="-IN2-" ,change_submits=True),sg.FileBrowse(key="-IN-")],
            [sg.Text("Acuraccy", font='Lucida')],
            [sg.Slider(orientation ='horizontal', key='stSlider', range=(100,90000))],
            [sg.Text("X:      "),sg.Input(key="-IN3-" ,change_submits=True)],
            [sg.Text("Y:      "),sg.Input(key="-IN4-" ,change_submits=True)],
            [sg.Text("Height: "),sg.Input(key="-IN5-" ,change_submits=True)],
            [sg.Text("Width:  "),sg.Input(key="-IN6-" ,change_submits=True)],
            [sg.Button("Submit")]
        ]

window = sg.Window('My File Browser', layout, size=(600,600))
event, values = window.read()
file=values["-IN-"]    
acuraccySlider=values["stSlider"]   
print('ACURACYYYYYYYYY:',acuraccySlider)





def press_it():
            STime=0
            FTime=0
            frame_counter=0
            moji = True
            first_frame = None
            status_list = [None,None]
            times = []
            startTime=datetime.now()
            print(startTime)
            #Dataframe to store the time values during which object detection and movement appears | "C:/Users/mojta/Desktop/videos/pred.mp4"
            df = pd.DataFrame(columns=['Start','End','Duration'])
            cam = cv2.VideoCapture(file)
            num_of_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cam.get(cv2.CAP_PROP_FPS)
            seconds = round(frames / fps)
            length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
            print(length)
            y=int(values["-IN4-"])
            x=int(values["-IN3-"])
            h= int(values["-IN5-"])
            w= int(values["-IN6-"])


            
            #Iterate through frames and display the window
            
            while cam.isOpened():

                check, frame = cam.read()
                length-=1
                frame_counter+=1
                if moji==True:
                    STime=datetime.now()
                    moji=False

                
                frame = frame[y:y+h, x:x+w]
                #Status at beginning of the recording is zero as the object is not visisble
                status = 0

                #Converting each frame into gray scale image
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                #Convert grayscale image to GaussianBlur
                gray = cv2.GaussianBlur(gray, (21,21), 0)

                #This is used to store the first image/frame of the video
                if first_frame is None or length%500==0:
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
                Acuraccy = acuraccySlider
                for cont in cnts:
                    if cv2.contourArea(cont) < Acuraccy:
                        continue
                    #Change in status when the object is being detected
                    status = 1
                    #creates a rectangular box around the object in the frame
                    (x1, y1, w1, h1) = cv2.boundingRect(cont)
                    cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), (0,0,255), 3)

                #List of status for every frame
                status_list.append(status)
                status_list = status_list[-2:]

                #Record datetime in a list when change occurs
                if status_list[-1]==1 and status_list[-2]==0:
                    #times.append(datetime.now()-startTime)
                    times.append(frame_counter)

                if status_list[-1]==0 and status_list[-2]==1:
                    #times.append(datetime.now()-startTime)
                    times.append(frame_counter)
                    

                #Opening all types of frames/images
                cv2.imshow("Grey Scale",gray)
                cv2.imshow("Delta", delta_frame)
                cv2.imshow("Threshold", thresh_delta)
                cv2.imshow("Colored frame",frame)

                last_frame_num = cam.get(cv2.CAP_PROP_FRAME_COUNT)
                #Generate a new frame after every 1 millisecond
                key = cv2.waitKey(1)
                #If entered 'q' on keyboard, breaks out of loop, and window gets destroyed
                
                print(length)
                if key == ord('q') or length<=10:
                    if status==1:
                        #times.append(datetime.now()-startTime)
                        times.append(frame_counter)
                        FTime=datetime.now()
                    break

                

            #Store time values in a Dataframe

            DURATION=FTime-STime
            FINAL = DURATION/seconds
            RESULT = float(seconds/num_of_frames)
            print(type(DURATION))
            print(FINAL)
            print(seconds)
            print(fps)
            print(RESULT)
            #date = datetime.fromtimestamp((59*(times[0]/DURATION)) / 1e3)
            
            #59/892


            for i in range(0,len(times),2):
                if len(times)%2==1 and i==len(times)-1:
                    break
                df = df.append({'Start':RESULT*times[i],'End':RESULT*times[i+1],'Duration':RESULT*(times[i+1]-times[i])}, ignore_index=True)
                #df = df.append({'Start':datetime.fromtimestamp((59*(times[i]/DURATION)) / 1e3),'End':datetime.fromtimestamp((59*(times[i+1]/DURATION)) / 1e3),'Duration':datetime.fromtimestamp((59*(times[i+1]/DURATION)) / 1e3)-datetime.fromtimestamp((59*(times[i]/DURATION)) / 1e3)}, ignore_index=True)


            #Write the dataframe to a CSV file
            df.to_csv("Times.csv")

            cam.release()

            #Closes all the windows
            cv2.destroyAllWindows
            window.Close()

while True:             # Event Loop
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event == 'Submit':
        press_it()
    
window.Close()


