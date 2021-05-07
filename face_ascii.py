import cv2 as cv
import numpy as np
import os
import math
import random
import string

class faceASCII:
    def __init__(self):
        self.welcome()
        cv.namedWindow("preview", cv.WINDOW_GUI_NORMAL)
        self.vc = cv.VideoCapture(0)
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.message = ""
        self.lower_bound = [0]*3
        self.upper_bound = [0]*3
        self.bounds = 0
        cv.setMouseCallback('preview',self.mouseRGB)

        if self.vc.isOpened(): # try to get the first frame
            self.rval, self.frame = self.vc.read()
        else:
            self.rval = False

        os.system('clear') # fixes buggy cv2 messages
        self.find_bounds()

    def welcome(self):
        print("\nWelcome to faceASCII!\n")
        print("To exit the application press ESC at any time")
        print("\nTips: ")
        print(" - Take care when choosing the bounds of your face tone")
        print(" - Ensure the room is well lit and you have a contrasty background")
        print(" - Solid white text on sold black background is recommended")

        input("\nPress Enter to continue...")

    def find_bounds(self):
        print("Please select bounds for skin mask")
        print("Left click to select a bound, right click to reset")
        self.message = "Please click for the darkest tone"
        # print("Please click for the lower bound")
        while self.rval:
            self.rval, self.frame = self.vc.read()
            key = cv.waitKey(20)
            if(self.bounds == 2):
                thresh = self.get_face_mask()
                cv.imshow("preview", self.print_on_frame(thresh))
            else:

                cv.imshow("preview", self.print_on_frame(self.frame))
                if(self.bounds == 3):
                    cv.destroyAllWindows()
                    print("\nNow Please scale terminal font such that the line below")
                    print("fits on one line (then press enter):")

                    print("\Tip: use contrasting colours between text and background")
                    print("\n")
                    print("\n")
                    for i in range(round(self.frame.shape[1]/2)):
                        print('-', end='')
                    print("\n")
                    input("")
                    self.run()
                    break
            if key == 27: # exit on ESC
                break

    #gets pixel value at mouse position
    def mouseRGB(self, event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN: #checks mouse left button down condition

            if(self.bounds == 0 ):
                colours = cv.cvtColor(self.frame,cv.COLOR_BGR2HSV)[y][x]
                print("Lower bound is (HSV): ", colours)
                self.message = "Please click for the lightest tone"
                self.lower_bound = colours
                self.bounds+=1
            elif(self.bounds == 1 ):
                colours = cv.cvtColor(self.frame,cv.COLOR_BGR2HSV)[y][x]
                print("Upper bound is (HSV): ", colours)
                self.upper_bound = colours
                self.message = "Please left click to confirm"
                self.bounds+=1
            else:
                self.bounds+=1
        elif event == cv.EVENT_RBUTTONDOWN:
            if(self.bounds == 1 ):
                print("Resetting lower bound")
                self.message = "Please click for the darkest tone"
                self.bounds = 0
            elif(self.bounds == 2):
                print("Resetting upper bound")
                self.message = "Please click for the lightest tone"
                self.bounds = 1
                cv.destroyWindow("thresh")

    #updates text on initialisation
    def print_on_frame(self, frame):
        cv.putText(frame, self.message,
                (50, 50),
                self.font, 1,
                (255, 255, 255),
                2,
                cv.LINE_4)
        cv.putText(frame, "Right click to go back",
                (50, 100),
                self.font, 1,
                (255, 255, 255),
                2,
                cv.LINE_4)
        return frame

    # main function
    def run(self):
        while self.rval:
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            # cv.imshow("preview", self.frame)
            self.rval, self.frame = self.vc.read()
            key = cv.waitKey(20)

            thresh = self.get_face_mask()
            cv.imshow("thresh", thresh)

            height = self.frame.shape[0]
            width = self.frame.shape[1]
            for i in range(round(height/4)):
                for j in range(round(width/2)):
                    if(thresh[i*4][j*2] == 255):
                        print(random.choice(string.ascii_letters+'1234567890'), end='')
                    else:
                        print(' ', end='')
                print('\n', end='')

            if key == 27: # exit on ESC
                break
        cv.destroyWindow("preview")

    def get_face_mask(self):
        # skin mask
        hsvim = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        # RACISM
        lower = np.array(self.lower_bound, dtype = "uint8")
        upper = np.array(self.upper_bound, dtype = "uint8")
        # /RACISM
        new_lower = np.zeros(lower.shape)
        new_lower[0] = lower[0] - 10
        new_lower[1] = lower[1] - 40
        new_lower[2] = lower[2] - 40
        new_upper = np.zeros(lower.shape)
        new_upper[0] = upper[0] + 10
        new_upper[1] = upper[1] + 40
        new_upper[2] = upper[2] + 40
        skinRegionHSV = cv.inRange(hsvim, new_lower, new_upper)
        blurred = cv.blur(skinRegionHSV, (2,2))
        ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
        return thresh

if __name__ == "__main__":
    fs = faceASCII()
