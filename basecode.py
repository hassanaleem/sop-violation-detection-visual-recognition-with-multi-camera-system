import cv2

def main():

    print("Press 1 for pre-recorded videos, 2 for live stream: ")
    option = int(input())

    if option == 1:
        # Record video
        windowName = "Sample Feed from Camera 1"
        cv2.namedWindow(windowName)

        windowName1 = "Sample Feed from Camera 2"
        cv2.namedWindow(windowName1)

        windowName2 = "Sample Feed from Camera 3"
        cv2.namedWindow(windowName2)

        # capture0 = cv2.VideoCapture(0)  # laptop's camera
        capture1 = cv2.VideoCapture("http://10.130.138.224:4747/video")   # sample code for mobile camera video capture using IP camera
        capture2 = cv2.VideoCapture("http://10.130.138.68:4747/video")   # sample code for mobile camera video capture using IP camera
        capture3 = cv2.VideoCapture("http://10.130.5.86:4747/video")   # sample code for mobile camera video capture using IP camera

        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # define size for recorded video frame for video 2
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # define size for recorded video frame for video 3
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)

        # frame of size is being created and stored in .avi file
        optputFile1 = cv2.VideoWriter(
            'Stream1Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)

        optputFile2 = cv2.VideoWriter(
            'Stream1Recording1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)

        optputFile3 = cv2.VideoWriter(
            'Stream1Recording2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)

        # check if feed exists or not for camera 1
        # if capture0.isOpened():
        #     ret1, frame1 = capture0.read()
        # else:
        #     ret1 = False

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        if capture2.isOpened():  # check if feed exists or not for camera 1
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        if capture3.isOpened():  # check if feed exists or not for camera 1
            ret3, frame3 = capture3.read()
        else:
            ret3 = False

        while ret1:
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            
            # sample feed display from camera 1
            cv2.imshow(windowName, frame1)
            cv2.imshow(windowName1, frame2)
            cv2.imshow(windowName2, frame3)

            # saves the frame from camera 1
            optputFile1.write(frame1)
            optputFile2.write(frame2)
            optputFile3.write(frame3)

            # escape key (27) to exit
            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        capture2.release()
        optputFile2.release()
        capture3.release()
        optputFile3.release()
        cv2.destroyAllWindows()

    elif option == 2:
        # live stream
        windowName = "Sample Feed from Camera 1"
        cv2.namedWindow(windowName)

        windowName1 = "Sample Feed from Camera 2"
        cv2.namedWindow(windowName1)

        windowName2 = "Sample Feed from Camera 3"
        cv2.namedWindow(windowName2)

        # capture0 = cv2.VideoCapture(0)  # laptop's camera
        capture1 = cv2.VideoCapture("http://10.130.138.224:4747/video")   # sample code for mobile camera video capture using IP camera
        capture2 = cv2.VideoCapture("http://10.130.138.68:4747/video")   # sample code for mobile camera video capture using IP camera
        capture3 = cv2.VideoCapture("http://10.130.5.86:4747/video")   # sample code for mobile camera video capture using IP camera
        

        # if capture0.isOpened():  # check if feed exists or not for camera 1
        #     ret0, frame0 = capture0.read()
        # else:
        #     ret0 = False

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        if capture2.isOpened():  # check if feed exists or not for camera 1
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        if capture3.isOpened():  # check if feed exists or not for camera 1
            ret3, frame3 = capture3.read()
        else:
            ret3 = False

        while ret1:  # and ret2 and ret3:
            
            # ret0, frame0 = capture0.read() 
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            # cv2.imshow(windowName1, frame0)
            cv2.imshow(windowName, frame1)
            cv2.imshow(windowName1, frame2)
            cv2.imshow(windowName2, frame3)

            if cv2.waitKey(1) == 27:
                break

        # capture0.release()
        capture1.release()
        capture2.release()
        capture3.release()
        cv2.destroyAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
