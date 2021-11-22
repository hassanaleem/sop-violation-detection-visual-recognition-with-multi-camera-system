import cv2

def main():

    top = cv2.imread("sstop.png")

    print("Press 1 for recording videos, 2 for live stream, 3 for loading pre recorded video: ")
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
        capture1 = cv2.VideoCapture("http://10.130.138.224:4747/video")
        capture2 = cv2.VideoCapture("http://10.130.138.68:4747/video")  
        capture3 = cv2.VideoCapture("http://10.130.5.86:4747/video") 

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

        if capture2.isOpened():  # check if feed exists or not for camera 2
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        if capture3.isOpened():  # check if feed exists or not for camera 3
            ret3, frame3 = capture3.read()
        else:
            ret3 = False

        while ret1:
            ret1, frame = capture1.read()
            ret2, frame1 = capture2.read()
            ret3, frame2 = capture3.read()

            ################################################################################# frame 
            source = np.array([[300,800],[760,800],[50,1600],[920,1600]], dtype=float)
            destination = np.array([[300,460], [430, 290], [600, 775], [760, 600]], dtype=float)
            M, mask = cv2.findHomography(source, destination)
            top_rows = top.shape[0]
            top_cols = top.shape[1]
            top_img = cv2.warpPerspective(frame, M, (top_cols, top_rows))
            cv2.imshow("top_img", top_img)
            ################################################################################# frame 1
            source1 = np.array([[620,1274],[807,1266],[2,1307],[538,1636]], dtype=float)
            destination1 = np.array([[777,787], [643, 871], [513, 289], [313, 417]], dtype=float)
            M1, mask1 = cv2.findHomography(source1, destination1)
            top_img1 = cv2.warpPerspective(frame1, M1, (top_cols, top_rows))
            cv2.imshow("top_img1", top_img1)
            ################################################################################# frame 1
            source2 = np.array([[3,1318],[305,1310],[150,1777],[1340,1290]], dtype=float)
            destination2 = np.array([[833,729], [720, 1817], [585, 365], [335, 463]], dtype=float)
            M2, mask2 = cv2.findHomography(source2, destination2)
            top_img2 = cv2.warpPerspective(frame2, M2, (top_cols, top_rows))
            cv2.imshow("top_img2", top_img2)
            print(top_img2.shape)
            #cv2.fillConvexPoly(top, destination.astype(int), 0, 16)
            print("in")
            result = top + top_img1 + top_img2
        
            cv2.imshow("Result", result)
            
            # sample feed display from camera 1
            cv2.imshow(windowName, frame)
            cv2.imshow(windowName1, frame1)
            cv2.imshow(windowName2, frame2)

            # saves the frame from camera 1
            optputFile1.write(frame)
            optputFile2.write(frame1)
            optputFile3.write(frame2)

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
        capture1 = cv2.VideoCapture("http://10.130.138.224:4747/video")
        capture2 = cv2.VideoCapture("http://10.130.138.68:4747/video")
        capture3 = cv2.VideoCapture("http://10.130.5.86:4747/video")
        

        # if capture0.isOpened():  # check if feed exists or not for camera 1
        #     ret0, frame0 = capture0.read()
        # else:
        #     ret0 = False

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        if capture2.isOpened():  # check if feed exists or not for camera 2
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        if capture3.isOpened():  # check if feed exists or not for camera 3
            ret3, frame3 = capture3.read()
        else:
            ret3 = False

        while ret1:  # and ret2 and ret3:
            
            # ret0, frame0 = capture0.read() 
            ret1, frame = capture1.read()
            ret2, frame1 = capture2.read()
            ret3, frame2 = capture3.read()

            ################################################################################# frame 
            source = np.array([[300,800],[760,800],[50,1600],[920,1600]], dtype=float)
            destination = np.array([[300,460], [430, 290], [600, 775], [760, 600]], dtype=float)
            M, mask = cv2.findHomography(source, destination)
            top_rows = top.shape[0]
            top_cols = top.shape[1]
            top_img = cv2.warpPerspective(frame, M, (top_cols, top_rows))
            cv2.imshow("top_img", top_img)
            ################################################################################# frame 1
            source1 = np.array([[620,1274],[807,1266],[2,1307],[538,1636]], dtype=float)
            destination1 = np.array([[777,787], [643, 871], [513, 289], [313, 417]], dtype=float)
            M1, mask1 = cv2.findHomography(source1, destination1)
            top_img1 = cv2.warpPerspective(frame1, M1, (top_cols, top_rows))
            cv2.imshow("top_img1", top_img1)
            ################################################################################# frame 1
            source2 = np.array([[3,1318],[305,1310],[150,1777],[1340,1290]], dtype=float)
            destination2 = np.array([[833,729], [720, 1817], [585, 365], [335, 463]], dtype=float)
            M2, mask2 = cv2.findHomography(source2, destination2)
            top_img2 = cv2.warpPerspective(frame2, M2, (top_cols, top_rows))
            cv2.imshow("top_img2", top_img2)
            print(top_img2.shape)
            #cv2.fillConvexPoly(top, destination.astype(int), 0, 16)
            print("in")
            result = top + top_img1 + top_img2
        
            cv2.imshow("Result", result)

            # cv2.imshow(windowName1, frame0)
            cv2.imshow(windowName, frame)
            cv2.imshow(windowName1, frame1)
            cv2.imshow(windowName2, frame2)

            if cv2.waitKey(1) == 27:
                break

        # capture0.release()
        capture1.release()
        capture2.release()
        capture3.release()
        cv2.destroyAllWindows()
    elif option == 3:
        #add path
        cap = cv2.VideoCapture('v.avi')     #video 1
        cap1 = cv2.VideoCapture('vv.mov')   #video 2
        cap2 = cv2.VideoCapture('vvv.mov')  #video 3

        while (cap.isOpened()):

            ret, frame = cap.read()
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #################################################################################
            cv2.imshow("Top", top)
            cv2.imshow("Frame", frame)
            cv2.imshow("Frame1", frame1)
            cv2.imshow("Frame2", frame2)

            ################################################################################# frame 
            source = np.array([[300,800],[760,800],[50,1600],[920,1600]], dtype=float)
            destination = np.array([[300,460], [430, 290], [600, 775], [760, 600]], dtype=float)
            M, mask = cv2.findHomography(source, destination)
            top_rows = top.shape[0]
            top_cols = top.shape[1]
            top_img = cv2.warpPerspective(frame, M, (top_cols, top_rows))
            cv2.imshow("top_img", top_img)
            ################################################################################# frame 1
            source1 = np.array([[620,1274],[807,1266],[2,1307],[538,1636]], dtype=float)
            destination1 = np.array([[777,787], [643, 871], [513, 289], [313, 417]], dtype=float)
            M1, mask1 = cv2.findHomography(source1, destination1)
            top_img1 = cv2.warpPerspective(frame1, M1, (top_cols, top_rows))
            cv2.imshow("top_img1", top_img1)
            ################################################################################# frame 1
            source2 = np.array([[3,1318],[305,1310],[150,1777],[1340,1290]], dtype=float)
            destination2 = np.array([[833,729], [720, 1817], [585, 365], [335, 463]], dtype=float)
            M2, mask2 = cv2.findHomography(source2, destination2)
            top_img2 = cv2.warpPerspective(frame2, M2, (top_cols, top_rows))
            cv2.imshow("top_img2", top_img2)
            print(top_img2.shape)
            #cv2.fillConvexPoly(top, destination.astype(int), 0, 16)
            print("in")
            result = top + top_img1 + top_img2
        
            cv2.imshow("Result", result)
            key = cv2.waitKey(1)

            if key == 27:
                break
        cap.release()
        cv2.destrouAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
