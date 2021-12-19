import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_hhs.pt')

top = cv2.imread("sstop.png")
cap = cv2.VideoCapture('cam1.mov')
cap1 = cv2.VideoCapture('cam2.mov')
cap2 = cv2.VideoCapture('cam3.mov')

top_rows = top.shape[0]
top_cols = top.shape[1]

while (cap.isOpened()):

    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #################################################################################
    windowName = "Sample Feed from Camera 1"
    cv2.namedWindow(windowName)
    results = model(frame)
    results.render()
    cv2.imshow(windowName, results.imgs[0])

    try:
        cooridnates = results.pandas().xyxy[0].to_dict(orient="records")[0]
        x1 = cooridnates['xmin']
        x2 = cooridnates['xmax']
        y1 = cooridnates['ymin']
        y2 = cooridnates['ymax']
        print(x1, x2, y1, y2)
        base_mid = [[((x1+x2)/2)],[y2], [0]]
        print("Mid Point: ", base_mid)
    except:
        pass

    windowName1 = "Sample Feed from Camera 2"
    cv2.namedWindow(windowName1)
    results1 = model(frame1)
    results1.render()
    cv2.imshow(windowName1, results1.imgs[0])

    try:
        cooridnates1 = results1.pandas().xyxy[0].to_dict(orient="records")[0]
        x11 = cooridnates1['xmin']
        x21 = cooridnates1['xmax']
        y11= cooridnates1['ymin']
        y21= cooridnates1['ymax']
        print(x11, x21, y11, y21)
        base_mid1 = np.array([((x11+x21)/2),y21, 0])
        print("Mid Point 1: ", base_mid1)
    except:
        pass

    windowName2 = "Sample Feed from Camera 3"
    cv2.namedWindow(windowName2)
    results2 = model(frame2)
    results2.render()
    cv2.imshow(windowName2, results2.imgs[0])

    try:
        cooridnates2 = results2.pandas().xyxy[0].to_dict(orient="records")[0]
        x12 = cooridnates2['xmin']
        x22 = cooridnates2['xmax']
        y12 = cooridnates2['ymin']
        y22 = cooridnates2['ymax']
        print(x12, x22, y12, y22)
        base_mid2 = [[((x12+x22)/2)],[y22], [0]]
        print("Mid Point 2: ", base_mid2)
    except:
        pass

    ################################################################################# frame hh cam 1
    #ground points
    source = np.array([[605,925],[386,766],[234,653],[124,573],[415,503],[654,447],[988,373],[1325,439],[1575,485],[1880,543],[1738,633],[1572,753],[1340,917]], dtype=float)
    destination = np.array([[612,462],[716,459],[822,461],[928,462],[928, 568],[928,672],[928,883],[717,883],[612,883],[507,883],[507,779],[507,671],[507,567]], dtype=float)
    
    M, mask = cv2.findHomography(source, destination)
    top_img = cv2.warpPerspective(frame, M, (top_cols, top_rows))

    try:
        top_img_dot = np.array(np.dot(M,base_mid))
        print("M = ", M)
        top_img_dot = (abs(top_img_dot[0][0]), abs(top_img_dot[1][0]))
        print("Dot corridnates",top_img_dot[0],top_img_dot[1])
        cv2.circle(top,(int(top_img_dot[0]),int(top_img_dot[1])), 8, (0,0,255),-1)
        cv2.imshow("top", top)
    except:
        pass
    cv2.imshow("top_img", top_img)


    ################################################################################# frame 1 aa cam 2
    source1 = np.array([[195,341],[120,287],[68,251],[28,221],[125,203],[200,188],[310,171],[415,193],[491,214],[583,235],[543,261],[496,296],[427,343]], dtype=float)
    destination1 = np.array([[507,989],[507,884],[507,778],[507,673],[612, 673],[716,673],[927,673],[927,883],[927,988],[927,1093],[822,1093],[717,1093],[613,1093]], dtype=float)
    
    M1, mask1 = cv2.findHomography(source1, destination1)
    top_img1 = cv2.warpPerspective(frame1, M1, (top_cols, top_rows))
    
    try:
        top_img_dot1 = np.matmul(M1, base_mid1)
        print(M1.shape, base_mid1.shape)
        print("M1 = ", M1)
        print("top_img_dot1", top_img_dot1)
        # top_img_dot1 = np.array(top_img_dot1[0][0], top_img_dot1[1][0],top_img_dot1[2][0])
        # print("Dot corridnates 1",top_img_dot1[0],top_img_dot1[1])
        # cv2.circle(top,(int(top_img_dot1[0]),int(top_img_dot1[1])), 8, (0,0,255),-1)
        # cv2.imshow("top", top)
    except:
        pass
    cv2.imshow("top_img", top_img)

    ################################################################################# frame 2 jj cam 3 done
    # Head points 
    # source = np.array([[370,254],[690,460],[937,433],[1480,580],[1320,668]], dtype=float)
    # destination = np.array([[1140,675],[1033,673],[932,667],[928,463],[1035, 460]], dtype=float)

    # ground points
    source2 = np.array([[580,935],[384,786],[239,683],[144,605],[416,548],[639,509],[946,445],[1237,523],[1453,571],[1708,645],[1598,707],[1455,805],[1238,945]], dtype=float)
    destination2 = np.array([[1140,655],[1140,672],[1140,777],[1140,886],[1033, 883],[928,885],[717,881],[716,672],[716,568],[716,455],[823,416],[929,462],[1035,461]], dtype=float)
    
    M2, mask2 = cv2.findHomography(source2, destination2)
    top_img2 = cv2.warpPerspective(frame2, M2, (top_cols, top_rows))
    
    try:
        top_img_dot2 = np.array(np.dot(M2,base_mid2))
        print("M2 = ", M2)
        top_img_dot2 = (abs(top_img_dot2[0][0]), abs(top_img_dot2[1][0]))
        print("Dot corridnates",top_img_dot2[0],top_img_dot2[1])
        cv2.circle(top,(int(top_img_dot2[0]),int(top_img_dot2[1])), 8, (0,0,255),-1)
        cv2.imshow("top", top)
    except:
        pass
    cv2.imshow("top_img", top_img)
    
    result1 = cv2.addWeighted(top_img, 0.3, top_img2, 0.3, 0)
    result2 = cv2.addWeighted(top_img1, 0.3, result1, 1, 0)
    result = cv2.addWeighted(top, 0.5, result2, 1, 0)
    cv2.imshow("Result", result)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows() 




# import cv2
# import torch
# import numpy as np

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_new.pt')
# top = cv2.imread("sstop.png")
# top_rows = top.shape[0]
# top_cols = top.shape[1]

# cap = cv2.VideoCapture('cam3.avi')
# # cap1 = cv2.VideoCapture('vv.avi')
# # cap2 = cv2.VideoCapture('vvv.avi')

# windowName = "Sample Feed from Camera 1"
# cv2.namedWindow(windowName)

# while (cap.isOpened()):

#     ret, frame = cap.read()
#     results = model(frame)
#     results.render()
#     cv2.imshow(windowName, results.imgs[0])
    
#     try:
#         cooridnates = results.pandas().xyxy[0].to_dict(orient="records")[0]
#         x1 = cooridnates['xmin']
#         x2 = cooridnates['xmax']
#         y1 = cooridnates['ymin']
#         y2 = cooridnates['ymax']
#         print(x1, x2, y1, y2)
#         base_mid = [[((x1+x2)/2)],[y2], [0]]
#         print("Mid Point: ", base_mid)
#     except:
#         pass
    

#     # ret1, frame1 = cap1.read()
#     # ret2, frame2 = cap2.read()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     #################################################################################
#     # cv2.imshow("Top", top)
#     cv2.imshow("Frame", frame)
#     # cv2.imshow("Frame1", frame1)
#     # cv2.imshow("Frame2", frame2)

#     ################################################################################# frame 
#     source = np.array([[580,935],[384,786],[239,683],[144,605],[416,548],[639,509],[946,445],[1237,523],[1453,571],[1708,645],[1598,707],[1455,805],[1238,945]], dtype=float)
#     destination = np.array([[1140,655],[1140,672],[1140,777],[1140,886],[1033, 883],[928,885],[717,881],[716,672],[716,568],[716,455],[823,416],[929,462],[1035,461]], dtype=float)

#     # source = np.array([[370,254],[690,460],[937,433],[1480,580],[1320,668]], dtype=float)
#     # destination = np.array([[1140,675],[1033,673],[932,667],[928,463],[1035, 460]], dtype=float)
    
#     M, mask = cv2.findHomography(source, destination)
#     top_img = cv2.warpPerspective(results.imgs[0], M, (top_cols, top_rows))
#     try:
#         top_img_dot = np.array(np.dot(M,base_mid))
#         print("M = ", M)
#         top_img_dot = (abs(top_img_dot[0][0]), abs(top_img_dot[1][0]))
#         print("Dot corridnates",top_img_dot[0],top_img_dot[1])
#         cv2.circle(top,(int(top_img_dot[0]),int(top_img_dot[1])), 8, (0,0,255),-1)
#         cv2.imshow("top", top)
#     except:
#         pass
#     cv2.imshow("top_img", top_img)

#     ################################################################################# frame 1
#     # source1 = np.array([[240,293],[378,317],[452,330],[630,370],[100,315],[160,470],[130,300]], dtype=float)
#     # destination1 = np.array([[98,275],[98,175],[98,152],[105,115],[160, 180], [161, 110],[145,283]], dtype=float)
    
#     # M1, mask1 = cv2.findHomography(source1, destination1)
#     # top_img1 = cv2.warpPerspective(frame1, M1, (top_cols, top_rows))
#     # cv2.imshow("top_img1", top_img1)

#     # ################################################################################# frame 2
#     # source2 = np.array([[540,270],[377,284],[308,290],[30,315],[638,270],[630,475]], dtype=float)
#     # destination2 = np.array([[165,275],[165,175],[165,152],[165,107],[130, 305],[100,108]], dtype=float)
    
#     # M2, mask2 = cv2.findHomography(source2, destination2)
#     # top_img2 = cv2.warpPerspective(frame2, M2, (top_cols, top_rows))
#     # cv2.imshow("top_img2", top_img2)

#     #################################################################################
#     # b_channel,g_channel,r_channel = cv2.split(top_img2)
#     # a_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*0.1
#     # top_img2 = cv2.merge((b_channel,g_channel,r_channel,a_channel))
    
#     result = top_img
#     cv2.imshow("Result", result)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows() 