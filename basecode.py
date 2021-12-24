from typing_extensions import ParamSpec
import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='final.pt')
top = cv2.imread("sstop.png")

src = np.array([[605,925],[386,766],[234,653],[124,573],[415,503],[654,447],[988,373],[1325,439],[1575,485],[1880,543],[1738,633],[1572,753],[1340,917]], dtype=float)
dst = np.array([[612,462],[716,459],[822,461],[928,462],[928, 568],[928,672],[928,883],[717,883],[612,883],[507,883],[507,779],[507,671],[507,567]], dtype=float)

src1 = np.array([[195,341],[120,287],[68,251],[28,221],[125,203],[200,188],[310,171],[415,193],[491,214],[583,235],[543,261],[496,296],[427,343]], dtype=float)
dst1 = np.array([[507,989],[507,884],[507,778],[507,673],[612, 673],[716,673],[927,673],[927,883],[927,988],[927,1093],[822,1093],[717,1093],[613,1093]], dtype=float)

src2 = np.array([[580,935],[384,786],[239,683],[144,605],[416,548],[639,509],[946,445],[1237,523],[1453,571],[1708,645],[1598,707],[1455,805],[1238,945]], dtype=float)
dst2 = np.array([[1140,655],[1140,672],[1140,777],[1140,886],[1033, 883],[928,885],[717,881],[716,672],[716,568],[716,455],[823,416],[929,462],[1035,461]], dtype=float)
        

def find_dis(p_list, result, pointsSopViolation):
    list_len = len(p_list)
    for i in range(0,list_len):
        for j in range(0,list_len):
            dist = np.sqrt(np.power(p_list[i][0]-p_list[j][0],2) + np.power(p_list[i][1]-p_list[j][1],2))
            #dist = np.linalg.norm(p_list[i] - p_list[j])
            print("\ndist = ",dist,"\n")
            if(dist < 108 and dist > 10):       
                pointsSopViolation.append(p_list[i])
                pointsSopViolation.append(p_list[j])    
                print("\n\n\n\n SOCIAL DISTANCE PLZZZ \n\n\n\n")             # 2m in pix is 108 
                cv2.line(result, (p_list[i][0], p_list[i][1]), (p_list[j][0], p_list[j][1]), (0, 255, 0), thickness=3, lineType=8)                
                cv2.putText(result, "violation", (p_list[i][0], p_list[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

    return result
def heatmap(points, type, heatmapW, gtop, pointCount, windowName):
    k = 21
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = (gauss/(gauss[int(k/2), int(k/2)]))

    img2 = np.zeros((top.shape[0], top.shape[1], 3)).astype(np.float32)
    j = cv2.cvtColor(cv2.applyColorMap(((gauss)*255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    for p in points:
        b = img2[p[0]-int(k/2):p[0]+int(k/2)+1, p[1]-int(k/2): p[1]+int(k/2)+1, :]
        c = j+b
        img2[p[0]-int(k/2):p[0]+int(k/2)+1, p[1]-int(k/2): p[1]+int(k/2)+1, :] = c


    g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask = np.where(g > 0.2, 1, 0).astype(np.float32)
    mask_3 = np.ones((top.shape[0], top.shape[1], 3)) * (1 - mask)[:, :, None]
    mask_4 = img2 * (mask)[:, :, None]

    new_top = mask_3 * gtop
    mask_4=mask_4.astype(np.float64)
    heatmap = cv2.addWeighted(new_top,1,mask_4,1,0)

    if type == "static":
        heatmap=(heatmap*255).astype(np.uint8)
        cv2.imshow(windowName, heatmap)
        heatmapW.write(heatmap)

    if type == "animated":
        sum = np.sum(pointCount)
        cnt = 0
        if sum > 20:
            cnt = pointCount[-1]
            pointCount = pointCount[1:]

        points = points[cnt:] 
        
        heatmap=(heatmap*255).astype(np.uint8)
        cv2.imshow(windowName, heatmap)

        heatmapW.write(heatmap)
        return points

def main_func(cap, cap1, cap2, source, destination, source1, destination1, source2, destination2):
    points_list = []
    pointCount = []
    pointsAnimated = []
    pointsSopViolation = []
    points = list()
    heatmapS = cv2.VideoWriter('heatmapStatic.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (top.shape[1], top.shape[0]))
    heatmapA = cv2.VideoWriter('heatmapAnimated.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (top.shape[1], top.shape[0]))
    heatmapViolation = cv2.VideoWriter('heatmapViolation.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (top.shape[1], top.shape[0]))
    
    top_rows = top.shape[0]
    top_cols = top.shape[1]
    optputFile1 = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (top_cols, top_rows))

    while (cap.isOpened()):

        points_list.clear()

        ret, f = cap.read()
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        frame = f.copy()
        frame1 = f1.copy()
        frame2 = f2.copy()

        results = model(frame)
        results.render()
        cv2.imshow("frame", frame)

        results1 = model(frame1)
        results1.render()
        cv2.imshow("1", frame1)
      

        results2 = model(frame2)
        results2.render()
        cv2.imshow("2", frame2)

        
        M, mask = cv2.findHomography(source, destination)
        top_img = cv2.warpPerspective(f, M, (top_cols, top_rows))
        
        M1, mask1 = cv2.findHomography(source1, destination1)
        top_img1 = cv2.warpPerspective(f1, M1, (top_cols, top_rows))

        M2, mask2 = cv2.findHomography(source2, destination2)
        top_img2 = cv2.warpPerspective(f2, M2, (top_cols, top_rows))

        result1 = cv2.addWeighted(top_img, 0.3, top_img2, 0.3, 0)
        result2 = cv2.addWeighted(top_img1, 0.3, result1, 1, 0)
        result = cv2.addWeighted(top, 0.5, result2, 1, 0)

        cooridnatesAll = None
        try:
            cooridnatesAll = results.pandas().xyxy[0].to_dict(orient="records")
        except:
            pass

        try:
            for coordinates in cooridnatesAll:
                x1 = coordinates['xmin']
                x2 = coordinates['xmax']
                y1 = coordinates['ymin']
                y2 = coordinates['ymax']
                label = coordinates['names']
                base_mid = [((x1+x2)/2),y2+90, 1]
                boxcordinates = np.array([
                [[base_mid[0], base_mid[1]]],
                [[base_mid[0], base_mid[1]]],
                [[base_mid[0], base_mid[1]]],
                [[base_mid[0], base_mid[1]]]
                ])

                imgdot = cv2.perspectiveTransform(boxcordinates, M)
                dotx = imgdot[0][0][0]
                doty = imgdot[0][0][1]
                dot = (int(dotx)), int((doty))

                points_list.append(dot)
                points.append(dot)
                pointsAnimated.append(dot)

                if(label == 'mask'):
                    cv2.circle(result,dot, 8, (0,0,255),-1)
                if(label == 'no_mask'):
                    cv2.circle(result,dot, 8, (0,128,0),-1)

        except:
            pass


        coordinatesAll1 = None
        
        coordinatesAll1 = results1.pandas().xyxy[0].to_dict(orient="records")    

        try:
            for coordinates1 in coordinatesAll1:
                x11 = coordinates1['xmin']
                x21 = coordinates1['xmax']
                y11= coordinates1['ymin']
                y21= coordinates1['ymax']
                label1 = coordinates1['name']
                base_mid1 = [((x11+x21)/2),y21+90, 1]
                boxcordinates1 = np.array([
                [[base_mid1[0], base_mid1[1]]],
                [[base_mid1[0], base_mid1[1]]],
                [[base_mid1[0], base_mid1[1]]],
                [[base_mid1[0], base_mid1[1]]]
                ])

                imgdot1 = cv2.perspectiveTransform(boxcordinates1, M1)
                dot1x = imgdot1[0][0][0]
                dot1y = imgdot1[0][0][1]
                dot1 = (int(dot1x)), int((dot1y))

                points_list.append(dot1)
                points.append(dot1)
                pointsAnimated.append(dot1)

                if(label1 == 'mask'):
                    cv2.circle(result, dot1, 8, (0,0,255), -1)

                if(label1 == 'no_mask'):
                    cv2.circle(result,dot1, 8, (0,128,0),-1)

        except:
            pass


        coordinatesAll2 = None
        try:
            coordinatesAll2 = results2.pandas().xyxy[0].to_dict(orient="records")
        except:
            pass

        try:
            for coordinates2 in coordinatesAll2:
                x12 = coordinates2['xmin']
                x22 = coordinates2['xmax']
                y12 = coordinates2['ymin']
                y22 = coordinates2['ymax']
                label2 = coordinates2['name']

                base_mid2 = [((x12+x22)/2),y22+140, 0]

                boxcordinates2 = np.array([
                [[base_mid2[0], base_mid2[1]]],
                [[base_mid2[0], base_mid2[1]]],
                [[base_mid2[0], base_mid2[1]]],
                [[base_mid2[0], base_mid2[1]]]
                ])
                

                imgdot2 = cv2.perspectiveTransform(boxcordinates2, M2)

                dot2x = imgdot2[0][0][0]
                dot2y = imgdot2[0][0][1]
                dot2 = (int(dot2x)), int((dot2y))

                points_list.append(dot2)
                points.append(dot2)
                pointsAnimated.append(dot2)

                if(label2 == 'mask'):
                    cv2.circle(result,dot2, 8, (0,0,255),-1)
                if(label2 == 'no_mask'):
                    cv2.circle(result,dot2, 8, (0,128,0),-1)

        except:
            pass

        try:
            pointCount.append(len(points_list))

        except:
            pass

        if(len(points_list) > 1):
            result = find_dis(points_list, result, pointsSopViolation)

        heatmap(points, "static", heatmapS, result, pointCount, "heatmap static")
        pointsAnimated = heatmap(pointsAnimated, "animated", heatmapA, result, pointCount, "heatmap animated")
        heatmap(pointsSopViolation, "static", heatmapViolation, result, 0, "heatmap SOP violation")
        
        optputFile1.write(result)
        cv2.imshow("Result", result)
        key = cv2.waitKey(1)
        if key == 27:
            break

    optputFile1.release()
    cap.release()
    cv2.destroyAllWindows() 

def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
  
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[0], str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', params[0])
        pointSrc = params[2]
        pointSrc.append([x, y])

    if event == cv2.EVENT_RBUTTONDOWN:
 
 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[1], str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('Top', params[1])
        pointDst = params[3]
        pointDst.append([x, y])
    

ans = int(input("Enter 1 prerecorded 2 live : "))


if ans == 1:
    cap = cv2.VideoCapture('cam1.avi')
    cap1 = cv2.VideoCapture('cam2.avi')
    cap2 = cv2.VideoCapture('cam3.avi')
    main_func(cap, cap1, cap2, src, dst, src1, dst1, src2, dst2)
    

elif ans == 2:
    
    cap = cv2.VideoCapture("http://10.130.138.224:4747/video")
    cap1 = cv2.VideoCapture("http://10.130.138.68:4747/video")
    cap2 = cv2.VideoCapture(0)
    
    answer = int(input("Press 1 for dynamic allocation of points, press 2 for preexisting points"))

    if answer == 1:
        _, pts = cap.read()
        _, pts1 = cap1.read()
        _, pts2 = cap2.read()

        s = list()
        d = list()
        s1 = list()
        d1 = list()
        s2 = list()
        d2 = list()

        topc = top.copy()
        cv2.imshow("image", pts)
        cv2.setMouseCallback('image', click_event, [pts, topc, s, d])
        cv2.imshow("Top", topc)
        cv2.setMouseCallback('Top', click_event, [pts, topc, s, d])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        topc1 = top.copy()
        cv2.imshow("image", pts1)
        cv2.setMouseCallback('image', click_event, [pts1, topc1, s1, d1])
        cv2.imshow("Top", topc1)
        cv2.setMouseCallback('Top', click_event, [pts1, topc1, s1, d1])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        topc2 = top.copy()
        cv2.imshow("image", pts2)
        cv2.setMouseCallback('image', click_event, [pts2, topc2, s2, d2])
        cv2.imshow("Top", topc2)
        cv2.setMouseCallback('Top', click_event, [pts2, topc2, s2, d2])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        s = np.array(s)
        d = np.array(d)
        s1 = np.array(s1)
        d1 = np.array(d1)
        s2 = np.array(s2)
        d2 = np.array(d2)
        main_func(cap, cap1, cap2, s, d, s1, d1, s2, d2)

    if answer == 2:
        main_func(cap, cap1, cap2, src, dst, src1, dst1, src2, dst2)



