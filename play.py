import time
import cv2
from utils.screengrab import screen_record
from utils.directkeys import PressKey, ReleaseKey
import argparse
import numpy as np
import os

"""
With support from:
pythonprogramming.net - Python plays GTA V
https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
##########################################
https://www.ponggame.org/ requires flash
##########################################
"""

"""
AI logic:
Change with ai = 1, 2, 3 input below
SIMPLE:
Move player paddle to ball y-position
INT:
Move player paddle to predicted ball position 
by direction vector
ADV:
Move player paddle to predicted ball position 
by direction vector + reflection*
"""

def process_img(img, threshold = 127, factor = 0.5):
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=factor, fy=factor).astype('uint8')
    return np.uint8((gray > threshold) * 255)

def get_pong(img, factor = 0.5):
    # finds pong about 1 every 2 frames. Consistent to not get other objects
    pong_size_a = 72
    pong_size_b = 64
    connectivity = 8

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_16U)
    # area is last column (-1) in stats
    bool_index_a = stats[:, -1] == pong_size_a
    bool_index_b = stats[:, -1] == pong_size_b
    bool_index = bool_index_a ^ bool_index_b
    pong_cent = centroids[bool_index] * (1 / factor)
    if pong_cent.size > 0:
        pong_cent = tuple(map(int, pong_cent.tolist()[0]))

    return pong_cent

def get_objects_in_masked_region(img, vertices,  connectivity = 8):
    ''':return connected components with stats in masked region
    [0] retval number of total labels 0 is background
    [1] labels image
    [2] stats[0] leftmostx, [1] topmosty, [2] horizontal size, [3] vertical size, [4] area
    [3] centroids
    '''
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, [vertices], 255)
    # now only show the area that is the mask
    mask = cv2.bitwise_and(img, mask)
    conn = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_16U)
    return conn

def get_paddles(img, factor = 0.5):
    '''
    img: segmented and processed image
    factor: scaling value of downsized image from original unprocessed image
    return: computer paddle centroid, player paddle centroid position
    '''

    y,x = img.shape
    xlc_f = 0.04 #x, left computer factor
    xrc_f = 0.10
    xlp_f = 0.90 #x, left player factor
    xrp_f = 0.96
    yl_f = 0.03 # y doesn't need a distinction for comp or player
    yr_f = 0.96

    vertices_left = np.array([[xlc_f*x,yl_f*y],[xlc_f*x,yr_f*y],[xrc_f*x,yr_f*y],[xrc_f*x,yl_f*y]], np.int32)
    comp_conn = get_objects_in_masked_region(img, vertices_left)
    # background is calculated as 0th value. Index from 1 onward
    comp_cent = comp_conn[3][1:,:][comp_conn[2][1:,-1] == comp_conn[2][1:,-1].max()] * (1 / factor)
    if comp_cent.size > 0: #if found connected components
        comp_cent = tuple(map(int, comp_cent.tolist()[0]))

    vertices_right = np.array(
        [[xlp_f * x, yl_f * y], [xlp_f * x, yr_f * y], [xrp_f * x, yr_f * y], [xrp_f * x, yl_f * y]], np.int32)
    play_conn = get_objects_in_masked_region(img, vertices_right)
    play_cent = play_conn[3][1:, :][play_conn[2][1:,-1] == play_conn[2][1:,-1].max()] * (1 / factor)
    if play_cent.size > 0:
        play_cent = tuple(map(int, play_cent.tolist()[0]))

    return comp_cent, play_cent

def simple_move_paddle(pong, paddle, up, down):
    '''logic to move paddle based on ball pos and paddle pos'''
    if (pong[1] - paddle[1]) > 15:
        #print('down')
        ReleaseKey(up)
        PressKey(down)
    elif (pong[1] - paddle[1]) < -15:
        #print('up')
        ReleaseKey(down)
        PressKey(up)
    else:
        #print('hold')
        ReleaseKey(down)
        ReleaseKey(up)

def dir_move_paddle(ftr_pong, paddle, up, down):
    '''logic to move paddle based on ball pos and paddle pos'''
    if (ftr_pong[1] - paddle[1]) > 50: # add random to try and get offence length of paddle
        #print('down')
        ReleaseKey(up)
        PressKey(down)
    elif (ftr_pong[1] - paddle[1]) < -50:
        #print('up')
        ReleaseKey(down)
        PressKey(up)
    else:
        #print('hold')
        ReleaseKey(down)
        ReleaseKey(up)

def get_pong_direction(pong_pos_list):
    '''
    called only if len(pong_pos_list) > 2
    :param pong_pos_list:
    :return: average direction_vector
    '''

    vector_list = []
    for i in range(len(pong_pos_list)-1):
        x1, y1 = pong_pos_list[i]
        x2, y2 = pong_pos_list[i+1]
        vect = [x2 - x1, y2 - y1]
        vector_list.append(vect)

    dir_vect = np.divide(np.sum(vector_list, axis=0),len(vector_list))
    # x1, y1 = pong_pos_list[0]
    # x2, y2 = pong_pos_list[-1]
    # dir_vect = [x2 - x1, y2 - y1]
    c = np.sqrt(np.square(dir_vect[0])+np.square(dir_vect[1]))

    return np.divide(dir_vect,c + 0.00001)

def pong_ray(pong_pos, dir_vec, l_paddle, r_paddle, boundaries, steps = 250):
    future_pts_list = []
    for i in range(steps):
        x_tmp = int(i * dir_vect[0] + pong_pos[0])
        y_tmp = int(i * dir_vect[1] + pong_pos[1])

        if y_tmp > boundaries[3]: #bottom
            y_end = int(2*boundaries[3] - y_tmp)
            x_end = x_tmp

        elif y_tmp < boundaries[2]: #top
            y_end = int(-1*y_tmp)
            x_end = x_tmp
        else:
            y_end = y_tmp

        ##stop where paddle can reach
        if x_tmp > r_paddle[0]: #right
            x_end = int(boundaries[1])
            y_end = int(pong_pos[1] + ((boundaries[1] - pong_pos[0])/dir_vec[0])*dir_vec[1])

        elif x_tmp < boundaries[0]: #left
            x_end = int(boundaries[0])
            y_end = int(pong_pos[1] + ((boundaries[0] - pong_pos[0]) / dir_vec[0]) * dir_vec[1])

        else:
            x_end = x_tmp

        end_pos = (x_end, y_end)
        future_pts_list.append(end_pos)

    return future_pts_list

def get_play_area(img, bkgrnd=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
    thresh = np.uint8((gray < bkgrnd) * 255)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--top", type=int, default=120,
                    help="top offset for capture position")
    ap.add_argument("-l", "--left", type=int, default=1081, help="left offset for capture position")
    ap.add_argument("-wx", "--width", type=int, default=700, help="width for capture position")
    ap.add_argument("-hx", "--height", type=int, default=500, help="height for capture position")
    ap.add_argument("-s", "--save-frames", required=False, default=False,
                    help="save frames as .png - will create new folder Frames. Cuts fps to 30fps")
    ap.add_argument("-f", "--show-frames", required=False, default=True,
                    help="display captured frames")
    ap.add_argument("-a", "--ai", type=int, default=3, help="ai option: 1 simple, 2 direction, 3 rebound")
    args = vars(ap.parse_args())

    save_frames = args['save_frames']
    show_frames = args['show_frames']
    show_fps = True

    if save_frames: # cuts frames to 30 fps
        if not os.path.exists('Frames'):
            os.mkdir('Frames')
    image_dir = 'Frames'
    image_prefix = 'pong_ai_'
    ai = args['ai'] # 1 for simple follow, 2 for added direction, 3 for rebound

    # Screen capture position
    pos = {"top": args['top'], "left": args['left'], "width": args['width'], "height": args['height']}
    # Keyboard scan codes
    up = 0x25 #K
    down = 0x32 #M
    start_time = time.time()
    factor = 0.5 # downscaling factor when processing
    fwd = 50

    for i in list(range(3))[::-1]:
        print(i+1)
        time.sleep(1)

    x = 1  # displays the frame rate average every 1 second
    counter = 0
    frame = 0
    fps = 0
    pong_pos_list = []
    future_pos = []
    boundaries = [0, pos['width'], 0, pos['height']] #l, r, t , b

    try:
        while True:
            frame += 1
            screen = screen_record(pos)
            processed_img = process_img(screen, factor)
            pong_pos = get_pong(processed_img, factor)
            comp_paddle, play_paddle = get_paddles(processed_img, factor)
            play_box = get_play_area(screen) # x, y, w, h

            if len(comp_paddle) > 0 and len(play_paddle) > 0 and len(play_box):
                boundaries = [comp_paddle[0], play_paddle[0], play_box[1], play_box[1]+play_box[3]] # left, right, top , bottom

            if len(pong_pos) > 0:
                pong_pos_list.append(pong_pos)
                if len(pong_pos_list) >= 2:
                    dir_vect = get_pong_direction(pong_pos_list)
                    x_end = int(fwd*dir_vect[0] + pong_pos[0])
                    y_end = int(fwd*dir_vect[1] + pong_pos[1])
                    end_pos = (x_end, y_end)
                    future_pos = pong_ray(pong_pos, dir_vect, comp_paddle, play_paddle, boundaries)
                    if show_frames and ai > 1:
                        cv2.arrowedLine(screen, pong_pos_list[-1], end_pos, (0,255,0), 2)
                        if len(future_pos) > 0 and ai > 2:
                            for i,c in enumerate(future_pos):
                                if i % 5 == 0:
                                    cv2.circle(screen, c, 1, (250, 0, 250), -1)
            if len(pong_pos_list) > 5:
                pong_pos_list.pop(0)

            if ai == 1:
                if len(pong_pos) > 0 and len(play_paddle) > 0:
                    simple_move_paddle(pong_pos, play_paddle, up, down)
            elif ai == 2:
                if len(pong_pos_list) >= 2 and len(play_paddle) > 0:
                    dir_move_paddle(end_pos, play_paddle, up, down)
            elif ai == 3:
                if len(future_pos) > 0 and len(play_paddle) > 0:
                    dir_move_paddle(future_pos[-1], play_paddle, up, down)

            counter += 1

            if (time.time() - start_time) > x:
                fps = counter // (time.time() - start_time)
                # print("FPS: ", fps)
                counter = 0
                start_time = time.time()

            if len(pong_pos) > 0:
                cv2.circle(screen, pong_pos, 10, (0, 0, 255), -1)
            if len(comp_paddle) > 0:
                cv2.circle(screen, comp_paddle, 8, (0, 0, 255), -1)
            if len(play_paddle) > 0:
                cv2.circle(screen, play_paddle, 8, (0, 255, 0), -1)
            if fps > 0 and show_fps:
                cv2.putText(screen, "{}".format(fps), (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font size
                        (209, 80, 0, 255),
                        2)  # font stroke

            if show_frames:
                cv2.imshow('window', screen)
            if save_frames:
                cv2.imwrite(os.path.join(image_dir,"{}_{}.jpg".format(image_prefix, frame)), screen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


    except KeyboardInterrupt:
        pass
