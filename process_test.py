import cv2
import numpy as np

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

def get_paddles(img, factor = 0.5):
    connectivity = 8
    y,x = img.shape
    xlc_f = 0.03 #x, left computer factor
    xrc_f = 0.1
    xlp_f = 0.9 #x, left player factor
    xrp_f = 0.97
    yl_f = 0.03 # y doesn't need a distinction for comp or player
    yr_f = 0.96

    vertices_left = np.array([[xlc_f*x,yl_f*y],[xlc_f*x,yr_f*y],[xrc_f*x,yr_f*y],[xrc_f*x,yl_f*y]], np.int32)
    computer_mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(computer_mask, [vertices_left], 255)
    # now only show the area that is the mask
    computer_mask = cv2.bitwise_and(img, computer_mask)
    comp_conn = cv2.connectedComponentsWithStats(computer_mask, connectivity, cv2.CV_16U)
    # background is calculated as 0th value. Index from 1 onward
    comp_cent = comp_conn[3][1:,:][comp_conn[2][1:,-1] == comp_conn[2][1:,-1].max()] * (1 / factor)
    if comp_cent.size > 0:
        comp_cent = tuple(map(int, comp_cent.tolist()[0]))
    #print(comp_cent)
    cv2.imshow('window2', computer_mask)

    vertices_right = np.array(
        [[xlp_f * x, yl_f * y], [xlp_f * x, yr_f * y], [xrp_f * x, yr_f * y], [xrp_f * x, yl_f * y]], np.int32)
    player_mask = np.zeros_like(img)
    cv2.fillPoly(player_mask, [vertices_right], 255)
    player_mask = cv2.bitwise_and(img, player_mask)
    play_conn = cv2.connectedComponentsWithStats(player_mask, connectivity, cv2.CV_16U)
    play_cent = play_conn[3][1:, :][play_conn[2][1:,-1] == play_conn[2][1:,-1].max()] * (1 / factor)
    if play_cent.size > 0:
        play_cent = tuple(map(int, play_cent.tolist()[0]))
    #print(play_cent)
    cv2.imshow('window3', player_mask)

    return comp_cent, play_cent

def get_play_area(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
    thresh = np.uint8((gray < 10) * 255)
    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    #cv2.drawContours(img, [c], 0, (0, 255, 0), 3)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return x, y, w, h


pong_pos_list = [(500, 300)]
img = cv2.imread(r'images\test_img.png')
processed_img = process_img(img)
pong_pos = get_pong(processed_img)
pong_pos_list.append(pong_pos)
print(pong_pos_list, len(pong_pos_list))
pong_pos_list.pop(0)
print(pong_pos_list,len(pong_pos_list))
#cv2.circle(img, pong_pos, 8, (255, 0, 0), -1)

area_box = get_play_area(img)
print(area_box)

comp_paddle, play_paddle = get_paddles(processed_img)
cv2.circle(img, comp_paddle, 8, (0, 0, 255), -1)
cv2.circle(img, play_paddle, 8, (0, 255, 0), -1)

# if len(pong_pos) > 0 and len(play_paddle) > 0:
#     print(play_paddle[1], pong_pos[1])

cv2.imshow('window1', img)
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()
