import cv2
import numpy as np
import time

CHARS = " .:-=+*#%@" #characters to use 
COLS = 120 # number of columss
CELL_W, CELL_H = 8, 14  # size of the pixel in each cell

#size of the window will be the number of colums * width of the cells

#builds a stamp for each character but i didnt understanded very much, it has the thickness of lines, smooth and font
glyphs = np.zeros((len(CHARS), CELL_H, CELL_W), dtype=np.float32)
for i, ch in enumerate(CHARS):
    img = np.zeros((CELL_H, CELL_W), dtype=np.uint8)
    cv2.putText(img, ch, (0, CELL_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA) #draws the characters
    glyphs[i] = img / 255.0


#camera settings and sizes and webcam index
def open_camera(index=0, width=640, height=480, retries=5):
    for attempt in range(retries):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF) 
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(60):  
                read, frame = cap.read() #reads up to 60 frames until one is successfull
                if read and frame.mean() > 5:
                    return cap
        cap.release()
        print(f"number of at: {attempt + 1} failed, retry") #if the read fail it will release and try again after 1 sec
        time.sleep(1)
#turns the camere into ascii
def to_color_ascii(frame):
    h, w = frame.shape[:2]
    rows = int(COLS * CELL_W * h / (w * CELL_H))# gets teh frames height and width and counts how many characters rows are needed
    small = cv2.resize(frame, (COLS, rows), interpolation=cv2.INTER_AREA) # one pixel = one character

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) # brightness of each pixel
    idx = (gray / 255 * (len(CHARS) - 1)).astype(int) # converts brightness into character number

    masks = glyphs[idx] # looks for number in idk it looks for a matching character in glyphs i think
    colors = small[:, :, None, None, :].astype(np.float32)  #takes the color of the pixel and reshape it, the nones add dummy axes of size 1
    out = masks[..., None] * colors                       

    out = out.transpose(0, 2, 1, 3, 4).reshape(rows * CELL_H, COLS * CELL_W, 3) 
    return out.astype(np.uint8)

cap = open_camera() # loop

#opens the camera with the retry logic
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("ASCII camera", to_color_ascii(frame))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): #key to close the window and not the X 
        break

cap.release()
cv2.destroyAllWindows()