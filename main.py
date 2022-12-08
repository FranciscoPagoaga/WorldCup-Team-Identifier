import cv2
import numpy as np

def main():
    img = cv2.imread('training_dataset/01200.jpg')
    hh, ww = img.shape[:2]
    img = grabCut(img)
    
    #average height 565, average width 733
    cv2.imshow('window', img)
    # cv2.imshow('window', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def resizeImg(sourceImg):
    desired_width = 733
    desired_height = 565
    dim = (desired_width, desired_height)
    resized_img = cv2.resize(sourceImg, dsize=dim, interpolation=cv2.INTER_AREA)
    return resized_img

def grabCut(img):
    img = resizeImg(img)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,733,565)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    moddedImg = img*mask2[:,:,np.newaxis]
    return moddedImg

if __name__ == "__main__":
    main()
    np