# Helper
import cv2
def showImage(img1, img2):
    cv2.imshow('image',img1)
    cv2.imshow('image2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




img = cv2.imread('/home/xinyang/Documents/roadSeg/data/data_road/training/processed_image/processed_image/um_000000.png')
msk = cv2.imread('/home/xinyang/Documents/roadSeg/data/data_road/training/processed_mask/processed masks/umm_road_000000.png',0)
