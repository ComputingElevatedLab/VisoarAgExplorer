import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2, numpy


# functions for blending operations
# takes a pixel from image 1 (pix_1) and blends it with a pixel from image 2 (pix_2)
# depending on the value given in perc (percentage);
# if perc = 0 or 255 (or 0,0,0 or 255,255,255) it will perform no blending at all
# and return the value of image 1 or image 2;
# by contrast, all values in between (e.g., 140) will give a weighted blend of the two images
# function can be used with scalars or numpy arrays (perc will be greyscale numpy array then)
def mix_pixel(pix_1, pix_2, perc):
    return (perc / 255 * pix_2) + ((255 - perc) / 255 * pix_1)


# function for blending images depending on values given in mask
def blend_images_using_mask(img_orig, img_for_overlay, img_mask):
    # turn mask into 24 bit greyscale image if necessary
    # because mix_pixel() requires numpy arrays having the same dimension
    # if image is 24-bit BGR, the image has 3 dimensions, if 8 bit greyscale 2 dimensions
    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR, cv2.CV_8U)

    # interpolate between two images (img_orig and img_to_insert)
    # using the values in img_mask (each pixel serves as individual weight)
    # as weighting factors (ranging from [0,0,0] to [255,255,255] or 0 to 100 percent);
    # because all three images are numpy arrays standard operators
    # for multiplication etc. will be applied to all values in arrays
    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)

    return img_res.astype(numpy.uint8)

# Convert ViSUS array to numpy
# pdim = input.dims.getPointDim()
# img = Array.toNumPy(input, bShareMem=True)
DEBUG = False
CV_SHOW = True
FILE_INPUT = True
PIXEL = False
if FILE_INPUT:
    dirName= '/Volumes/ViSUSAg/MAPIR/rapini_11-1-2018/S3WOCN_examples/'
    saveDir ='VisusSlamFiles/ViSOARIDX/'
    filename = '2018_1101_100723_240.JPG'

    dirName= '/Volumes/ViSUSAg/MAPIR/BodyShopE/'
    filename ='BodyShopE_2021_0427_133550_114.JPG'

    dirName= '/Volumes/ViSUSAg/MawsHill/MAPIR MawMaws Hill 4.22.21/'
    filename ='2021_0422_111630_131.JPG'

    dirName= '/Volumes/ViSUSAg/MawsHill/smallTest/'
    filename ='2021_0422_111630_131.jpg'
    img = cv2.imread(dirName+filename).astype(numpy.float32)
else:
    img = input.astype(numpy.float32)


orange = img[:, :, 0]/255
cyan = img[:, :, 1]/255
NIR = img[:, :, 2]/255

NDVI_u = (NIR - orange)
NDVI_d = (NIR + orange)
#NDVI_d[NDVI_d == 0 ] = 0.01
NDVI = NDVI_u / (NDVI_d+.0001)
#NDVI = (NDVI+1.0)/2.0
NDVI = cv2.normalize(NDVI, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalize data [0,1]

gray =  numpy.float32(NDVI)


cdictN = [ (0.56, 0.02 ,0.02), (0.74, 0.34 ,0.04), (0.94, 0.65 ,0.27), (0.2, 0.4 ,0.0), (0.2, 0.4 ,0.0),]
#cdictN = [ (0.56, 0.02 ,0.02), (0.74, 0.74 ,0.04), (0.0, 0.65 ,0.0), (0.0, 0.4 ,0.0), (0.0, 0.4 ,0.0),]
nodesN =[0.0,0.25,0.5,0.75,1.0,]



cdictN = [ (int(0.56*255), int(0.02*255) ,int(0.02*255)),
           (int(0.74*255), int(0.34*255) ,int(0.04*255)),
           (int(0.94*255), int(0.65*255) ,int(0.27*255)),
           (int(0.2*255), int(0.4*255) ,int(0.0*255)),
           (int(0.2*255), int(0.4*255) ,int(0.0*255)),]
#Red to Green
cdictN = [ (0.56, 0.02 ,0.02), (0.74, 0.34 ,0.04), (0.94, 0.65 ,0.270), (0.20, 0.4 ,0.0), (0.20, 0.4 ,0.0),]
#Green to Red
cdictN = [  (0.20, 0.4 ,0.0), (68/255, 102/255, 85/255),(41/255, 77/255, 66/255),(217/255, 207/255, 176/255), (0.56, 0.02 ,0.02), ]
nodesN =[0.0,0.5,0.75,0.9,1.0,]

f0 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, cv2.CV_8U)
f0[:, :] = (0, 0, 0)
img_blended = f0
for n in range(0, len(nodesN) - 1):
    mn = cv2.inRange(gray, nodesN[n], nodesN[n+1])

    fn = f0
    fn[:, :] = (cdictN[n+1])
    fn = (fn * 255).astype(numpy.uint8)
    # f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2BGR, cv2.CV_8U)
    if CV_SHOW:
        cv2.imshow('img {0} {1}'.format('img f1 should be  ', cdictN[n+1]), fn)
        k = cv2.waitKey(0)
    f0[:, :] = (0, 0, 0)

    if len(mn.shape) != 3:
        mn = cv2.cvtColor(mn, cv2.COLOR_GRAY2BGR, cv2.CV_8U)

    img_blended= ((mn / 255) * fn) + ((255 - mn) / 255 * img_blended)
    img_blended=img_blended.astype(numpy.uint8)
    #img_blended = blend_images_using_mask(img_blended, fn, mn)
    if CV_SHOW:
        cv2.imshow('img {0}'.format(n), img_blended)
        k = cv2.waitKey(0)


output = cv2.cvtColor(img_blended, cv2.COLOR_BGR2RGB)
if CV_SHOW:
    cv2.imshow('img {0}'.format('FINAL'), output)
    k = cv2.waitKey(0)

output = output.astype(numpy.uint8)