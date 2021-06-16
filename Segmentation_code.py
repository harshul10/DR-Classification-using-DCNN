import numpy as np 
import cv2 
import os



def segment(filename):
    # Image operation using thresholding 
    image= cv2.imread(filename) 
    image=cv2.resize(image,(512,512))
    #cv2.imshow('Original Image',image)


    # convert to RGB
    #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #show image
    #cv2.imshow("Gray Image", result)

    equ = cv2.equalizeHist(result)

    #cv2.imshow("Histogram Equalization", equ)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float

    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()


    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    #cv2.imshow('Semented Image',segmented_image)
    return segmented_image


folder_list = os.listdir('Dataset1')


for folder in folder_list:
        
        # create a path to the folder
        path = 'Dataset1/' + str(folder)
        path1 = 'segmented/' + str(folder)
        img_files = os.listdir(path)
        print(path)
        for file in img_files:
	
            #imgpath = ds_path +'\\'+ file
            src = os.path.join(path, file)
            #main_img = cv2.imread(src)
            img=segment(src)
            cv2.imwrite(os.path.join(path1, file),img)
    
            
