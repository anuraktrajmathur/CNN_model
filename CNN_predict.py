from keras.models import load_model
import os 
import cv2
import numpy as np
path = os.path.dirname(__file__)
Test_folder_path = path + '\\GTSRB\\Test\\'
wait_time = 500  # Delay in milliseconds

# Load model
model = load_model(os.path.join(path, 'CNN_model.keras'))
# Open the camera
cap = cv2.VideoCapture(0)
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    # Display the image
    cv2.imshow('Video', frame)
    if(cv2.waitKey(wait_time) & 0xFF == ord('b')):
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (30,30))

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    
    # Define the range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth out the noise
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply Canny edge detection to detect edges
    edges_img = cv2.Canny(blur_img, 50, 150)

    # Apply morphological closing to fill in any gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(closed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")


    # Loop over the contours to find the speed limit sign
    for cnt in contours:
        # Get the area and perimeter of the contour
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Compute the circularity of the contour
        if(perimeter==0):continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        print(f"Contour area: {area:.2f}, perimeter: {perimeter:.2f}, circularity: {circularity:.2f}")

        # If the contour is circular enough and its area is within a certain range, it's likely a speed limit sign
        if circularity > 0.8 and area > 500 : #and area < 5000
            # Draw a bounding box around the sign
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"Bounding box w: {w}, h: {h}, area: {area}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the ROI of the sign
            sign_roi = resized_frame[y:y + h, x:x + w]
            sign_roi = sign_roi.reshape(1, 30, 30 ,3) 
            print("Predicting class for detected contour...") 
            result = model.predict(sign_roi)   
            class_pred = np.argmax(result,axis=1)
            # print(class_pred[0])
            if class_pred[0] == 0:
                print("20 Km/h")
            elif class_pred[0] == 1:
                print("30 Km/h")
            elif class_pred[0] == 2:
                print("50 Km/h")
            elif class_pred[0] == 3:
                print("60 Km/h")
            elif class_pred[0] == 4:
                print("70 Km/h")
            elif class_pred[0] == 5:
                print("80 Km/h")   
            elif class_pred[0] == 6:
                print("Not a speed sign")
            elif class_pred[0] == 7:
                print("100")
            elif class_pred[0] == 8:
                print("120") 


    # # Extract the ROI of the sign
    # # sign_roi = frame[y:y + h, x:x + w]
    # sign_roi = cv2.resize(frame , (30, 30))
    # sign_roi = sign_roi.reshape(1, 30, 30 ,3)  
    # result = model.predict(sign_roi)   
    # class_pred = np.argmax(result,axis=1)
    # # print(class_pred[0])
    # if class_pred[0] == 0:
    #     print("20 Km/h")
    # elif class_pred[0] == 1:
    #     print("30 Km/h")
    # elif class_pred[0] == 2:
    #     print("50 Km/h")
    # elif class_pred[0] == 3:
    #     print("60 Km/h")
    # elif class_pred[0] == 4:
    #     print("70 Km/h")
    # elif class_pred[0] == 5:
    #     print("80 Km/h")   
    # elif class_pred[0] == 6:
    #     print("Not a speed sign")
    # elif class_pred[0] == 7:
    #     print("100")
    # elif class_pred[0] == 8:
    #     print("120") 



# # Input data
# f = os.path.join(Test_folder_path,"00093.png")

# # Load the image
# img = cv2.imread(f)# Data preprocessing
# # Resize the image
# resized_img = cv2.resize(img , (30, 30))
# # Reshape array
# img = resized_img.reshape(1, 30, 30 ,3)
# use the model for classification
# result = model.predict(img)   
# class_pred = np.argmax(result,axis=1)
# print(class_pred)