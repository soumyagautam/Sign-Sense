import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)           # Creating data directory, if it doesn't exist

number_of_classes = 26      # No. of letters in English
dataset_size = 100      # 100 images from each class

cap = cv2.VideoCapture(0)       # Setting the integrated camera of laptop to capture videos

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))         # Creating sub-folders with the name of class

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()     # Reading the frames
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)        # Putting the text
        cv2.imshow('frame', frame)      # Showing the frame
        if cv2.waitKey(25) == ord('q'):
            break               # Pressing 'q' to start the recording

    counter = 0
    while counter < dataset_size:       # Getting the frames till it reaches the dataset_size
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)  # Saving the frame in .jpg format

        counter += 1

cap.release()
cv2.destroyAllWindows()
