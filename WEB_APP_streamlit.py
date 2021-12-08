from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import cvlib as cv
import streamlit as st
from imutils.video import VideoStream
import imutils
import smtplib


st.title("Gender & Facemask Detection in Real Time Video Stream")

model = load_model('C:\BAN_676\Group_Project\Gender\gender_detection.model')
st.header('Loading the Gender Detector CNN model...')
st.subheader('Model Loaded: gender_detection.model')

# opening webcam
webcam = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    
    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        gender=label[0]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    
        break
    
    cv2.waitKey(7000)
    webcam.release()
    cv2.destroyAllWindows()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
    if gender == 'm':
        st.subheader("Hello Sir!!")
        msg = 'Hello Sir!!'
    else:
        st.subheader('Hello Madam!!')
        msg = 'Hello Madam!!'
        
    break

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))


    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"C:\BAN_676\Group_Project\Facemask\deploy.prototxt"
weightsPath = r"C:\BAN_676\Group_Project\Facemask\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("C:\BAN_676\Group_Project\Facemask\mask_detector.model")

st.title("Face-Mask Detection...")
st.header('Loading the Facemask Detector CNN model...')
st.subheader('Model Loaded: mask_detector.model')

flag = 0
count=0

# initialize the video stream
st.subheader("starting video stream...")
#run2=st.checkbox('Run')
FRAME_WINDOW = st.image([])
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
   
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # display the label and bounding box rectangle on the output
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) 
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        verify_no_mask=label[0]
        if verify_no_mask == "N":
            
            mask = 'Mask is mandatory, please put on the mask'''
            msg_mask = msg +' '+mask
            if count==0:
                st.error(msg_mask)
                msg_mask=''
                cv2.waitKey(2000)
                count=1
            if flag == 0:
                server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                server.login("dp_project@gmail.com", "dp_project@2021")
                message = 'Subject: {}\n\n{}'.format('Mask Detection', 'Administrator!! No Mask Detected')
                server.sendmail("dp_project@gmail.com", "xyz@gmail.com",message)
                st.info('Mail sent to the administrator...')
                flag = 1
                server.quit()
                
        else:
            go='You are good to go'
            go_msg = msg +' '+go      
            if(count==1):
                st.success(go_msg)
                go_msg=""
                cv2.waitKey(7000)
                count=0
                verify_no_mask="N"
                
            continue
        st.empty()
        
    # show the output frame
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

else:
    st.write('stopped')

