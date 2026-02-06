import cv2

face_cascade = cv2.CascadeClassifier("haarcascades\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades\\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascades\\haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frames = cap.read()
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
    for (x,y,w,h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 10)
        if len(eyes)>0:
            cv2.putText(frames, "Eyes Detected", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 255), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        if len(smiles)>0:
            cv2.putText(frames, "Smiling", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        
    cv2.imshow("Video", frames)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
