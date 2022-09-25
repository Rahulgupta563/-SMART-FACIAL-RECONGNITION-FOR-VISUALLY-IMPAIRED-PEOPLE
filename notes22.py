import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import java.util.Scanner;
public class Sampling {
    static void collect()
    {
     Scanner obj=new Scanner(System.in);
     // Instantiating the CascadeClassifier
     String xmlFile = "E:/opencv/face_detect/haarcascade_frontalface_default.xml";
     CascadeClassifier classifier = new CascadeClassifier(xmlFile);
     // Instantiating the VideoCapture class (camera:: 0) 
     VideoCapture capture = new VideoCapture();
     Mat matrix = new Mat(); 
     //System.out.println(capture);
     int id;
     System.out.println("Enter unique id");
     id = obj.nextInt();
     int SampleN=0;
     boolean ret=true;
     while(true)
     {
      ret=capture.read(matrix);
      //Converting to gray
      Mat matrix11 = new Mat(); 
      Imgproc.cvtColor(matrix, matrix11, Imgproc.COLOR_RGB2GRAY);
      // Detecting the face in the snap
      MatOfRect faceDetections = new MatOfRect();
      classifier.detectMultiScale(matrix11, faceDetections,1.3,5);
      Rect rect_Crop=null;
      // Drawing boxes
      for (Rect rect : faceDetections.toArray())
      {
       SampleN+=1;
       Imgproc.rectangle(matrix11,new Point(rect.x, rect.y),new Point(rect.x +
                                                                      rect.width, rect.y + rect.height),new Scalar(0, 0, 255),2);
       rect_Crop = new Rect(rect.x, rect.y,rect.width,rect.height);
       Mat image_roi = new Mat(matrix11,rect_Crop);
       Imgcodecs.imwrite("E:/opencv/dataset/User."+Integer.toString(id)+"."+Integer.toString(Sample
                                                                                             N)+"."+"jpg", image_roi);
       
       }
      System.out.println(SampleN+"Completed");
      if (SampleN==100)
      break;
      
      }
     System.out.println("DataSet Collected!");

     }
    public static void main(String[] args)throws Exception {
        // TODO Auto-generated method stub
        System.loadLibrary(Core.Native_library_name);
        collect();
        }
    }
import numpy as np
import cv2
import time
import os
from PIL import Image
class SmartFaceRecognition():
    #constructor initialization
    def __init__(self):
        print("Welcome to Smart Face Recognition")
        def TrainModel(self):
            #Function to create dataset
            recognizer = cv2.face.LBPHFaceRecognizer_create();
            path="./facesData"
            def getImagesWithID(path):
                imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
                # print image_path
                #getImagesWithID(path)
                faces = []
                IDs = []
                for imagePath in imagePaths:
                    # Read the image and convert to grayscale
                    facesImg = Image.open(imagePath).convert('L')
                    faceNP = np.array(facesImg, 'uint8')
                    # Get the label of the imag
                    ID= int(os.path.split(imagePath)[-1].split(".")[1])
                    # Detect the face in the image
                    faces.append(faceNP)
                    IDs.append(ID)
                    cv2.imshow("Adding faces for traning",faceNP)
                    cv2.waitKey(10)
                    return np.array(IDs), faces
                Ids,faces = getImagesWithID(path)
                recognizer.train(faces,Ids)
                recognizer.save("./faceREC/trainingdata.yml")
                cv2.destroyAllWindows()
                print("Model Successfully Trained")
                def FaceRecognize(self):
                    #Function to recognize a face and categorize him or her as known or unknown based on data set
                    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
                    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                    rec = cv2.face.LBPHFaceRecognizer_create();
                    rec.read("./faceREC/trainingdata.yml")
                    id=0
                    font=cv2.FONT_HERSHEY_COMPLEX_SMALL
                    while 1:
                        ret, img = cap.read()
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
                        for (x,y,w,h) in faces:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                            id,conf=rec.predict(gray[y:y+h,x:x+w])
                            if id==1 or id==2 or id==3 or id==4 or id==5 or id==6 or id==7 or id==8:
                                if id==1:
                                    id="shashwat"
                                    if id==2:
                                        id="ishita"
                                        if id==3:
                                            id="Abhilasha"
                                            if id==4:
                                                id="Rohit"
                                                if id==5:
                                                    id="Rohit"
                                                    if id==6:
                                                        id="Siddarth"
                                                        if id==7:
                                                            id="Jaya"
                                                            if id==8:
                                                                id="Shivani"
                                                                if id==9:
                                                                    id="Tridib"
                                                                    if id==10:
                                                                        id=”Ayushi”
                                                                        else:
                                                                            id="Warning - the person is Unknown"
                                                                            cv2.putText(img,str(id),(x,y+h),font,1,255)
                                                                            cv2.imshow('img',img)
                                                                            print("End screen?-Y/N")
                                                                            ch=input()
                                                                            if cv2.waitKey(1) == ord('q'):
                                                                                break
                                                                            cap.release()
                                                                            cv2.destroyAllWindows()
                                                                            obj=SmartFaceRecognition()
                                                                            obj.TrainModel()
                                                                            obj.FaceRecognize()





























                    





    




        
        




