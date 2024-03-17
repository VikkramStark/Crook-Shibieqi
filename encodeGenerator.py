import cv2 
import face_recognition 
import pickle 
import os 
import pandas as pd 
import numpy as np 

imageFolder = os.path.join("images","valid_images")  
imagesList = []
idsList = [] 
image_files = os.listdir(imageFolder)
print(image_files)   

for file in image_files:
    raw_image = np.array(cv2.imread(os.path.join(imageFolder,file))) 
    image_name = file.split(".")[0] 
    imagesList.append(raw_image) 
    idsList.append(image_name)     


# df = pd.DataFrame(imageList,columns = ['raw_format','image_name']) 
# print(df)   
# print(df.loc[0]["raw_format"].shape)  


def findEncodings(imagesList):
    encodings = [] 

    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        encode = face_recognition.face_encodings(img)[0]  
        encodings.append(encode) 

    return encodings

encodings = findEncodings(imagesList) 

print(encodings) 

# df = pd.DataFrame(data = {"Id":np.array(idsList), "image":np.array(imagesList), "encoding":np.array(encodings)})  

# df.to_csv("recognition_data.csv", index = False) 
encodings_with_id = [encodings, idsList] 

with open("recognition_data.pkl","wb") as f: 
    pickle.dump(encodings_with_id,f)    
