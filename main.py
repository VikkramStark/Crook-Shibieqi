import cv2 
import streamlit as st 
import pickle
import face_recognition  
import numpy as np 
import cvzone 
import pandas as pd 
import os 
import geocoder 


st.set_page_config(
    page_title="Real-Time Face Recognition",
    page_icon="🤖",
    layout="wide",  
)

st.session_state["run"] = False 


st.sidebar.title("Crook Shibieqi") 

with st.sidebar:
    st.markdown("---") 
    st.write("# Team Members") 
    st.write("* ### Kaleeshwari") 
    st.write("* ### Mathumitha")    

cap = cv2.VideoCapture(0) 
cap.set(3,1280) 
cap.set(4,720)


#Load Encodings File 
f = open("recognition_data.pkl", "rb") 
encodings, ids = pickle.load(f)    
f.close() 
print("Encodings Loaded SUccessfully!") 

print(ids) 

with st.sidebar:
    pass 


def start_stop():
    st.session_state["run"] = not st.session_state["run"] 

# st.write("## Crook Shibieqi") 
# st.markdown("---") 
st.markdown("# Face Detection and Recognition ")     
st.markdown("---")    

# start_stop = st.checkbox("Start Recognition") 
# control_btn = st.button("Start / Stop", on_click=start_stop)       
# st.markdown("***") 
name = st.empty()
ipl = st.empty()  
st.text("") 
col1, col2 = st.columns(2) 

with col1:
    image_placeholder = st.empty() 

with col2: 
    res_image = st.empty() 

 



# mappings = pd.read_excel("mappings.xlsx", index_col = 0) 
mappings = pd.read_csv("mappings.csv", header = 0, sep = "\t")   

print(mappings) 
     

while True:     
    success, img = cap.read()

    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    img_small = cv2.resize(img,(0,0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  
    face_cur_location = face_recognition.face_locations(img_small) 
    face_cur_enc = face_recognition.face_encodings(img_small, face_cur_location) 

    for encodeFace, faceLoc in zip(face_cur_enc, face_cur_location): 
        matches = face_recognition.compare_faces(encodings,encodeFace)  
        distance = face_recognition.face_distance(encodings, encodeFace)  
        # print("Matches: ", matches) 
        # print("Distance", distance) 

        match_index = np.argmin(distance) 

        if matches[match_index] == True: 
            face_id = ids[match_index] 
            # mappings.set_index("ID", inplace = True)  
            face_name =  mappings[mappings["ID"] == int(face_id)]["NAME"].iloc[0]  # [match_index]["Name"] 
            name.write(f"## Recognized: {face_name}")  
            res = cv2.cvtColor(cv2.imread(os.path.join("images","valid_images", face_id+".jpg")), cv2.COLOR_BGR2RGB)  #"./images/valid_images/" +face_id+".jpg"
            res_image.image(res)             
            print("Known Face Detected!",face_name) 
            g = geocoder.ip("me") 
            latitude, longitude = g.latlng      
            ipl.write(f"### Latitude: {latitude} - Logitude: {longitude}")   
            y1,x2,y2,x1 = faceLoc 
            x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4 # Resizing because we scaled down image size by 1/4
            bbox = x1, y1, x2-x1, y2-y1 
            img = cvzone.cornerRect(img,bbox,rt = 0) 
        else:
            name.write("## No Known Faces Found") 
            res_image.empty() 


    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
    image_placeholder.image(rgb_img, use_column_width=True)   

    cv2.imshow("Detection",img) 

        

    if cv2.waitKey(1) == ord("q"):
        break   