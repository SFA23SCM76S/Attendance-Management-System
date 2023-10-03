import tkinter as tk
import cv2,os
import numpy as np
import pickle
from PIL import Image
import csv


from datetime import datetime 

xdim = 1000
ydim = 700

Names = {}                          # key = Roll Number, Value  = Name
Blocks = {}                          # key = Roll Number, Value  = Block
Year = {}                          # key = Roll Number, Value  = Year
Dept = {}                          # key = Roll Number, Value  = Dept




reader = csv.reader(open('Data/Names.csv', 'r'))
for row in reader:
   k, v = row
   Names[int(k)] = v

reader = csv.reader(open('Data/Blocks.csv', 'r'))
for row in reader:
   k, v = row
   Blocks[int(k)] = v

reader = csv.reader(open('Data/Dept.csv', 'r'))
for row in reader:
   k, v = row
   Dept[int(k)] = v

reader = csv.reader(open('Data/Year.csv', 'r'))
for row in reader:
   k, v = row
   Year[int(k)] = v

window = tk.Tk()
window.title("Attendance Monitor")
screen_resolution = str(xdim)+'x'+str(ydim)
window.geometry(screen_resolution)

# tEXT

tbox1 = tk.Text(master = window, height = 10,width = int(xdim/20))	
tbox1.place(x=(xdim / 4), y=430, anchor="center")

tbox2 = tk.Text(master = window, height = 10,width = int(xdim/20))
tbox2.place(x=3*(xdim / 4), y=430, anchor="center")

# Functions
def Register_Me():
	cam = cv2.VideoCapture(0)
	detector=cv2.CascadeClassifier('Classifiers/face.xml')
	offset=50
	i = 0
	name=str(Roll_Entry.get())
	while True:
		ret, im = cam.read()
		if ret == True:
			gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
			while i<40:
				ret, im =cam.read()
				if ret==True:
					gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
					faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
					for(x,y,w,h) in faces:
						i=i+1
						cv2.imwrite("dataSet/"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
						cv2.rectangle(gray,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
						cv2.imshow('im',gray[y-50:y+h+50,x-50:x+w+50])
						cv2.waitKey(100)
			cam.release()
			cv2.destroyAllWindows()
			break
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	cascadePath = "Classifiers/face.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath);
	path = 'dataSet'
	#cv2.imshow('test',images[0])
	#cv2.waitKey(1)
	def get_images_and_labels(path):
		image_paths = [os.path.join(path, f) for f in os.listdir(path)]   
		images = []
		labels = []
		for image_path in image_paths:
			image_pil = Image.open(image_path).convert('L')
			image = np.array(image_pil, 'uint8')
			nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
			faces = faceCascade.detectMultiScale(image)
			for (x, y, w, h) in faces:
				images.append(image[y: y + h, x: x + w])
				labels.append(nbr)
				cv2.waitKey(10)
		return images, labels
	
	images, labels = get_images_and_labels(path)
	recognizer.train(images, np.array(labels))
	recognizer.write('trainer/trainer.yml')
	cv2.destroyAllWindows()
	str0 = "\nCapturing Face ..."
	tbox1.insert(tk.END, str0)
	str1 = str(Roll_Entry.get()) 

	Roll_No = int(Roll_Entry.get())
	Names[Roll_No] = str(Name_Entry.get())
	Blocks[Roll_No] = str(Block_Entry.get())
	Dept[Roll_No] = str(Dept_Entry.get())
	Year[Roll_No] = str(Year_Entry.get())


	path = "Data/Names.csv"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	w = csv.writer(open(path, append_write))
	for key, val in Names.items():
	    w.writerow([key, val])

	path = "Data/Blocks.csv"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	w = csv.writer(open(path, append_write))
	for key, val in Blocks.items():
	    w.writerow([key, val])


	path = "Data/Dept.csv"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	w = csv.writer(open(path, append_write))
	for key, val in Dept.items():
	    w.writerow([key, val])

	path = "Data/Year.csv"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	w = csv.writer(open(path, append_write))
	for key, val in Year.items():
	    w.writerow([key, val])

	str2 = "\nRegistered " + str1
	tbox1.insert(tk.END, str2)



def Clear_ALL():
	if os.path.exists("dataSet"):
		os.system("rm -rf dataSet")
	if os.path.exists("trainer"):
		os.system("rm -rf trainer")
	if os.path.exists("attendance"):
		os.system("rm -rf attendance")
	if os.path.exists("Data"):
		os.system("rm -rf Data")
	os.system("mkdir attendance")
	os.system("mkdir attendance/Block")
	os.system("mkdir attendance/Year")
	os.system("mkdir attendance/Dept")
	os.system("mkdir attendance/Day")
	os.system("mkdir dataSet")
	os.system("mkdir trainer")
	os.system("mkdir Data")
	Names = {}
	Blocks = {}
	Year = {}
	Dept = {}
	w = csv.writer(open("Data/Names.csv", "w"))
	for key, val in Names.items():
	    w.writerow([key, val])

	w = csv.writer(open("Data/Blocks.csv", "w"))
	for key, val in Blocks.items():
	    w.writerow([key, val])

	w = csv.writer(open("Data/Dept.csv", "w"))
	for key, val in Dept.items():
	    w.writerow([key, val])

	w = csv.writer(open("Data/Year.csv", "w"))
	for key, val in Year.items():
	    w.writerow([key, val])
	

def Mark_Attendance(Roll_Number):

	now = datetime.now()
	date = str(now).split(" ")[0]
	time =  str(now).split(" ")[1].split(":")[0] +":" + str(now).split(" ")[1].split(":")[1]
	Name = Names[Roll_Number]
	Block = Blocks[Roll_Number]
	Depts = Dept[Roll_Number]
	year = Year[Roll_Number]
	
	path = "attendance/attendance_all" + ".txt"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	ALL = open(path, append_write)
	str1 = "\n" + str(Roll_Number) + "\t" + str(Name) + "\t" + str(Block) + "\t" + str(Depts) + "\t" + str(year) + "\t" + str(date) + "\t" + str(time)   
	ALL.write(str1)
	ALL.close()

	path = "attendance/Block/attendance_" + str(Block) + ".txt"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	ALL = open(path,append_write)
	str1 = "\n" + str(Roll_Number) + "\t" + str(Name) + "\t" + str(Depts) + "\t" + str(year) + "\t" + str(date) + "\t" + str(time)
	ALL.write(str1)
	ALL.close()
	

	path = "attendance/Dept/attendance_" + str(Depts) + ".txt"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	ALL = open(path,append_write)
	str1 = "\n" + str(Roll_Number) + "\t" + str(Name) + "\t" + str(Block) + "\t" + str(year) + "\t" + str(date) + "\t" + str(time)
	ALL.write(str1)
	ALL.close()

	
	path = "attendance/Year/attendance_" + str(year) + ".txt"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	ALL = open(path,append_write)
	str1 = "\n" + str(Roll_Number) + "\t" + str(Name) + "\t" + str(Block) + "\t" + str(Depts) + "\t" + str(date) + "\t" + str(time)
	ALL.write(str1)
	ALL.close()

	path = "attendance/Day/attendance_" + str(date) + ".txt"
	if os.path.exists(path):
	    append_write = "a" # append if already exists
	else:
	    append_write = "w" # make a new file if not
	ALL = open(path, append_write)
	str1 = "\n" + str(Roll_Number) + "\t" + str(Name) + "\t" + str(Block) + "\t" + str(Depts) + "\t" + str(year) + "\t" + str(time)   
	ALL.write(str1)
	ALL.close()
		
def Recognize_Me():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read('trainer/trainer.yml')
	cascadePath = "Classifiers/face.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath);
	path = 'dataSet'
	cam = cv2.VideoCapture(0)
	while True:
		ret, im =cam.read()
		count = 0
		gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
		for(x,y,w,h) in faces:
			nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
			cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
			name = Names[nbr_predicted]
       			#cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font,1, (255, 200, 200),2)#Draw the text
        		#cv2.imshow('im',im)
        		#cv2.waitKey(10)
			f = open("Attendance-Sheet.txt", "a")
			f.write("\n" + str(nbr_predicted))
			f.close() 
			str2 = "\nAttendance Marked for " + str(name) + " " + "."
			Mark_Attendance(nbr_predicted)
			count = count + 1					
			tbox2.insert(tk.END, str2)
		if count>0:
        		cam.release()
        		cv2.destroyAllWindows()
        		break

# Label
title = tk.Label(text = "Welcome to Attendance Monitor",bd = 10,anchor="center"	, font = ("Times new roman",15))
title.place(x=(xdim/2), y=30, anchor="center")


get_Name = tk.Label(text = "Enter Your Name",font = ("Times new roman",12))
get_Name.place(x=(xdim/6)*2, y=70, anchor="center")
Name_Entry = tk.Entry()
Name_Entry.place(x=(xdim/6)*4, y=70, anchor="center")


get_number = tk.Label(text = "Enter Your Roll No.",font = ("Times new roman",12))
get_number.place(x=(xdim/6)*2, y=110, anchor="center")
Roll_Entry = tk.Entry()
Roll_Entry.place(x=(xdim/6)*4, y=110, anchor="center")

get_Block = tk.Label(text = "Enter Your Block",font = ("Times new roman",12))
get_Block.place(x=(xdim/6)*2, y=150, anchor="center")
Block_Entry = tk.Entry()
Block_Entry.place(x=(xdim/6)*4, y=150, anchor="center")

get_Dept = tk.Label(text = "Enter Your Department",font = ("Times new roman",12))
get_Dept.place(x=(xdim/6)*2, y=190, anchor="center")
Dept_Entry = tk.Entry()
Dept_Entry.place(x=(xdim/6)*4, y=190, anchor="center")

get_Year = tk.Label(text = "Enter Your Year { FY / SY / TY / BTech } ",font = ("Times new roman",12))
get_Year.place(x=(xdim/6)*2, y=230, anchor="center")
Year_Entry = tk.Entry()
Year_Entry.place(x=(xdim/6)*4, y=230, anchor="center")


# Buttons 

Register_button = tk.Button(text = "Register", command = Register_Me)
Register_button.place(x=(xdim/4), y=310, anchor="center")

Recognize_button = tk.Button(text = "Recognize", command = Recognize_Me)
Recognize_button.place(x=3*(xdim/4), y=310, anchor="center")

Clear_button = tk.Button(text = "Clear", command = Clear_ALL)
Clear_button.place(x=2*(xdim/4), y=310, anchor="center")

window.mainloop()








