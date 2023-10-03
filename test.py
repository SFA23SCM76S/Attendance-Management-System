from datetime import datetime 
  
  
# returns current date and time 
now = datetime.now()
print("Date = ",str(now).split(" ")[0])  
print("Time = ", str(now).split(" ")[1].split(":")[0] +":" + str(now).split(" ")[1].split(":")[1] ) 
