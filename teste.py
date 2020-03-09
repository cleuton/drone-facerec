from easytello import tello
import cv2,time


my_drone = tello.Tello()

## Descomente estes comandos para fazer o drone decolar:
#my_drone.takeoff()
#time.sleep(2)
my_drone.streamon()
time.sleep(5)

for i in range(4):
    ## Descomente estes comandos para fazer o drone decolar:
	#my_drone.forward(50)
    #time.sleep(1)
    ## Comente o próximo comando: 
    pass

## Descomente estes comandos para se fez o drone decolar:	
#my_drone.land
#time.sleep(2)

def process_request():
    print("Waiting video")

class SetInterval :
    def __init__(self,interval,action) :
        self.interval=interval
        self.action=action
        self.stopEvent=threading.Event()
        thread=threading.Thread(target=self.__setInterval)
        thread.start()
        #log.debug('iniciou o thread do set interval')

    def __setInterval(self) :
        nextTime=time.time()+self.interval
        #log.debug('entrou no thread do set interval')
        while not self.stopEvent.wait(nextTime-time.time()) :
            nextTime+=self.interval
            self.action()

    def cancel(self) :
        self.stopEvent.set()

interval = SetInterval(5,process_request)
