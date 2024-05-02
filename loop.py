from mask import main
from init import main as init_main
import datetime

time = datetime.datetime.now()
#run main every 15 mionutes
while True:
    print("Starting loop")
    main()
    print("Finished loop")
    print("Sleeping for 5 minutes")
    #sleep for 5 minutes
    time = datetime.datetime.now()
    while datetime.datetime.now() < time + datetime.timedelta(minutes=5):
        pass
    print("Waking up")