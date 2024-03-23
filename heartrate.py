
import time

class Heartrate:

    def __init__(self):
        self.start = 0
        self.val_list = []
        self.heartbeat = 0
        self.bpm = 0
        self.highest_val = 0
        self.lowest_val = 0


    def bpm_calc(self, current_val):

        self.val_list.append(current_val)
        self.val_list.sort()
        self.lowest_val = self.val_list[0]
        self.highest_val = self.val_list[-1]
        print(current_val)

        if current_val >= self.highest_val:
            self.heartbeat+=1
            self.val_list.append(current_val)

        if int(time.time()-self.start) >= 10:
            self.bpm = self.heartbeat*6
            print(f"BPM: {self.bpm}")
            self.heartbeat = 0
            self.start = time.time()

        if len(self.val_list) > 20:
            self.val_list[10:]