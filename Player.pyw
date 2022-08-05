import os
import time
# On music
import winsound


file = open("SongNumbers.txt","r")
a = int(file.read())
melodyes = os.listdir("PianoMusic")

winsound.PlaySound(f'PianoMusic/{melodyes[a]}', winsound.SND_FILENAME)

file.write(" ")

file.close()

