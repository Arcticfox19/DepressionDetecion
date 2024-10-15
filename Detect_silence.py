#这个文件用于检测并删除音频数据中的静音片段
import os
from pydub import AudioSegment
from pydub.silence import detect_silence

path = './Mydata'
for root, folders, files in os.walk(path):
    for file in files:
        filepath = os.path.join(root, file)
        print(filepath)
        if file.endswith('.wav'):
            sound = AudioSegment.from_file(filepath, format="wav")
            start_end = detect_silence(sound, 250, -35, 1)
            soundstat = 0
            soundend = len(sound)
            for o in start_end:
                if o[1] == len(sound):
                    soundend = o[0]
                if o[0] == 0:
                    soundstat = o[1]
            final_name = os.path.join(root, f"{os.path.splitext(file)[0]}_out.wav")
            sound[soundstat:soundend].export(final_name, format='wav')
