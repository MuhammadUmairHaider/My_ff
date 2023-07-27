import os
import random
import shutil


def get_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files

files = get_files('/u/amo-d1/grad/mha361/work/My_ff/java_5000_polycoder')

random.seed(0xdead)
random.shuffle(files)

files = files[:20]

for file in files:
    dest = file.split('/')
    dest = dest[len(dest)-1]

    shutil.copyfile(file, "/u/amo-d1/grad/mha361/work/My_ff/java_5000_polycoder_20rand/"+dest)