import os
def get_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files
files = get_files('/u/amo-d1/grad/mha361/work/Code-LMs/Data/Code/Java/')

print(len(files))