from os import listdir
from os.path import isfile, join

my_path = 'D:/projects_python/_datasets/VOCdevkit_2012_test/VOC2012/Annotations/'
only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

only_files = [f.replace('.xml', '') for f in only_files]
print(only_files)

test_file = open('D:/projects_python/_datasets/VOCdevkit_2012_test/VOC2012/ImageSets/Main/test.txt', 'r')
lines = test_file.readlines()

for line in lines:
    if line.rstrip() in only_files: print(line)