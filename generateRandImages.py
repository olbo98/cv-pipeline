import random, os, shutil

img_path = "D:/Exjobb/cv-pipeline/unlabeled_images"
label_path = "D:/Exjobb/cv-pipeline/box_2d_annotations"
copy_to_img_path = "D:/Exjobb/cv-pipeline/labeled_images"
copy_to_label_path = "D:/Exjobb/cv-pipeline/annotations"
images = os.listdir(img_path)

for i in range(0,823):
    rand_index = random.randint(0, len(images))
    img = images[rand_index]
    shutil.copyfile(img_path + "/" + img, copy_to_img_path + "/" + img)
    os.remove(img_path + "/" + img)
    label = img[:-4]
    shutil.copyfile(label_path + "/" + label + ".txt", copy_to_label_path + "/" + label + ".txt")
    os.remove(label_path + "/" + label + ".txt")
    del images[rand_index]



