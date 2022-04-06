import os

def main():
    path = '/Users/olofbourghardt/Downloads/box_2d_annotations'
    label_files = os.listdir(path)
    image_width = 1280
    image_height = 966
    for label_file in label_files:
        new_labels = []
        with open(os.path.join(path, label_file), 'r') as f:
            for line in f:
                values = line[:-1].split(",")
                labels = values[1:]
                c = labels.pop(0)

                labels[0] = float(labels[0])/image_width
                labels[1] = float(labels[1])/image_height
                labels[2] = float(labels[2])/image_width
                labels[3] = float(labels[3])/image_height

                labels.append(float(c))
                new_labels.append(labels)

        with open(os.path.join(path, label_file), 'w') as f:
            for label in new_labels:
                for i in range(0, len(label)):
                    f.write(str(label[i]))
                    if i == len(label)-1:
                        f.write("\n")
                    else:
                        f.write(" ")

if __name__ == '__main__':
    main()