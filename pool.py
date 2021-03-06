from random import sample


class Pool():
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def get_len(self):
        return len(self._images)

    def get_sample(self, index):
        sample = ''
        label = []
        if self.get_len() > 0:
            sample = self._images[index]
            label = self._labels[index]
        return sample, label

    def get_all_samples(self):
        return self._images

    def add_sample(self, image, label):
        self._images.append(image)
        self._labels.append(label)

    def delete_sample(self, image):
        index = self._images.index(image)
        self._images.remove(image)
        del self._labels[index]
    
    def exists(self, image):
        try:
            index = self._images.index(image)
            return True
        except:
            return False

    def change_sample_path(self, index, new_path):
        self._images[index] = new_path