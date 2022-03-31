from models import (
    YoloV3, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
)
from sklearn.model_selection import train_test_split
import math
from pool import Pool
from view import View
import tensorflow as tf
import numpy as np
import os
from absl import logging
import utils as utils
#import dataset as dataset
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import time
import numpy as np
#
class Module():
    """
    A class that represents the adaptive super vision module. 
    The class essentially is the entire algorithm where it samples images,
    save coordinates, annotates images to then send to the interface
    as well as pseudolabels the images.
    """
    def __init__(self, view: View, path, set_images):
        self.set_images = set_images
        self.view = view
        self.circle_coords = {}
        self.rect_coords = {}
        self.shape_IDs = []
        self.active_image = ''
        self.path= path
        self.strong_annotations = False
        self.unlabeled_pool = []
        self.labeled_pool = Pool()
        self.weak_labeled_pool = Pool()
        self.model = self.setup_model()

    def prepare_imgs(self,set_images):
        set_images = iter(set_images)
        while True:
            try:
                image = next(set_images)
                self.circle_coords[image] = []
            except StopIteration:
                break

    #Setting up the yolo model for model prediction
    def setup_model(self):
        #yolo = YoloV3()
        #yolo.load_weights('./checkpoints/yolov3.tf').expect_partial()
        model = YoloV3(416, training=True, classes=80)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = [YoloLoss(anchors[mask], classes=80)
            for mask in anchor_masks]

        model.compile(optimizer=optimizer, loss=loss)

        return model, optimizer, loss, anchors, anchor_masks
    
    #Opens the image the prepocess it to a functionable size
    def prepocess_img(self, image, size=416):
        with open(os.path.join(self.path,image), 'rb') as i:
            img_raw =  tf.image.decode_image(i.read(), channels=3)
            #img = tf.expand_dims(img_raw, 0)
            #img = self.transform_image(img_raw, size)
            img = tf.image.resize(img_raw, (size,size))
            return img
    

    def transform_image(self, image, size=416):
        img = tf.image.resize(image, (size, size))
        img = img / 255
        return img

    #Handles button press from the interface when annotating an image
    #The user can draw a circle or a rectangle depending if it is a
    #strong or weak annotation
    def handle_buttonpress(self, event):
        if self.strong_annotations:
            self.x = event.x
            self.y = event.y
        else:
            x,y = event.x, event.y
            r = 8
            x0 = x - r
            y0 = y - r
            x1 = x + r
            y1 = y + r
            self.view.draw_circle(x0,y0,x1,y1)
            self.shape_IDs.append(self.view.ID)
            self.add_circle_coords(x,y)
    
    #Button release when annotating with strong labels
    def handle_buttonrelease(self, event):
        if self.strong_annotations:
            x0,y0 = self.x, self.y
            x1,y1 = event.x, event.y
            self.view.draw_rectangle(x0,y0,x1,y1)
            self.shape_IDs.append(self.view.ID)
            self.add_rect_coords(x0,y0,x1,y1)

    #Sample images from dataset using a Least Confident method.
    #Confidence for an image is calculated as the highest bounding box probability in that image
    #Images with the least confidence are selected
    def active_smapling(self,model, set, sample_size):
        highest_scores = []
        for image in set:
            full_img_path = os.path.join(self.path, image)
            img = self.prepocess_img(full_img_path)
            _, scores, _, _ = model.predict(img)
            highest_scores.append(scores[0][0])
        
        images_and_scores = zip(set, highest_scores)
        sorted_images_and_scores = sorted(images_and_scores, key = lambda x: x[1])
        least_confident_samples = [row[0] for row in sorted_images_and_scores[0:sample_size]]
        return least_confident_samples
    
    #Queries for weak annotations
    #Drawing a circle by center-clicking on an object
    #Move on into the next images to annotate
    def query_weak_annotations(self): 
        self.strong_annotations = False
        self.view.draw_weak_Annotations()
        self.set_images = iter(self.set_images)
        self.next_img()

   #Calculates distance from the bounding box's center to the position of the weak annotation
    def dist_from_point(self, box, weak_annotation):
        box_x1 = box[0]
        box_x2 = box[2]
        box_y1 = box[1]
        box_y2 = box[3]
        box_width = box_x2 - box_x1
        box_height = box_y2 - box_y1
        box_center = (box_x1 + box_width/2.0, box_y1 + box_height/2.0)
        dist = math.sqrt(math.pow(box_center[0] - weak_annotation[0], 2) + math.pow(box_center[1] - weak_annotation[1], 2))
        return dist

    #finds the bounding box with its center closest to the weak annotation
    def find_closest_box(self, bounding_boxes, scores, classes, annotation):
        closest_box = ()
        closest_distance = float('inf')
        for box, score, c in zip(bounding_boxes, scores, classes):
            dist = self.dist_from_point(box, annotation)
            if dist < closest_distance:
                closest_box = (box.append(c), score)
                closest_distance = dist
        return closest_box

    def pseudo_labels(self, sample, weak_annotations):
        #predict bounding boxes
        #use weak labels to choose best possible bounding box
        #   - for every click location, we pseudo label that object with a
        #     bounding-box with center closest to the click location.
        #   - The object is classified as the class with the
        #     highest probability for the chosen bounding box
        #   - For each image we calculate the confidence score wich is the mean score
        #     which is the mean probability score obtained for each predicted object
        labels_and_confidence = []
        for i, (image, annotations) in enumerate(zip(sample, weak_annotations)):
            pseudo_labels = []
            img = self.prepocess_img(image)
            boxes, scores, classes, _ = self.model.predict(img)
            for annotation in annotations:
                closest_box = self.find_closest_box(boxes[0], scores[0], classes[0], annotation)
                pseudo_labels.append(closest_box)
            confidence_score = 0

            for label in pseudo_labels:
                confidence_score += label[1]
            confidence_score = confidence_score/len(pseudo_labels)
            labels_and_confidence.append((pseudo_labels, confidence_score, i))

        return labels_and_confidence

    #Queries for strong annotation
    #Strong annotate by drawing a bounding box around an object
    #Annotate by selecting the top left corner and release at the bottom right corner
    def query_strong_annotations(self):
        self.strong_annotations = True
        self.view.draw_strong_Annotations()
        self.set_images = iter(self.set_images)
        self.next_img()

    def soft_switch(self, samples, pseudo_labels, conf_thresh):
        pseudo_high = []
        s_low = []
        for pseudo_label in pseudo_labels:
            confidence = pseudo_label[1]
            index = pseudo_label[2]
            if confidence > conf_thresh:
                pseudo_high.append((samples[pseudo_label[2]], pseudo_label[0]))
            else:
                s_low.append(samples[index])
        return s_low, pseudo_high
        #TODO: Maybe we should return the updated labeled pool and weak labeled pool here instead?
        #Note: Should we delete the samples from the other pools when they are inserted to the new pools? And should that be done in active_sampling?

    def adaptive_supervision(self, unlabeled_pool, labeled_pool: Pool, weak_labeled_pool: Pool, model, episode_num, sample_size, soft_switch_thresh):
        #sample from unlabeled pool and weak labeled pool
        union_set = unlabeled_pool.append(weak_labeled_pool.get_all_samples())
        s = self.active_smapling(model, union_set, 10)
        #delete samples from pools
        for sample in s:
            if sample in unlabeled_pool:
                unlabeled_pool.remove(sample)
            elif weak_labeled_pool.exists(sample):
                weak_labeled_pool.delete_sample(sample)
        
        w_s = self.query_weak_annotations()
        p_s = self.pseudo_labels()
        s_low_strong, pseudo_high = self.soft_switch(s, p_s, 1)
        #update pools with new samples
        for sample in s_low_strong:
            self.labeled_pool.add_sample(sample, s_low_strong[sample])
        for sample in pseudo_high:
            weak_labeled_pool.add_sample(sample[0], sample[1])

        return model

    def first_state(self):
        self.model_train()
        #sample from unlabeled pool and weak labeled pool
        union_set = self.unlabeled_pool.append(self.weak_labeled_pool.get_all_samples())
        self.samples = self.active_smapling(self.model, union_set, 10)
        #delete samples from pools
        for sample in self.samples:
            if sample in self.unlabeled_pool:
                self.unlabeled_pool.remove(sample)
            elif self.weak_labeled_pool.exists(sample):
                self.weak_labeled_pool.delete_sample(sample)
        self.second_state(self.samples)

    def second_state(self,samples):
        self.prepare_imgs(samples)
        self.set_images = samples
        self.query_weak_annotations()

    def third_state(self):
        circle_coords = self.get_circle_coords()
        p_s = self.pseudo_labels(self.samples, circle_coords)
        s_low, pseudo_high = self.soft_switch(self.samples, p_s)
        for sample in pseudo_high:
            self.weak_labeled_pool.add_sample(sample[0], sample[1])
        self.fourth_state(s_low)
    
    def fourth_state(self, s_low):
        self.set_images = s_low
        self.query_strong_annotations()
        

    def fifth_state(self):
        s_low_strong = self.get_rect_coords()
        for sample in s_low_strong:
            self.labeled_pool.add_sample(sample, s_low_strong[sample])
        self.first_state()
        

    def model_train(self):
        return
    #Iterating through each image and showing them on the interface
    def next_img(self,event=None):
        try:
            image = next(self.set_images)
            self.active_image = image
        except StopIteration:
            if self.strong_annotations == False:
                self.third_state
            elif self.strong_annotations == True:
                self.fifth_state

        self.view.show_img(self.active_image,self.path)

    def get_circle_coords(self):
        return self.circle_coords
    
    def get_rect_coords(self):
        return self.rect_coords

    def add_circle_coords(self,x,y):
        self.circle_coords[self.active_image].append([x,y])

    def add_rect_coords(self,x0,y0,x1,y1):
        self.rect_coords[self.active_image].append([x0,y0,x1,y1])
    
    #Deleting the latest annotations from the list and the interface  
    def delete_annotations(self,event=None):
        self.view.canvas_image.delete(self.shape_IDs.pop())
        if self.strong_annotations:
            del self.rect_coords[self.active_image][-1]
        else:
            del self.circle_coords[self.active_image][-1]

    def _setup_datasets(self,labeled_pool: Pool, weak_labeled_pool: Pool, anchors, anchor_masks, batch_size):
        x_train = []
        y_train = []
        for i in range(0, labeled_pool.get_len()):
            image, labels = labeled_pool.get_sample(i)
            img = self.prepocess_img(image)
            x_train.append(img)
            y_train.append(labels)
        for i in range(0, weak_labeled_pool.get_len()):
            image, labels = weak_labeled_pool.get_sample(i)
            img = self.prepocess_img(image)
            x_train.append(img)
            y_train.append(labels)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33)
        
        y_train = tf.ragged.constant(y_train)
        y_val = tf.ragged.constant(y_val)
        y_train = y_train.to_tensor()
        y_val = y_val.to_tensor()
        

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(batch_size)
      
        train_dataset = train_dataset.map(lambda x, y: (self.transform_image(x,416), utils.transform_targets(y, anchors, anchor_masks, 416)))
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      
        
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.map(lambda x, y: (self.transform_image(x, 416), utils.transform_targets(y, anchors, anchor_masks, 416)))

        return train_dataset, val_dataset
    
    def _train_loop(self, model, epochs, train_dataset, val_dataset, optimizer, loss, debug=False):
        if debug:
            avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

            for epoch in range(1, epochs + 1):
                for batch, (images, labels) in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        outputs = model(images)
                        regularization_loss = tf.reduce_sum(model.losses)
                        pred_loss = []
                        for output, label, loss_fn in zip(outputs, labels, loss):
                            pred_loss.append(loss_fn(label, output))
                        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                    
                    
                    logging.info("{}_train_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_loss.update_state(total_loss)

                for batch, (images, labels) in enumerate(val_dataset):
                    outputs = model(images)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    logging.info("{}_val_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_val_loss.update_state(total_loss)

                logging.info("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))

                avg_loss.reset_states()
                avg_val_loss.reset_states()
                model.save_weights(
                    'checkpoints/yolov3_train_{}.tf'.format(epoch))
        else:

            callbacks = [
                ReduceLROnPlateau(verbose=1),
                EarlyStopping(patience=3, verbose=1),
                ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                                verbose=1, save_weights_only=True),
                TensorBoard(log_dir='logs')
            ]

            start_time = time.time()
            history = model.fit(train_dataset, epochs=epochs,callbacks=callbacks, validation_data=val_dataset)
            end_time = time.time() - start_time
            print(f'Total Training Time: {end_time}')

    def train_model(self, model, labeled_pool: Pool, weak_labeled_pool: Pool, epochs, optimizer, loss, anchors, anchor_masks, batch_size):# 
        #setup datasets
        train_dataset, val_dataset = self._setup_datasets(labeled_pool, weak_labeled_pool, anchors, anchor_masks, batch_size)
        self._train_loop(model, epochs, train_dataset, val_dataset, optimizer, loss)
        
    def test_train(self):
        
        weak_labeled_pool = Pool([], [])
        labeled_pool = Pool([], [])
        img_path = "D:/Voi/test_loop_imgs/20imgs"
        label_path = "D:/Voi/test_loop_imgs/labels20imgs"
        file_names = os.listdir(img_path)
        for file in file_names:
            image_labels = []
            with open(os.path.join(label_path, file[:-4]+".txt"), "r") as f:
                for line in f:
                    line = line[:-1]
                    line = line.split(" ")
                    c = line.pop(0)
                    line.append(c)
                    for i in range(0, len(line)):
                        line[i] = float(line[i])
                    x1 = line[0] - (line[2]/2) #x1 = x - w/2
                    y1 = line[1] - (line[3]/2) #y1 = y - h/2
                    x2 = line[0] + (line[2]/2) #x2 = x + w/2
                    y2 = line[1] + (line[3]/2) #y2 = y + h/2
                    line[0] = x1
                    line[1] = y1
                    line[2] = x2
                    line[3] = y2
                    image_labels.append(line)
            labeled_pool.add_sample(file, image_labels)
            
        model, optimizer, loss, anchors, anchor_masks = self.setup_model()
        self.train_model(model, labeled_pool, weak_labeled_pool, 10, optimizer, loss, anchors, anchor_masks, batch_size = 13)
