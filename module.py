import enum
import datetime
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
import shutil
import random
from mapcalc import calculate_map

class Module():
    """
    A class that represents the adaptive super vision module. 
    The class essentially is the entire algorithm where it samples images,
    save coordinates, annotates images to then send to the interface
    as well as pseudolabels the images.
    """
    def __init__(self, view: View, path_to_labeled_imgs, path_to_labels, path_to_unlabeled_imgs, path_to_weak_imgs, path_to_testset):
        self.i = 0
        self.set_images = []
        self.view = view
        self.circle_coords = {}
        self.rect_coords = {}
        self.shape_IDs = []
        self.active_image = ''
        self.path_to_labeled_imgs = path_to_labeled_imgs
        self.path_to_labels = path_to_labels
        self.path_to_unlabeled_imgs = path_to_unlabeled_imgs
        self.path_to_weak_imgs = path_to_weak_imgs
        self.path_to_testset = path_to_testset
        self.strong_annotations = False 
        self.unlabeled_pool, self.weak_labeled_pool, self.labeled_pool = self.load_pools(self.path_to_unlabeled_imgs, self.path_to_labels, self.path_to_labeled_imgs, self.path_to_weak_imgs)
        self.curr_episode = 1
        self.model, self.optimizer, self.loss, self.epochs, self.batch_size = self.setup_model()

    def prepare_imgs(self,set_images):
        set_images = iter(set_images)
        self.circle_coords = {}
        self.rect_coords = {}
        while True:
            try:
                image = next(set_images)
                if self.strong_annotations == False:
                    self.circle_coords[image] = []
                elif self.strong_annotations == True:
                    self.rect_coords[image] = []
            except StopIteration:
                break
    
    #Setting up the yolo model for model prediction
    def setup_model(self, training=True):
        #yolo = YoloV3()
        #yolo.load_weights('./checkpoints/yolov3.tf').expect_partial()
        epochs = 50
        batch_size = 1
        learning_rate=1e-5
        model = YoloV3(416, training=training, classes=5)
        model.load_weights("./checkpoints/yolov3_train.tf").expect_partial()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = [YoloLoss(yolo_anchors[mask], classes=5) for mask in yolo_anchor_masks]

        model.compile(optimizer=optimizer, loss=loss)


        return model, optimizer, loss, epochs, batch_size

    def load_model(self, training=False):
        if training == True:
            self.model = YoloV3(416, training=training, classes=5)
        else:
            self.model = YoloV3(classes=5)
        self.model.load_weights(f"./checkpoints/yolov3_train.tf").expect_partial()
        if training == True:
            #loss = [YoloLoss(yolo_anchors[mask], classes=80) for mask in yolo_anchor_masks]
            self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def load_pools(self, unlabeled_path, labels_path, img_path, weak_labeled_path):
        unlabeled_pool = []
        weak_labeled_pool = Pool([], [])
        labeled_pool = Pool([], [])
        for image in os.listdir(unlabeled_path):
            unlabeled_pool.append(os.path.join(unlabeled_path, image))
        random.shuffle(unlabeled_pool)
        labeled_images = os.listdir(img_path)
        for image in labeled_images:
            labels = []
            full_path = os.path.join(img_path, image)
            with open(os.path.join(labels_path, image[:-4]+'.txt'), 'r') as f:
                for line in f:
                    values = line.split(" ")
                    values = [float(v) for v in values]
                    labels.append(values)
            labeled_pool.add_sample(full_path, labels)
        
        weak_img_path = os.path.join(weak_labeled_path, 'images')
        weak_label_path = os.path.join(weak_labeled_path, 'annotations')
        weak_labeled_images = os.listdir(weak_img_path)
        for image in weak_labeled_images:
            labels = []
            full_path = os.path.join(weak_img_path, image)
            with open(os.path.join(weak_label_path, image[:-4]+'.txt'), 'r') as f:
                for line in f:
                    values = line.split(" ")
                    values = [float(v) for v in values]
                    labels.append(values)
            weak_labeled_pool.add_sample(full_path, labels)
        
        return unlabeled_pool, weak_labeled_pool, labeled_pool
    
    #Opens the image the prepocess it to a functionable size
    def prepocess_img(self, image, size=416):
        with open(image, 'rb') as i:
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
            w,h = self.view.get_width_height_img()
            self.view.draw_circle(x0, y0, x1, y1)
            self.shape_IDs.append(self.view.ID)
            self.add_circle_coords(x/w,y/h)
    
    #Button release when annotating with strong labels
    def handle_buttonrelease(self, event):
        if self.strong_annotations:
            x0,y0 = self.x, self.y
            x1,y1 = event.x, event.y
            w,h = self.view.get_width_height_img()
            self.view.draw_rectangle(x0, y0, x1, y1)
            self.shape_IDs.append(self.view.ID)
            self.add_rect_coords(x0/w,y0/h,x1/w,y1/h, self.view.active_class)

    #Sample images from dataset using a Least Confident method.
    #Confidence for an image is calculated as the highest bounding box probability in that image
    #Images with the least confidence are selected
    def active_smapling(self,set, sample_size):
        
        highest_scores = []
        i = 1
        for image in set:
            print("SAMPLING IMAGE:", str(i) + "/" + str(len(set)))
            img = self.prepocess_img(image)
            img = tf.expand_dims(img, axis=0)
            img = img/255
            _,scores, _,_ = self.model.predict(img)
            highest_scores.append(scores[0][0])
            i = i + 1

        images_and_scores = zip(set, highest_scores)
        sorted_images_and_scores = sorted(images_and_scores, key = lambda x: x[1])
        least_confident_samples = [row[0] for row in sorted_images_and_scores[0:sample_size]]
        return least_confident_samples
    
    #Queries for weak annotations
    #Drawing a circle by center-clicking on an object
    #Move on into the next images to annotate
    def query_weak_annotations(self): 
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
            box = box.tolist()
            if dist < closest_distance:
                box.append(c)
                closest_box = (box, score)
                closest_distance = dist
        return closest_box

    def pseudo_labels(self, weak_annotations):
        #predict bounding boxes
        #use weak labels to choose best possible bounding box
        #   - for every click location, we pseudo label that object with a
        #     bounding-box with center closest to the click location.
        #   - The object is classified as the class with the
        #     highest probability for the chosen bounding box
        #   - For each image we calculate the confidence score wich is the mean score
        #     which is the mean probability score obtained for each predicted object
        labels_and_confidence = []
        i = 0
        for image in weak_annotations:
            pseudo_labels = []
            img = self.prepocess_img(image)
            img = tf.expand_dims(img, axis=0)
            img = img/255
            boxes, scores, classes, _ = self.model.predict(img)
            for annotation in weak_annotations[image]:
                closest_box = self.find_closest_box(boxes[0], scores[0], classes[0], annotation)
                pseudo_labels.append(closest_box)
            confidence_score = 0
            boxes = []
            for label in pseudo_labels:
                confidence_score += label[1]
                boxes.append(label[0])
            if len(pseudo_labels) != 0:
                confidence_score = confidence_score/len(pseudo_labels)
                labels_and_confidence.append((boxes, confidence_score, i))
            else:
                os.remove(image)
            i += 1
        return labels_and_confidence

    #Queries for strong annotation
    #Strong annotate by drawing a bounding box around an object
    #Annotate by selecting the top left corner and release at the bottom right corner
    def query_strong_annotations(self):
        
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
                pseudo_high.append((samples[index], pseudo_label[0]))
            else:
                s_low.append(samples[index])
        return s_low, pseudo_high
        #TODO: Maybe we should return the updated labeled pool and weak labeled pool here instead?
        #Note: Should we delete the samples from the other pools when they are inserted to the new pools? And should that be done in active_sampling?

    def first_state(self):
        self.view.start_training_sampling()
        if self.curr_episode != 1:
            self.load_model(training=True)

        if self.labeled_pool.get_len() != 0:
            self.train_model()
        self.load_model(training=False)
        self.test_model()

        #sample from unlabeled pool and weak labeled pool
        union_set = self.unlabeled_pool
        for sample in self.weak_labeled_pool.get_all_samples():
            union_set.append(sample)
        self.samples = self.active_smapling(union_set, 300)
        #delete samples from pools
        for sample in self.samples:
            if sample in self.unlabeled_pool:
                self.unlabeled_pool.remove(sample)
            elif self.weak_labeled_pool.exists(sample):
                self.weak_labeled_pool.delete_sample(sample)
        self.second_state()

    def second_state(self):
        self.strong_annotations = False
        self.set_images = self.samples
        self.prepare_imgs(self.set_images)
        self.query_weak_annotations()

    def third_state(self):
        circle_coords = self.get_circle_coords()
        p_s = self.pseudo_labels(circle_coords)
        s_low, pseudo_high = self.soft_switch(self.samples, p_s, 0.6)
        for sample in pseudo_high:
            self.weak_labeled_pool.add_sample(sample[0], sample[1])
        self.fourth_state(s_low)
    
    def fourth_state(self, s_low):
        self.strong_annotations = True
        self.set_images = s_low
        self.prepare_imgs(self.set_images)
        self.query_strong_annotations()
        
    def save_labels(self):
        #save labeled pool
        for i in range(0, self.labeled_pool.get_len()):
            image_path, labels = self.labeled_pool.get_sample(i)
            basename = os.path.basename(image_path)
            try:
                shutil.copyfile(image_path, os.path.join(self.path_to_labeled_imgs, basename))
            except:
                continue
            os.remove(image_path)
            self.labeled_pool.change_sample_path(i, os.path.join(self.path_to_labeled_imgs, basename))
            with open(self.path_to_labels+"/"+basename[:-4]+'.txt', 'w') as f:
                for label in labels:
                    for i in range(0, len(label)):
                        f.write(str(label[i]))
                        if i == len(label)-1:
                            f.write("\n")
                        else:
                            f.write(" ")
        print("LABELS ARE SAVED")
        
        #save weak labeled pool
        for i in range(0, self.weak_labeled_pool.get_len()):
            image_path, labels = self.weak_labeled_pool.get_sample(i)
            basename = os.path.basename(image_path)
            try:
                shutil.copyfile(image_path, self.path_to_weak_imgs+'/images/' + basename)
            except:
                continue
            os.remove(image_path)
            self.weak_labeled_pool.change_sample_path(i, self.path_to_weak_imgs + '/images/' + basename)

            with open(self.path_to_weak_imgs+'/annotations/'+basename[:-4]+'.txt', 'w') as f:
                for label in labels:
                    for i in range(0, len(label)):
                        f.write(str(label[i]))
                        if i == len(label)-1:
                            f.write("\n")
                        else:
                            f.write(" ")

    def fifth_state(self):
        s_low_strong = self.get_rect_coords()
        for sample in s_low_strong:
            self.labeled_pool.add_sample(sample, s_low_strong[sample])
        self.curr_episode += 1
        self.save_labels()
        self.first_state()
        

    #Iterating through each image and showing them on the interface
    def next_img(self,event=None):
        try:
            image = next(self.set_images)
            self.active_image = image
            self.i += 1
        except StopIteration:
            self.i = 0
            if self.strong_annotations == False:
                self.third_state()
            elif self.strong_annotations == True:
                self.fifth_state()

        if self.strong_annotations == False:
            self.view.show_img(self.active_image, self.i, len(self.circle_coords))
        elif self.strong_annotations == True:
            self.view.show_img(self.active_image, self.i, len(self.rect_coords))

    def get_circle_coords(self):
        return self.circle_coords
    
    def get_rect_coords(self):
        return self.rect_coords

    def add_circle_coords(self,x,y):
        self.circle_coords[self.active_image].append([x,y])

    def add_rect_coords(self,x0,y0,x1,y1,c):
        self.rect_coords[self.active_image].append([x0,y0,x1,y1,c])
    
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
            #img = self.prepocess_img(image)
            #x_train.append(img)
            x_train.append(image)
            y_train.append(labels)
        for i in range(0, weak_labeled_pool.get_len()):
            image, labels = weak_labeled_pool.get_sample(i)
            #img = self.prepocess_img(image)
            #x_train.append(img)
            x_train.append(image)
            y_train.append(labels)
        
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        y_train = tf.ragged.constant(y_train)
        y_val = tf.ragged.constant(y_val)
        y_train = y_train.to_tensor()
        y_val = y_val.to_tensor()
        
        #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        
        #train_dataset = train_dataset.shuffle(buffer_size=512)
        #train_dataset = train_dataset.batch(batch_size)
      
        #train_dataset = train_dataset.map(lambda x, y: (self.transform_image(x,416), utils.transform_targets(y, anchors, anchor_masks, 416)))
        #train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      
        
        #val_dataset = val_dataset.batch(batch_size)
        #val_dataset = val_dataset.map(lambda x, y: (self.transform_image(x, 416), utils.transform_targets(y, anchors, anchor_masks, 416)))

        #return train_dataset, val_dataset

        train_dataset = (x_train, y_train)
        val_dataset = (x_val, y_val)
        return train_dataset, val_dataset
    
    def _train_loop(self, model, epochs, train_dataset, val_dataset, optimizer, loss, debug=False):
        if debug:
            avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
            x_train = train_dataset[0]
            y_train = train_dataset[1]
            x_val = val_dataset[0]
            y_val = val_dataset[1]
            early_stop_count = 0
            early_stop_threshold = 3
            min_epoch_loss = float('inf')
            log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = tf.summary.create_file_writer(logdir=log_dir)
            for epoch in range(1, epochs + 1):
                for (image, labels) in zip(x_train, y_train):
                    with tf.GradientTape() as tape:
                        img = self.prepocess_img(image)
                        img = self.transform_image(img, 416)
                        img = tf.expand_dims(img, 0)
                        labels = tf.expand_dims(labels, 0)
                        labels = utils.transform_targets(labels, yolo_anchors, yolo_anchor_masks, 416)
                        outputs = model(img)
                        
                        regularization_loss = tf.reduce_sum(model.losses)
                        pred_loss = []
                        for output, label, loss_fn in zip(outputs, labels, loss):
                            pred_loss.append(loss_fn(label, output))
                        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                    print("{}_train, {}, {}, {}".format(
                        epoch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)), image))

                    logging.info("{}_train, {}, {}".format(
                        epoch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_loss.update_state(total_loss)

                for (image, labels) in zip(x_val, y_val):
                    img = self.prepocess_img(image)
                    img = self.transform_image(img, 416)
                    img = tf.expand_dims(img, 0)
                    labels = tf.expand_dims(labels, 0)
                    labels = utils.transform_targets(labels, yolo_anchors, yolo_anchor_masks, 416)
                    outputs = model(img)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    print("{}_val, {}, {}, {}".format(
                        epoch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)), image))
                    logging.info("{}_val, {}, {}".format(
                        epoch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_val_loss.update_state(total_loss)
                
                print("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))
                print("EPOCH {} FINISHED".format(epoch))
                logging.info("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))

                
                with summary_writer.as_default():
                    tf.summary.scalar('epoch_train_loss_avg', avg_loss.result(), step=optimizer.iterations)
                    tf.summary.scalar('epoch_val_loss_avg', avg_val_loss.result(), step=optimizer.iterations)
                print("Previous loss: ", min_epoch_loss)
                print("Current loss: ", avg_val_loss.result().numpy())
                if avg_val_loss.result().numpy() < min_epoch_loss:
                    early_stop_count = 0
                    min_epoch_loss = avg_val_loss.result().numpy()
                    print("saving model and models weights...")
                    model.save_weights(
                    'checkpoints/yolov3_train.tf')
                    model.save('./saved_model/yolo_model')
                else:
                    early_stop_count += 1
                    print("Early stop count: ", early_stop_count)
                    if early_stop_count >= early_stop_threshold:
                        print("Early stopping...")
                        break

                avg_loss.reset_states()
                avg_val_loss.reset_states()
        else:
            
            callbacks = [
                ReduceLROnPlateau(verbose=1),
                EarlyStopping(patience=3, verbose=1),
                ModelCheckpoint('checkpoints/yolov3_train.tf',
                                verbose=1, save_weights_only=True),
                TensorBoard(log_dir='logs')
            ]

            start_time = time.time()
            history = model.fit(train_dataset, epochs=epochs,callbacks=callbacks, validation_data=val_dataset)
            end_time = time.time() - start_time
            print(f'Total Training Time: {end_time}')
            model.save('./saved_model/yolo_model')
    def train_model(self):# 
        #setup datasets
        train_dataset, val_dataset = self._setup_datasets(self.labeled_pool, self.weak_labeled_pool, yolo_anchors, yolo_anchor_masks, self.batch_size)
        self._train_loop(self.model, self.epochs, train_dataset, val_dataset, self.optimizer, self.loss, debug=True)
   
    def test_model(self):
        path_to_test_images = os.path.join(self.path_to_testset, "images2")
        path_to_test_labels = os.path.join(self.path_to_testset, "annotations2")
        test_images = os.listdir(path_to_test_images)
        gt_classes = []
        gt_boxes = []
        pred_classes = []
        pred_boxes = []
        pred_scores = []
        i = 1
        for image in test_images:
            print("TESTING IMAGE: ", str(i) + "/" + str(len(test_images)))
            path_to_image = os.path.join(path_to_test_images, image)
            img_raw = tf.image.decode_image(open(path_to_image, 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)
            img = self.transform_image(img, 416)
            with open(os.path.join(path_to_test_labels, image[:-4]+".txt"), "r") as f:
                for line in f:
                    label = line.split(" ")
                    label = [float(x) for x in label]
                    gt_classes.append(label.pop())
                    gt_boxes.append(label)
            boxes, scores, classes, _ = self.model.predict(img)
            for box, score, c in zip(boxes[0], scores[0], classes[0]):
                pred_classes.append(c)
                pred_boxes.append(box)
                pred_scores.append(score)
            i = i + 1
        gt = {
            "boxes": gt_boxes,
            "labels": gt_classes
        }

        pred = {'boxes':pred_boxes,'labels':pred_classes, 'scores':pred_scores}
        print("Calculating mAP........")
        mAP = calculate_map(gt, pred, 0.5)
        print("mAP:", mAP)
        with open("./checkpoints/mAPs.txt", "a") as f:
            f.write(str(mAP)+"\n")


    def quantize_tflite_model(self, quantization):
        #self.model = YoloV3(416, classes=80)
        #self.model.load_weights(f"./checkpoints/yolov3_train_{self.epochs}.tf").expect_partial()
        #converter = tf.lite.TFLiteConverter.from_keras_model('/saved_model')

        converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model/yolo_model')
        if quantization == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            dynamic_quantized_model = converter.convert()

            with open("./quantized_models/dynamic_quantized_model.tflite", 'wb') as f:
                f.write(dynamic_quantized_model)
            print("Dynamic range quantized model in Mb:", os.path.getsize("./quantized_models/dynamic_quantized_model.tflite") / float(2**20))

        if quantization == 'fullInt':
            def representative_dataset():
                for _ in range(250):
                    yield [tf.random.uniform(shape=[1, 416, 416, 3], minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            # Restricting supported target op specification to INT8
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # Set the input and output tensors to uint8 
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            fullInt_quantized_model = converter.convert()

            with open("./quantized_models/fullInt_quantized_model.tflite", 'wb') as f:
                f.write(fullInt_quantized_model)
            print("Full integer quantized model in Mb:", os.path.getsize("./quantized_models/fullInt_quantized_model.tflite") / float(2**20))

        if quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            float16_quantized_model = converter.convert()

            with open("./quantized_models/float16_quantized_model.tflite", 'wb') as f:
                f.write(float16_quantized_model)
            print("Float16 quantized model in Mb:", os.path.getsize("./quantized_models/float16_quantized_model.tflite") / float(2**20))

            