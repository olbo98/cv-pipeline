from ast import Return
import math
from view import View

class Model():
    def __init__(self, view: View,path, circle_coords, rect_coords):
        self.view = view
        self.circle_coords = circle_coords
        self.rect_coords = rect_coords
        self.shape_IDs = []
        self.active_image = ''
        self.path = path
        self.strong_annotations = False

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
    def active_smapling(self,model, dataset, sample_size):
        highest_scores = []
        for image in dataset:
            _, scores, _, _ = model.predict(image)
            highest_scores.append(scores[0][0])
        
        images_and_scores = zip(dataset, highest_scores)
        sorted_images_and_scores = sorted(images_and_scores, key = lambda x: x[1])
        least_confident_samples = [row[0] for row in sorted_images_and_scores[0:sample_size]]
        return least_confident_samples
    
    def query_weak_annotations(self,set_images): 
        self.strong_annotations = False
        self.view.draw_weak_Annotations()
        self.set_images = iter(set_images)
        self.next_img()
        circle_coords = self.get_circle_coords()
        self.view.window.mainloop()
        print("hej")
        return circle_coords

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
                closest_box = (box, score, c)
                closest_distance = dist
        return closest_box

    def pseudo_labels(self, model, sample, weak_annotations):
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
            boxes, scores, classes, _ = model.predict(image)
            for annotation in annotations:
                closest_box = self.find_closest_box(boxes[0], scores[0], classes[0], annotation)
                pseudo_labels.append(closest_box)
            confidence_score = 0

            for label in pseudo_labels:
                confidence_score += label[1]
            confidence_score = confidence_score/len(pseudo_labels)
            labels_and_confidence.append((pseudo_labels, confidence_score, i))

        return labels_and_confidence

    def query_strong_annotations(self,set_images):
        self.strong_annotations = True
        self.view.draw_strong_Annotations()
        self.set_images = iter(set_images)
        self.next_img()
        rectangle_coords = self.get_rect_coords()
        self.view.window.mainloop()
        return rectangle_coords

    def soft_switch(self, samples, pseudo_labels, conf_thresh):
        pseudo_high = []
        s_low = []
        for pseudo_label in pseudo_labels:
            confidence = pseudo_label[1]
            index = pseudo_label[2]
            if confidence > conf_thresh:
                pseudo_high.append(pseudo_label[0])
            else:
                s_low.append(samples[index])
        s_low_strong = self.query_strong_annotations(s_low)
        return s_low_strong, pseudo_high
        #TODO: Maybe we should return the updated labeled pool and weak labeled pool here instead?
        #Note: Should we delete the samples from the other pools when they are inserted to the new pools? And should that be done in active_sampling?

    def adaptive_supervision(self, unlabeled_pool, labeled_pool, weak_labeled_pool, model, episode_num, sample_size, soft_switch_thresh):
        #s = active_smapling()
        #w_s = query_weak_annotations()
        #p_s = pseudo_labels()

        return model
    
    def next_img(self,event=None):
        try:
            image = next(self.set_images)
            self.active_image = image
        except StopIteration:
            return

        self.view.show_img(self.active_image,self.path)

    def get_circle_coords(self):
        return self.circle_coords
    
    def get_rect_coords(self):
        return self.rect_coords

    def add_circle_coords(self,x,y):
        self.circle_coords[self.active_image].append([x,y])

    def add_rect_coords(self,x0,y0,x1,y1):
        self.rect_coords[self.active_image].append([x0,y0,x1,y1])
    
      
    def delete_annotations(self,event=None):
        self.view.canvas_image.delete(self.shape_IDs.pop())
        if self.strong_annotations:
            del self.rect_coords[self.active_image][-1]
        else:
            del self.circle_coords[self.active_image][-1]


