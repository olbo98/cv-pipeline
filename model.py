from view import View

class Model():
    def __init__(self, view: View,set_images,path):
        self.view = view
        self.circle_coords = {}
        self.set_images = iter(set_images)
        self.active_image = ''
        self.path = path
        self.prepare_imgs()
        self.set_images = iter(set_images)


    def active_smapling():
        return

    def query_weak_annotations(self,set_images): 
        return

    def pseudo_labels():
        return

    def query_annotations():
        return

    def adaptive_supervision(unlabeled_pool, labeled_pool, weak_labeled_pool, model, episode_num, sample_size, soft_switch_thresh):
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

    def add_circle_coords(self,x,y):
        self.circle_coords[self.active_image].append([x,y])

    def prepare_imgs(self):
         while True:
            try:
                image = next(self.set_images)
                self.circle_coords[image] = []
            except StopIteration:
                break



