import hnswlib
import numpy as np

class ExploreRewardKnn():

    def __init__(self, vec_dim, max_elements, distance_thresh):

        self.vec_dim = vec_dim # 1080 #4320 #1000
        self.max_elements = max_elements #5000 # max
        self.distance_thresh = distance_thresh
        # all states in index
        self.all_stored_frame_vecs = []

        # Generating sample data
        #data = np.float32(np.random.random((num_elements, dim)))
        #data_labels = np.arange(num_elements)

        # Declaring index
        self.knn_index = hnswlib.Index(space = 'l2', dim = self.vec_dim) # possible options are l2, cosine or ip

        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements = self.max_elements, ef_construction = 100, M = 16)

        # Controlling the recall by setting ef:
        #p.set_ef(50) # ef should always be > k

    def set_distance_thresh(self, thresh):
        self.distance_thresh = thresh
    
    def query(self, vec):
        return self.knn_index.knn_query(vec, k = 1)

    # vec needs empty first dim (could reshape here)
    def index(self, vec):
        cur_size = len(all_stored_frame_vecs)
        if (cur_size >= self.max_elements):
            print('Knn reward index is full!')
        p.add_items(vec, np.array([cur_size]))
        all_stored_frame_vecs.append(vec)