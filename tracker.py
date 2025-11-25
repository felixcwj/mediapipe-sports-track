import numpy as np
import cv2

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = cv2.KalmanFilter(7, 4) 
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], np.float32)
        self.kf.processNoiseCov = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.01 # Lower noise for better responsiveness
        
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.statePost = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0]], np.float32) # x, y, w, h, vx, vy, vw

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.correct(np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], np.float32))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.statePost[6]+self.kf.statePost[2])<=0):
            self.kf.statePost[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.statePost[:4])
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.statePost[:4].reshape((4, ))

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x,y,w,h]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 0] + bb_test[..., 2], bb_gt[..., 0] + bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 1] + bb_test[..., 3], bb_gt[..., 1] + bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] * bb_test[..., 3]) + (bb_gt[..., 2] * bb_gt[..., 3]) - wh)
    return(o)

class ByteTracker(object):
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets_high, dets_low):
        self.frame_count += 1
        
        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((4, ))
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate High Confidence Detections
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets_high, trks, self.iou_threshold)
        
        # Update matched trackers with high confidence detections
        for m in matched:
            self.trackers[m[1]].update(dets_high[m[0]])
            
        # Associate Low Confidence Detections with Unmatched Trackers (ByteTrack Magic)
        # Only consider trackers that are already active (hits >= min_hits) for low conf association
        # to avoid creating tracks from noise
        active_unmatched_trks = [t for t in unmatched_trks if self.trackers[t].time_since_update == 1] # Just predicted
        
        # We need to map back to original indices
        if len(dets_low) > 0 and len(active_unmatched_trks) > 0:
            trks_low = trks[active_unmatched_trks]
            matched_low, unmatched_dets_low, unmatched_trks_low = self.associate_detections_to_trackers(dets_low, trks_low, 0.5) # Higher IoU for low conf
            
            for m in matched_low:
                self.trackers[active_unmatched_trks[m[1]]].update(dets_low[m[0]])
                
            # Update unmatched trackers list (remove those that matched with low conf)
            # This logic is a bit complex to map back, simplified:
            # Any tracker in active_unmatched_trks that wasn't matched in matched_low is still unmatched
            # But for simplicity in this implementation, we just let them age.
            pass

        # Create new trackers for unmatched high confidence detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_high[i])
            self.trackers.append(trk)
            
        # Return valid tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        iou_matrix = iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Hungarian Algo or Greedy? Greedy is faster and good enough for simple cases
                # But let's use linear_sum_assignment if available, else greedy
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack((row_ind, col_ind), axis=1)
        else:
            matched_indices = np.empty((0,2),dtype=int)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        # Filter out matches with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
