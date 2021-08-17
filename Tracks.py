import os
from shapely.geometry import Polygon
import cv2
import pickle
class track:
    def __init__(self):
        self.cat_id = -1

        self.recordingId = -1
        self.trackId = -1
        self.frame = -1
        self.trackLifetime = -1
        
        self.polygon = None
        self.xCenter = -1
        self.yCenter = -1
        self.heading = -1
        self.width = -1
        self.length = -1

        self.xVelocity = -1
        self.yVelocity = -1
        self.xAcceleration = -1
        self.yAcceleration = -1
        
        self.lonVelocity = -1
        self.latVelocity = -1
        self.lonAcceleration = -1
        self.latAcceleration = -1

class tracks_manager:
    def __init__(self,cat_ids=[0,1,2,5,6]):
        self.id_count = 0
        self.live_ids = []
        self.dead_ids = []
        self.track_queue = dict()

        self.cat_ids = cat_ids

        self.scale_map = 0.1 #meter per pixel

        self.frame_rate = 30

    def get_scale(self,x,y):
        return self.scale_map

    def create_new_id(self):
        self.id_count+=1
        return self.id_count
    
    def det2polygon(self,det_polygon):
        polygon = []
        #print(det_polygon)
        for i in range(4):
            polygon.append((det_polygon[i*2],det_polygon[i*2+1]))
        return Polygon(polygon)
    def update(self,det_polygons,frame_id):
        for cat_id in self.cat_ids:
            for det_polygon in det_polygons:
                if not det_polygon[-1]==cat_id:
                    continue
                max_iou = 0
                max_iou_id = -1
                p1 = self.det2polygon(det_polygon)
                for live_id in self.live_ids:
                    latest_track = self.track_queue[live_id][-1]
                    p2 = self.det2polygon(latest_track.polygon)
                    if p1.intersects(p2)>max_iou and det_polygon[-1]==latest_track.cat_id:
                        max_iou = p1.intersects(p2)
                        max_iou_id = live_id
                
                new_track = track()
                new_track.cat_id = cat_id
                new_track.frame = frame_id
                new_track.polygon = det_polygon
                center = p1.centroid.coords[0]
                new_track.xCenter = center[0]
                new_track.yCenter = center[1]

                if max_iou_id>-1:
                    new_track.trackId = max_iou_id
                    new_track.trackLifetime = len(self.track_queue[max_iou_id])
                    
                    time_period = (new_track.frame-self.track_queue[max_iou_id][-1].frame)/self.frame_rate
                    if time_period==0:
                        #print(frame_id)
                        time_period = 1/self.frame_rate

                    new_track.xVelocity = abs(new_track.xCenter-self.track_queue[max_iou_id][-1].xCenter)/time_period
                    new_track.yVelocity = abs(new_track.yCenter-self.track_queue[max_iou_id][-1].yCenter)/time_period
                    
                    scale = self.get_scale(new_track.xCenter,new_track.yCenter)
                    
                    new_track.lonVelocity = new_track.xVelocity*scale
                    new_track.latVelocity = new_track.yVelocity*scale
                    
                    if new_track.trackLifetime>1:
                        new_track.xAcceleration = (new_track.xVelocity-self.track_queue[max_iou_id][-1].xVelocity)/time_period
                        new_track.yAcceleration = (new_track.yVelocity-self.track_queue[max_iou_id][-1].yVelocity)/time_period
                        
                        new_track.xAcceleration = (new_track.lonVelocity-self.track_queue[max_iou_id][-1].lonVelocity)/time_period
                        new_track.yAcceleration = (new_track.latVelocity-self.track_queue[max_iou_id][-1].latVelocity)/time_period

                    self.track_queue[max_iou_id].append(new_track)
                else:
                    new_track.trackId = self.create_new_id()
                    self.track_queue[new_track.trackId] = [new_track]
                    self.live_ids.append(new_track.trackId)
                #print(self.track_queue)

        for live_id in self.live_ids:
            if frame_id - self.track_queue[live_id][-1].frame>5:
                self.live_ids.remove(live_id)
                self.dead_ids.append(live_id)    


    def vis(self,frame):
        #print(self.live_ids)
        for live_id in self.live_ids:
            avg_lonVelocity = 0
            avg_latVelocity = 0
            avg_c = 0
            present_count=0
            for i in range(len(self.track_queue[live_id])-1,-1,-1):
                track = self.track_queue[live_id][i]
                if track.lonVelocity!=-1:
                    avg_lonVelocity+=track.lonVelocity
                    avg_c+=1
                cv2.circle(frame, (int(track.xCenter),int(track.yCenter)), radius=0, color=(0,0,255), thickness=3)
                present_count+=1
                if present_count>15:
                    break
            '''
            for track in self.track_queue[live_id]:
                #print((int(track.xCenter),int(track.yCenter)))
                cv2.circle(frame, (int(track.xCenter),int(track.yCenter)), radius=0, color=(0, 0, 255), thickness=2)
            '''
            if avg_c>0:
                avg_lonVelocity/=avg_c
            track = self.track_queue[live_id][-1]
            cv2.putText(frame, str(round(avg_lonVelocity,2)), (int(track.xCenter),int(track.yCenter)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)    
        return frame


    def save_results(self,dst_dir):
        with open(dst_dir, 'wb') as handle:
            pickle.dump(self.track_queue, handle)        