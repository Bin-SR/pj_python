# -*- coding: utf-8 -*-
"""
06_embodied/perception.py - Visual Perception for Embodied Agents
Object detection, depth estimation, scene description for VLA.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DetectedObject:
    label: str; position: np.ndarray; bbox: Tuple[int,int,int,int]
    confidence: float = 0.9; color: str = 'unknown'


class SimpleObjectDetector:
    """Simplified object detector for learning."""
    def __init__(self):
        self.objects = [
            DetectedObject('red_block', np.array([0.5,0.2,0.1]),(50,50,30,30),0.95,'red'),
            DetectedObject('blue_cup', np.array([0.3,-0.1,0.15]),(120,80,25,35),0.90,'blue'),
            DetectedObject('green_ball', np.array([0.7,0.3,0.08]),(180,60,20,20),0.88,'green'),
        ]
    def detect(self, image): return self.objects
    def find_by_color(self, image, color):
        for obj in self.detect(image):
            if obj.color == color.lower(): return obj
        return None


class DepthEstimator:
    """Simplified depth estimator."""
    def estimate(self, image):
        h, w = image.shape[:2]
        depth = np.ones((h,w), dtype=np.float32) * 0.5
        for x,y,bw,bh,d in [(50,50,30,30,0.3),(120,80,25,35,0.4)]:
            depth[y:y+bh, x:x+bw] = d
        return depth


class VisualPerceptionPipeline:
    """Complete visual perception for VLA."""
    def __init__(self):
        self.detector = SimpleObjectDetector()
        self.depth = DepthEstimator()
    def process(self, image):
        return {'objects': self.detector.detect(image), 'depth': self.depth.estimate(image)}
    def format_for_vlm(self, image):
        objs = self.detector.detect(image)
        lines = ['Scene:']
        for o in objs:
            p = o.position
            lines.append(f'  - {o.color} {o.label} at ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})')
        return '\n'.join(lines)


if __name__ == '__main__':
    pipe = VisualPerceptionPipeline()
    img = np.random.randint(0, 255, (224,224,3), dtype=np.uint8)
    print(pipe.format_for_vlm(img))
    print('Done!')