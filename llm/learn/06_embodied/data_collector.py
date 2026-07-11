# -*- coding: utf-8 -*-
"""
06_embodied/data_collector.py - Demo Data Collection for VLA Training
Collects (image, instruction, action) triples for VLA training.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import json, os, time


@dataclass
class DemoStep:
    timestamp: float; image: np.ndarray; instruction: str
    action: np.ndarray; joint_positions: np.ndarray; ee_pose: np.ndarray
    reward: float = 0.0
    def to_dict(self):
        return {'timestamp':self.timestamp,'instruction':self.instruction,
            'action':self.action.tolist(),'joints':self.joint_positions.tolist(),
            'ee_pose':self.ee_pose.tolist()}


@dataclass
class DemoEpisode:
    task_id: str; task_description: str
    steps: List[DemoStep] = field(default_factory=list); success: bool = False
    def add_step(self, s): self.steps.append(s)
    def __len__(self): return len(self.steps)
    def to_dict(self):
        return {'task_id':self.task_id,'desc':self.task_description,
            'steps':len(self.steps),'success':self.success,
            'data':[s.to_dict() for s in self.steps]}


class DataCollector:
    """Collects and manages demonstration data."""
    def __init__(self, save_dir='./demo_data'):
        self.save_dir = save_dir; self.episodes = []; self.current = None
        os.makedirs(save_dir, exist_ok=True)
    def start_episode(self, tid, desc):
        self.current = DemoEpisode(tid, desc)
    def record_step(self, img, inst, act, joints, ee, reward=0.0):
        if self.current is None: return
        self.current.add_step(DemoStep(time.time(),img.copy(),inst,
            np.asarray(act),np.asarray(joints),np.asarray(ee),reward))
    def end_episode(self, success=True):
        if self.current is None: return
        self.current.success = success
        self.episodes.append(self.current)
        p = os.path.join(self.save_dir, f'{self.current.task_id}.json')
        with open(p,'w') as f: json.dump(self.current.to_dict(),f)
        self.current = None
    def get_training_data(self):
        imgs, insts, acts = [], [], []
        for ep in self.episodes:
            for s in ep.steps:
                imgs.append(s.image); insts.append(s.instruction); acts.append(s.action)
        return np.array(imgs), insts, np.array(acts)
    def get_statistics(self):
        total = sum(len(ep) for ep in self.episodes)
        sr = sum(1 for ep in self.episodes if ep.success) / max(len(self.episodes), 1)
        return {'episodes':len(self.episodes),'steps':total,'success_rate':sr}
    def collect_synthetic(self, n=5):
        tasks = [('pick_block','pick up the red block'),('place','place on table'),('home','go home')]
        for i in range(n):
            tid, desc = tasks[i % len(tasks)]
            self.start_episode(tid, desc)
            for _ in range(np.random.randint(10,30)):
                self.record_step(np.random.randint(0,255,(224,224,3),dtype=np.uint8),
                    desc, np.random.randn(7)*0.01, np.random.randn(7)*0.5, np.random.randn(3)*0.3+0.5)
            self.end_episode(np.random.random() > 0.3)


if __name__ == '__main__':
    dc = DataCollector('./demo_data')
    dc.collect_synthetic(3)
    stats = dc.get_statistics()
    for k,v in stats.items(): print(f'  {k}: {v}')
    imgs, insts, acts = dc.get_training_data()
    print(f'Training data: {len(imgs)} samples, actions shape: {acts.shape}')
    print('Done!')