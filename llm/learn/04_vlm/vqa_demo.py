# -*- coding: utf-8 -*-
"""
04_vlm/vqa_demo.py - Visual Question Answering Demo

Demonstrates how a VLM answers questions about images.
Example: Show picture of a cat -> Ask 'What animal is this?' -> Answer 'cat'

VLA connection: VQA is the foundation for instruction-following robots.
'What object is on the table?' -> robot identifies target -> 'Pick it up'
"""

import torch
import torch.nn.functional as F


class VQADemo:
    """Demonstrates the VQA pipeline conceptually."""

    def __init__(self):
        # Mock knowledge base mapping (image_id, question_type) -> answer
        self.knowledge = {
            ('cat.jpg', 'what animal'): 'cat',
            ('cat.jpg', 'what color'): 'orange',
            ('table.jpg', 'what object'): 'red block',
            ('table.jpg', 'how many'): 'three',
            ('robot.jpg', 'what robot'): 'Franka Emika Panda',
            ('robot.jpg', 'how many joints'): 'seven',
        }

    def answer(self, image_id, question):
        """Simulate VQA: image + question -> answer."""
        q_lower = question.lower()
        for (img, qtype), answer in self.knowledge.items():
            if img == image_id and qtype in q_lower:
                return answer
        return 'I cannot determine this from the image.'

    def demo_vla_qa(self):
        """Demonstrate how VQA relates to robot tasks."""
        print('VLA Scenario: Robot sees a table with blocks')
        print()
        queries = [
            ('table.jpg', 'What objects are on the table?'),
            ('table.jpg', 'What color is the block?'),
            ('table.jpg', 'How many objects are there?'),
        ]
        for img, q in queries:
            a = self.answer(img, q)
            print(f'  Q: {q}')
            print(f'  A: {a}')
            print(f'  -> Robot action based on answer')
            print()


if __name__ == '__main__':
    print('=' * 60)
    print('VQA Demo (Visual Question Answering)')
    print('=' * 60)
    demo = VQADemo()

    # General VQA
    print('General VQA:')
    for img in ['cat.jpg', 'robot.jpg']:
        q = f'what is in {img}?'
        print(f'  Image: {img} -> Q: {q} -> A: {demo.answer(img, q)}')

    # VLA-specific
    demo.demo_vla_qa()
    print('Done!')