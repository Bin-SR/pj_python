# -*- coding: utf-8 -*-
'''
03_agent/demo_agent.py - Complete Agent System Demo

Demonstrates ReAct loop + Tool Use + Memory + Planning.
Shows how Agent framework maps to VLA robot control.
'''

import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

from tools import ToolRegistry, create_default_tools, Tool
from memory import ShortTermMemory, LongTermMemory, EpisodeBuffer, Episode
from planner import TaskPlanner, PlanStatus
from react_agent import ReActAgent, MockLLM


def robot_move_to(x=0.0, y=0.0, z=0.0):
    return f'Robot moved to ({x:.3f}, {y:.3f}, {z:.3f})'

def robot_grasp(force=10.0):
    return f'Grasped object with force {force}N'

def robot_release():
    return 'Object released'

def robot_look():
    objects = ['red block at (0.5, 0.3)', 'blue cup at (0.2, 0.4)', 'green ball at (0.7, 0.2)']
    seen = random.sample(objects, k=random.randint(1, 3))
    return 'Camera sees: ' + ', '.join(seen)


def create_robot_tools():
    registry = create_default_tools()
    registry.register(Tool(name='move_to',
        description='Move robot end-effector to (x, y, z) in meters.',
        parameters={'type':'object','properties':{'x':{'type':'number'},'y':{'type':'number'},'z':{'type':'number'}},'required':['x','y','z']},
        function=robot_move_to))
    registry.register(Tool(name='grasp',
        description='Grasp object at current position.',
        parameters={'type':'object','properties':{'force':{'type':'number'}},'required':[]},
        function=robot_grasp))
    registry.register(Tool(name='release',
        description='Release grasped object.',
        parameters={'type':'object','properties':{},'required':[]},
        function=robot_release))
    registry.register(Tool(name='look',
        description='Observe scene through camera.',
        parameters={'type':'object','properties':{},'required':[]},
        function=robot_look))
    return registry


class VLAAgentDemo:
    '''Demonstrates Agent -> VLA bridge for embodied tasks.'''

    def __init__(self):
        self.llm = MockLLM()
        self.tools = create_robot_tools()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.episode_buffer = EpisodeBuffer()
        self.planner = TaskPlanner()

    def run_embodied_task(self, task):
        print(f'Embodied Task: {task}')
        observation = self.tools.execute('look', {})
        self.short_term.add_system(f'Initial: {observation}')
        print(f'Observe: {observation}')

        plan = self.planner.create_plan(task)
        print(f'Plan:')
        print(plan.format_for_llm())

        for step in plan.steps:
            step.mark_in_progress()
            desc = step.description.lower()
            if 'observe' in desc or 'look' in desc:
                result = self.tools.execute('look', {})
            elif 'move' in desc or 'approach' in desc:
                result = self.tools.execute('move_to', {'x': 0.5, 'y': 0.3, 'z': 0.2})
            elif 'grasp' in desc or 'pick' in desc:
                result = self.tools.execute('grasp', {'force': 10.0})
            elif 'release' in desc or 'place' in desc:
                result = self.tools.execute('release', {})
            else:
                result = f'Completed: {step.description}'
            step.mark_completed(result)
            print(f'  [{step.status.value}] {step.description}: {result}')
            self.episode_buffer.add(Episode(
                observation=step.description, action=result, reward=1.0,
                next_observation=f'After: {result}', done=False))

        final = self.tools.execute('look', {})
        print(f'Verify: {final}')
        print(f'Done! Episodes: {len(self.episode_buffer)}')


def demo_general_agent():
    print('=== Demo 1: General ReAct Agent ===')
    llm = MockLLM()
    agent = ReActAgent(llm_call=llm, max_steps=5)
    for t in ['Calculate 2 + 3 * 4', 'What is MuJoCo?']:
        print(f'Task: {t}')
        print(f'Result: {agent.run(t)}')


def demo_embodied_agent():
    print('=== Demo 2: Embodied Agent (VLA Bridge) ===')
    VLAAgentDemo().run_embodied_task('pick up the red block and place it on the table')


def demo_architecture():
    print('=== Demo 3: Architecture Overview ===')
    print('Agent -> VLA mapping:')
    print('  Thought: reasoning -> planning')
    print('  Action: tool call -> robot command')
    print('  Observation: tool result -> sensor reading')
    print('Key: VLA is an Agent whose tools are robot actions.')


if __name__ == '__main__':
    demo_general_agent()
    demo_embodied_agent()
    demo_architecture()
    print('All demos complete!')