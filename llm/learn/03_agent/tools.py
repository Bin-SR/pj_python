# -*- coding: utf-8 -*-
'''
03_agent/tools.py - Tool Definition and Registration Framework
Tools are functions an Agent can call. VLA: robot actions are tools.
'''

import json, math, io, sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Any
    require_confirmation: bool = False
    def to_openai_schema(self):
        return {'type':'function','function':{'name':self.name,'description':self.description,'parameters':self.parameters}}
    def __call__(self, **kw):
        return self.function(**kw)


def calculator(expr):
    try:
        allowed = {k:v for k,v in math.__dict__.items() if not k.startswith('__')}
        allowed['abs'], allowed['round'] = abs, round
        return f'Result: {eval(expr, {"__builtins__":{}}, allowed)}'
    except Exception as e:
        return f'Error: {e}'


def web_search(query):
    mock = {'python':'Python is a programming language.',
            'transformer':'Transformer architecture from 2017.',
            'robot':'A robot carries out actions automatically.',
            'franka':'Franka Emika Panda: 7-DoF robotic arm.',
            'mujoco':'MuJoCo is a physics engine for robotics.'}
    for k, v in mock.items():
        if k in query.lower():
            return f'Result: {v}'
    return 'No results for query.'


def code_executor(code):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, {'__builtins__':{'print':print,'len':len,'range':range,'int':int,'float':float,'str':str,'list':list,'dict':dict,'sum':sum,'min':min,'max':max,'sorted':sorted}})
        out = sys.stdout.getvalue()
        return f'Output:\n{out}' if out else 'Executed (no output).'
    except Exception as e:
        return f'Error: {e}'
    finally:
        sys.stdout = old


def file_reader(path):
    try:
        with open(path,'r',encoding='utf-8') as f:
            content = f.read()
        if len(content) > 1000:
            content = content[:1000] + '...(truncated)'
        return f'File:\n{content}'
    except FileNotFoundError:
        return f'File not found: {path}'
    except Exception as e:
        return f'Error: {e}'


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    def register(self, tool):
        self._tools[tool.name] = tool
    def get(self, name):
        return self._tools.get(name)
    def get_schemas(self):
        return [t.to_openai_schema() for t in self._tools.values()]
    def get_descriptions(self):
        return '\n'.join([f'- {t.name}: {t.description}' for t in self._tools.values()])
    def execute(self, name, params):
        tool = self.get(name)
        if tool is None:
            return f'Tool not found: {name}. Available: {list(self._tools.keys())}'
        try:
            return tool(**params)
        except Exception as e:
            return f'Error: {e}'
    @property
    def tool_names(self):
        return list(self._tools.keys())


def create_default_tools():
    r = ToolRegistry()
    r.register(Tool('calculator','Evaluate math expressions.',
        {'type':'object','properties':{'expression':{'type':'string'}},'required':['expression']},calculator))
    r.register(Tool('web_search','Search for information.',
        {'type':'object','properties':{'query':{'type':'string'}},'required':['query']},web_search))
    r.register(Tool('code_executor','Execute Python code.',
        {'type':'object','properties':{'code':{'type':'string'}},'required':['code']},code_executor,True))
    r.register(Tool('file_reader','Read a file.',
        {'type':'object','properties':{'path':{'type':'string'}},'required':['path']},file_reader))
    return r


if __name__ == '__main__':
    print('='*60)
    print('Tool Registry Demo')
    print('='*60)
    r = create_default_tools()
    print(f'Tools: {r.tool_names}')
    print(f'Calc: {r.execute("calculator",{"expression":"2**10+math.sqrt(144)"})}')
    print(f'Search: {r.execute("web_search",{"query":"mujoco"})}')
    print('Done!')