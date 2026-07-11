# -*- coding: utf-8 -*-
'''
03_agent/react_agent.py - ReAct (Reasoning + Acting) Agent
Alternates between Thought, Action, and Observation.
VLA connection: same loop with robot actions as tools.
'''

import re
from typing import Optional, Dict, Any, Callable, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tools import ToolRegistry, create_default_tools
from memory import ShortTermMemory, LongTermMemory


class ReActAgent:
    def __init__(self, llm_call, tools=None, max_steps=10, verbose=True):
        self.llm = llm_call
        self.tools = tools or create_default_tools()
        self.max_steps = max_steps
        self.verbose = verbose
        self.short_term = ShortTermMemory(max_messages=50)
        self.long_term = LongTermMemory()

    def _build_prompt(self, task):
        tds = self.tools.get_descriptions()
        return f'''You are an AI assistant using ReAct.

Tools:
{tds}

Format:
Thought: <reasoning about what to do>
Action: tool_name[param="value"]
OR
Answer: <final answer>

Task: {task}
Begin!'''

    def _parse_action(self, text):
        m = re.search(r'Action:\s*(\w+)\[(.*?)\]', text, re.DOTALL)
        if not m: return None
        name = m.group(1)
        params = {}
        for pm in re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(2)):
            params[pm[0]] = pm[1]
        return name, params

    def _parse_thought(self, text):
        m = re.search(r'Thought:\s*(.*?)(?=Action:|Answer:|$)', text, re.DOTALL)
        return m.group(1).strip() if m else None

    def _parse_answer(self, text):
        m = re.search(r'Answer:\s*(.*?)$', text, re.DOTALL)
        return m.group(1).strip() if m else None

    def run(self, task):
        self._log(f"Task: {task}")
        self.short_term.add_user(task)
        prompt = self._build_prompt(task)
        ctx = [{"role":"user","content":prompt}]

        for step in range(self.max_steps):
            self._log(f"--- Step {step+1} ---")
            resp = self.llm(self._fmt(ctx))
            thought = self._parse_thought(resp)
            answer = self._parse_answer(resp)
            action = self._parse_action(resp)

            if thought:
                self._log(f"Thought: {thought}")

            if answer:
                self._log(f"Answer: {answer}")
                self.long_term.save(f"task:{task[:50]}", {"answer":answer,"steps":step+1})
                return answer

            if action:
                name, params = action
                self._log(f"Action: {name}({params})")
                obs = self.tools.execute(name, params)
                self._log(f"Observation: {obs}")
                ctx.append({"role":"assistant","content":resp})
                ctx.append({"role":"user","content":f"Observation: {obs}"})
            else:
                ctx.append({"role":"assistant","content":resp})
                ctx.append({"role":"user","content":"Continue."})

        return "Unable to complete within step limit."

    def _fmt(self, msgs):
        return "\n".join([m["content"] for m in msgs])

    def _log(self, msg):
        if self.verbose:
            print(msg)


class MockLLM:
    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1
        pl = prompt.lower()
        if self.call_count == 1 and "calculate" in pl:
            return 'Thought: I need to calculate.\nAction: calculator[expression="2 + 3 * 4"]'
        if "result" in pl and "14" in pl:
            return 'Thought: Done.\nAnswer: 2 + 3 * 4 = 14'
        if self.call_count == 1 and "search" in pl:
            return 'Thought: I should search.\nAction: web_search[query="mujoco"]'
        if "physics engine" in pl:
            return 'Thought: Found it.\nAnswer: MuJoCo is a physics engine for robotics.'
        return 'Thought: Direct answer.\nAnswer: Mock response for learning.'


if __name__ == '__main__':
    print("="*60)
    print("ReAct Agent Demo")
    print("="*60)
    llm = MockLLM()
    agent = ReActAgent(llm_call=llm, max_steps=5)
    result = agent.run("Calculate 2 + 3 * 4")
    print(f"Final: {result}")
    print(f"LLM calls: {llm.call_count}")
    print("Done!")