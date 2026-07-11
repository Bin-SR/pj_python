# -*- coding: utf-8 -*-
'''
03_agent/planner.py - Task Planner for AI Agents
Decomposes complex tasks into subtasks and tracks execution.
VLA connection: plans high-level task sequence for the robot.
'''

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class PlanStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    step_id: int
    description: str
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_completed(self, result=""):
        self.status = PlanStatus.COMPLETED
        self.result = result

    def mark_failed(self, reason=""):
        self.status = PlanStatus.FAILED
        self.result = reason

    def mark_in_progress(self):
        self.status = PlanStatus.IN_PROGRESS


@dataclass
class Plan:
    task: str
    steps: List[PlanStep]
    current_step_idx: int = 0

    @property
    def current_step(self):
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    @property
    def is_complete(self):
        return all(s.status == PlanStatus.COMPLETED for s in self.steps)

    @property
    def progress(self):
        return sum(1 for s in self.steps if s.status == PlanStatus.COMPLETED) / max(len(self.steps), 1)

    def advance(self):
        for i in range(self.current_step_idx + 1, len(self.steps)):
            if self.steps[i].status == PlanStatus.PENDING:
                self.current_step_idx = i
                self.steps[i].mark_in_progress()
                return self.steps[i]
        return None

    def format_for_llm(self):
        icons = {PlanStatus.PENDING:"[ ]",PlanStatus.IN_PROGRESS:"[>]",
                 PlanStatus.COMPLETED:"[x]",PlanStatus.FAILED:"[!]",PlanStatus.SKIPPED:"[-]"}
        lines = [f"Task: {self.task}", "Plan:"]
        for s in self.steps:
            r = f" -> {s.result}" if s.result else ""
            lines.append(f"  {icons[s.status]} Step {s.step_id}: {s.description}{r}")
        return "\n".join(lines)


class TaskPlanner:
    def __init__(self):
        self.current_plan: Optional[Plan] = None
        self.completed_plans: List[Plan] = []

    def create_plan(self, task, llm_fn=None):
        if llm_fn:
            steps_data = llm_fn(task)
        else:
            steps_data = self._template(task)
        steps = [PlanStep(step_id=i, description=d) for i, d in enumerate(steps_data, 1)]
        self.current_plan = Plan(task=task, steps=steps)
        return self.current_plan

    def _template(self, task):
        tl = task.lower()
        if "search" in tl or "find" in tl:
            return ["Analyze query","Perform search","Filter results","Present findings"]
        elif "calculate" in tl or "compute" in tl:
            return ["Parse expression","Execute calculation","Verify result","Present answer"]
        elif "pick" in tl or "grasp" in tl:
            return ["Observe scene","Locate target","Plan approach","Move to pre-grasp",
                    "Execute grasp","Lift object","Move to target","Release"]
        elif "navigate" in tl or "go to" in tl:
            return ["Localize position","Plan path","Execute navigation","Confirm arrival"]
        else:
            return ["Understand task","Identify tools","Execute action","Verify completion"]

    def execute_step(self, step, executor_fn):
        step.mark_in_progress()
        try:
            result = executor_fn(step.description)
            step.mark_completed(result)
            return result
        except Exception as e:
            step.mark_failed(str(e))
            raise

    def replan(self, feedback):
        if self.current_plan is None:
            raise ValueError("No current plan")
        sid = len(self.current_plan.steps) + 1
        self.current_plan.steps.append(PlanStep(sid, f"Adjust: {feedback}"))
        return self.current_plan

    def finish_plan(self):
        if self.current_plan:
            self.completed_plans.append(self.current_plan)
            self.current_plan = None


if __name__ == '__main__':
    print("="*60)
    print("Task Planner Demo")
    print("="*60)
    p = TaskPlanner()
    plan = p.create_plan("pick up the red block and place it on the table")
    print(plan.format_for_llm())
    def mock_exec(s):
        return f"Mock execution of: {s}"
    p.execute_step(plan.steps[0], mock_exec)
    print("\nAfter step 1:")
    print(plan.format_for_llm())
    plan.advance()
    print(f"\nProgress: {plan.progress:.0%}")
    print("Done!")