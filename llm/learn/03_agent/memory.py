# -*- coding: utf-8 -*-
'''
03_agent/memory.py - Agent Memory System

Memory is essential for Agents to maintain context across interactions.

Types of memory:
  1. Short-term (Working): Current conversation context (limited window)
  2. Long-term (Episodic): Important past interactions (stored + retrieved)
  3. Semantic: Facts and knowledge about the world

VLA connection:
  In embodied agents, memory stores:
  - Past observations (what did the robot see?)
  - Action history (what did the robot do?)
  - Task progress (what has been completed?)
'''

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque
import time


@dataclass
class Message:
    '''A single message in the conversation.'''
    role: str       # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Episode:
    '''A recorded episode of agent-environment interaction.'''
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    '''
    Short-term (working) memory using a sliding window.

    Stores recent conversation turns. When the window is full,
    oldest messages are dropped (or summarized).
    '''

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages)

    def add(self, role: str, content: str, **metadata):
        '''Add a message to short-term memory.'''
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)

    def add_user(self, content: str):
        self.add("user", content)

    def add_assistant(self, content: str):
        self.add("assistant", content)

    def add_system(self, content: str):
        self.add("system", content)

    def add_tool_result(self, tool_name: str, result: str):
        self.add("tool", f"[{tool_name}]: {result}")

    def get_context(self, num_recent: int = None) -> List[Dict[str, str]]:
        '''Get recent messages as context for the LLM.'''
        msgs = list(self.messages)
        if num_recent:
            msgs = msgs[-num_recent:]
        return [m.to_dict() for m in msgs]

    def get_last_n_messages(self, n: int) -> List[Message]:
        '''Get the last n Message objects.'''
        msgs = list(self.messages)
        return msgs[-n:]

    def clear(self):
        self.messages.clear()

    def summary(self) -> str:
        '''Generate a simple summary of the conversation.'''
        user_msgs = [m for m in self.messages if m.role == "user"]
        tool_msgs = [m for m in self.messages if m.role == "tool"]
        return (
            f"Conversation: {len(user_msgs)} user messages, "
            f"{len(tool_msgs)} tool calls"
        )

    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    '''
    Long-term memory for storing important information across sessions.

    Simplified implementation: key-value store with similarity search.
    In production, this would use a vector database (ChromaDB, Pinecone, etc.).

    VLA: Stores successful action sequences, object locations, task knowledge.
    '''

    def __init__(self):
        self.store: Dict[str, Any] = {}

    def save(self, key: str, value: Any):
        '''Save information with a key.'''
        self.store[key] = {
            "value": value,
            "timestamp": time.time(),
        }

    def retrieve(self, key: str) -> Optional[Any]:
        '''Retrieve information by key.'''
        entry = self.store.get(key)
        return entry["value"] if entry else None

    def search(self, query: str) -> List[str]:
        '''Simple keyword-based search (placeholder for semantic search).'''
        results = []
        query_lower = query.lower()
        for key in self.store:
            if query_lower in key.lower():
                results.append(key)
        return results

    def forget(self, key: str):
        '''Remove information.'''
        self.store.pop(key, None)

    def clear(self):
        self.store.clear()

    def __contains__(self, key: str) -> bool:
        return key in self.store


class EpisodeBuffer:
    '''
    Stores agent-environment interaction episodes for learning.

    Used in VLA for:
    - Behavior Cloning: store (observation, action) pairs
    - Reinforcement Learning: store full (s, a, r, s', done) transitions
    '''

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)

    def add(self, episode: Episode):
        self.episodes.append(episode)

    def sample(self, batch_size: int) -> List[Episode]:
        '''Randomly sample episodes (for training).'''
        import random
        return random.sample(list(self.episodes), min(batch_size, len(self.episodes)))

    def get_all(self) -> List[Episode]:
        return list(self.episodes)

    def clear(self):
        self.episodes.clear()

    def __len__(self):
        return len(self.episodes)


if __name__ == '__main__':
    print('=' * 60)
    print('Memory System Demo')
    print('=' * 60)

    # Short-term memory
    stm = ShortTermMemory(max_messages=10)
    stm.add_user("What is the capital of France?")
    stm.add_assistant("The capital of France is Paris.")
    stm.add_user("What about Germany?")
    stm.add_assistant("The capital of Germany is Berlin.")

    print(f'STM: {len(stm)} messages')
    print(f'Context: {stm.get_context()}')

    # Long-term memory
    ltm = LongTermMemory()
    ltm.save("user_preference", {"language": "Chinese", "style": "concise"})
    ltm.save("france_capital", "Paris")
    print(f'LTM retrieve: {ltm.retrieve("france_capital")}')
    print(f'LTM search "capital": {ltm.search("capital")}')

    # Episode buffer
    buf = EpisodeBuffer(max_episodes=100)
    buf.add(Episode(observation=[1, 2, 3], action=0, reward=1.0,
                    next_observation=[2, 3, 4], done=False))
    print(f'Episodes stored: {len(buf)}')
    print('Done!')
