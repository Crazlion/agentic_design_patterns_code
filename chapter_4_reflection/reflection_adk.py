# encoding=utf-8

import os
import json
import asyncio
from typing import Optional
from dotenv import load_dotenv
import uuid

from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

DOUBAO_MODEL = "openai/doubao-seed-1-6-251015"
doubao_llm = LiteLlm(model=DOUBAO_MODEL)

# The first agent generates the initial draft.
generator = LlmAgent(
    name="初稿",
    model=doubao_llm,
    description="生成给定主题的初始草稿内容。",
    instruction="写一个简短的，关于用户主题的信息段落。",
    output_key="draft_text"  # The output is saved to this state key.
)
# The second agent critiques the draft from the first agent.
reviewer = LlmAgent(
    name="审查者",
    model=doubao_llm,
    description="审查给定文本的事实准确性，并提供结构化的批评。",
    instruction="""
    你是一个一丝不苟的事实核查者。
    1. 读取状态键'draft_text'中提供的文本。
    2. 仔细核实所有声明的事实准确性。
    3. 你的最终输出必须是一个包含两个键的字典：
    - "status":字符串,可以是“ACCURATE”或“INACCURATE”。
    - "reasoning"：一个字符串，为你的行为提供一个清晰的解释状态，指出具体问题（如果发现的话）。
    """,
    output_key="review_output"  # The structured dictionary is saved here.
)

final_text = LlmAgent(
    name="最终版本稿件",
    model=doubao_llm,
    description="根据之前的初稿和审查者的信息重新改写初稿得到最终版本的稿件",
    instruction="""你是一个稿件修改人员：
    1. 读取初稿的输出draft_text 
    2. 读取审查者的修改意见
    3. 结合初稿和审查者的修改意见，重新改写初稿得到最终版本的稿件
    """,
    output_key="final_text"  # The output is saved to this state key.
)

# The SequentialAgent ensures the generator runs before the reviewer.
review_pipeline = SequentialAgent(
    name="WriteAndReview_Pipeline",
    sub_agents=[generator, reviewer, final_text]
)
# Execution Flow:
# 1. generator runs -> saves its paragraph to state['draft_text'].
# 2. reviewer runs -> reads state['draft_text'] and saves its dictionary output to state['review_output'].

# --- Execution Logic ---


async def run_with_agent(runner: InMemoryRunner, request: str):
    """Runs the coordinator agent with a given request and delegates."""
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id)
        event_times = 0
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            if event.is_final_response() and event.content:
                # Try to get text directly from event.content
                # to avoid iterating parts
                if hasattr(event.content, 'text') and event.content.text:
                    print("final_response event.content.text")
                    final_result = event.content.text
                elif event.content.parts:
                    part_nums = len(event.content.parts)
                    print(f"part_nums:{part_nums}")
                    text_parts = [
                        part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                # Assuming the loop should break after the final response
                break
            else:
                event_times += 1
                print(f"event_times:{event_times} - {event}")
                # 中间过程输出一些基础信息
                if event.partial and event.content and event.content.parts and event.content.parts[0].text:
                    print(
                        f"事件来源: {event.author} -- {event.content.parts[0].text}")

#    print(f"Coordinator Final Response: {final_result}")
        return final_result
    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


async def main():
    runner = InMemoryRunner(review_pipeline)

    result_a = await run_with_agent(runner, "帮我写一个关于小女孩的童话故事")
    print(f"Final Output A: {result_a}")


if __name__ == "__main__":
    asyncio.run(main())
