import os
from math import cos, e, exp, log, pi, sin, sqrt, tan

# tools
from typing import Any, Optional, Union

from langchain.agents import AgentType, initialize_agent

# from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool


class DirectoryTool(BaseTool):
    name = "Directory Tool"
    description = "use this tool to find json data about a person in the directory"

    def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async run not implemented yet...")

    def _run(self, person_name):
        directory = {
            "Brad": {
                "age": 32,
                "occupation": "engineer",
                "hobbies": ["hiking", "biking", "skiing"],
            },
            "Jane": {
                "age": 28,
                "occupation": "data scientist",
                "hobbies": ["painting", "cooking", "reading"],
            },
            "Bob": {
                "age": 45,
                "occupation": "manager",
                "hobbies": ["fishing", "hunting", "hiking"],
                "contact_info": {
                    "email": "bob@bob.com",
                    "phone": "555-555-5555",
                    "address": "123 Main St.",
                },
            },
        }
        return directory.get(person_name, f"I do not know {person_name}.")


class PythogorasTool(BaseTool):
    name = "Hypotenuse Calculator"
    description = """use this tool when you need to calculate the length of an hypotenuse of a right triangle given one or two sides of a triangle and/or an angle (in degrees).
    To use the tool you must provide at least two of the following parameters: ['adjacent_side', 'opposite_side', 'angle'].
    """

    def _run(
        self,
        adjacent_side: Optional[Union[int, float]] = None,
        opposite_side: Optional[Union[int, float]] = None,
        angle: Optional[Union[int, float]] = None,
    ):
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side) ** 2 + float(opposite_side) ** 2)
        elif adjacent_side and angle:
            return float(adjacent_side) / cos(float(angle))  # * pi / 180)
        elif opposite_side and angle:
            return float(opposite_side) / sin(float(angle))
        else:
            return "Could not calculate hypotenuse."

    def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async run not implemented yet...")


class DoublerTool(BaseTool):
    name = "DoublerTool"
    description = "a tool for doubling numbers"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self, message: Union[int, float]):
        return (float(message) * 2)

    def _arun(self, message):
        raise NotImplementedError("async run not implemented yet...")


llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.2,
    # 0.0 = no creativity with tool code, but even 1.0 seems to work.
    # going over 1 can cause lots of looping and errors because it has difficulty using the tools.
    model_name="gpt-3.5-turbo",
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
)

tools = [
    DoublerTool(),
    PythogorasTool(),
    DirectoryTool(),
]

# initialize agent
# name="Toolie",  # wears tan shoes with pink shoes laces
# description="A tool user",
# default_system_prompt = toolie.agent.llm_chain.prompt.messages[0].prompt.template
# print(f"{default_system_prompt=}")
default_prompt = """
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks,
   from answering simple questions to providing in-depth explanations 
   and discussions on a wide range of topics. 
As a language model, Assistant is able to generate human-like text based on the input it receives, 
   allowing it to engage in natural-sounding conversations and provide responses 
   that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
   It is able to process and understand large amounts of text, 
   and can use this knowledge to provide accurate and informative responses 
   to a wide range of questions. 
Additionally, Assistant is able to generate its own text based on the input it receives, 
   allowing it to engage in discussions and provide explanations and descriptions on a 
   wide range of topics.
Overall, Assistant is a powerful system that can help with a wide range of tasks 
   and provide valuable insights and information on a wide range of topics. 
   Whether you need help with a specific question or just want to have a conversation 
   about a particular topic, Assistant is here to assist. 
[TOOL_CLAUSE]
"""
# TOOL_DESCRIPTIONS = "\n".join([f"{tool.name} is {tool.description}" for tool in tools])
TOOL_CLAUSE = """
Unfortunately, Assistant is terrible at doubling numbers.
   when provided with doubling and math questions, no matter how simple, 
   assistant always refers to it's trusty tools and absolutely does not try to answer math or doubling questions by itself.
"""
TOOL_CLAUSE = """
Unfortunately, Assistant is terrible at math.
   when provided with math questions, no matter how simple, 
   assistant always refers to it's trusty tools and absolutely does not try to answer math questions by itself.
   Assistant will always consult its tools before answering a math question.
"""
ssytem_prompt = default_prompt.replace("[TOOL_CLAUSE]", TOOL_CLAUSE)
toolie = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # "chat-conversational-react-description",
    #  react -> ai should react to the user
    #  description -> ai should choose tools based upon their .description
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,  # how many steps of reasoning, evalultion, and reaction
    early_stopping_method="generate",  # the model will decide when to stop; "convergence",
    memory=conversational_memory,
)
new_prompt = toolie.agent.create_prompt(system_message=ssytem_prompt, tools=tools)
toolie.agent.llm_chain.prompt = new_prompt


initial_context = {
    "chat_history": [],  # 'conversational_memory': [],
    # 'input': "can you double 3.5?"
    "input": """If I have a triangle to two sides of lenth 51cm and 34cm, 
                what is the length of the hypotenuse?""",
}

res = toolie(initial_context)
print(f"{res=}")
res = toolie({"input": "can you double that?"})
print(f"{res=}")
res = toolie({"input": "can you tell me about Bob?"})
print(f"{res=}")
res = toolie({"input": "How can I contact him?"})
print(f"{res['output']=}")
res = toolie({"input": "How can I contact Jane?"})
print(f"{res['output']=}")
res = toolie({"input": "How can I contact Glenn?"})
print(f"{res['output']=}")
