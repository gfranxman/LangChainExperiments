import io

# tools
import logging

import openai
from requests.exceptions import ChunkedEncodingError

logger = logging.getLogger(__name__)

import os
import re
from collections import defaultdict
from pathlib import Path

import llm
import pandas as pd
import pdfplumber
from agent_tools import DirectoryTool, DoublerTool, PythogorasTool, StockTool
from langchain.agents import AgentType, initialize_agent

# from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

# collect, extract, summarize, embed, search, enable(tools), generate

# TODO: add gpt4all to download local non-gpu models.

class BaseCollector:
    """yields pdf files"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collect(self, *args, **kwargs):
        raise NotImplementedError("collect not implemented yet...")


class DirectoryCollector(BaseCollector):
    """yields pdf files from a folder"""

    def __init__(self, base_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_folder = base_folder

    def collect(self, *args, **kwargs):
        import glob

        for file in glob.glob(f"{self.base_folder}/*.pdf"):
            yield file


class BaseExtractor:
    """yields text from pdf files"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self, *args, **kwargs):
        raise NotImplementedError("extract not implemented yet...")

    @classmethod
    def point_is_in_box(cls, x, y, box):
        return x >= box[0] and x <= box[2] and y >= box[1] and y <= box[3]

    @classmethod
    def text_is_in_bboxes(cls, text_struct, bboxes):
        # x0, top, x1, bottom   # why?!?
        point = text_struct["x0"], text_struct["top"]
        for box_index, box in enumerate(bboxes):
            if cls.point_is_in_box(point[0], point[1], box):
                return True, box_index
        return False, "not in box"


class PdfMinerExtractor(BaseExtractor):
    """yields text from pdf files"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self, *args, **kwargs):
        raise NotImplementedError("extract not implemented yet...")


class PdfPlumberExtractor(BaseExtractor):
    """yields unstructured text from pdf files"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(
        self,
        pdf_file_path: str,
        include_table_footers: bool = True,
        include_page_footers: bool = False,
    ) -> str:
        """
        Extract text and tables from a PDF file.
        """
        output = io.StringIO()
        pdf = pdfplumber.open(pdf_file_path)
        logging.info(f"{pdf_file_path} has {len(pdf.pages)} pages")
        for page in pdf.pages:
            text_structs = page.extract_text_lines()

            # we're going to split the text into two groups:
            # 1. text that is in a table
            # 2. text that is not in a table
            # and then we'll interleave the two groups with --- markers for the csv tables
            non_table_text_structs = []
            table_text_structs = defaultdict(list)  # dict[list[struct]]

            bounding_boxes = []
            table_rows = defaultdict(list)

            for tbl_num, table in enumerate(page.find_tables()):
                logger.info(f"{tbl_num=}")
                # The bounding box of the table can be accessed using the table's bbox attribute
                bbox = table.bbox
                bounding_boxes.append(bbox)

                rows = table.extract()

                for row in rows:
                    table_rows[tbl_num].append(row)

            for text_struct in text_structs:
                in_box, tbl_index = self.text_is_in_bboxes(text_struct, bounding_boxes)
                if in_box:
                    table_text_structs[tbl_index].append(text_struct)

                else:
                    non_table_text_structs.append(text_struct)

            # Interleave the text with --- demarked csv tables
            last_table_num = None
            for text_struct in text_structs:
                in_box, tbl_index = self.text_is_in_bboxes(text_struct, bounding_boxes)
                if in_box and tbl_index == last_table_num:
                    pass
                elif in_box:
                    last_table_num = tbl_index
                    print("---", file=output)

                    columns = [
                        str(c).replace("\n", " ")
                        for i, c in enumerate(table_rows[tbl_index][0])
                    ]
                    df = pd.DataFrame(table_rows[tbl_index][1:], columns=columns)

                    df.replace(
                        to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                        value=[" ", " "],
                        regex=True,
                        inplace=True,
                    )

                    print(df.to_csv(index=False), end="", file=output)

                    print("---", file=output)
                    if include_table_footers:
                        print(f"Table {tbl_index}.CSV\n", file=output)
                else:
                    print(text_struct["text"], end="", file=output)

            if include_page_footers:
                print(
                    f"\n\nFile: {Path(pdf_file_path).stem}; Page {page.page_number}",
                    file=output,
                )

        retval = output.getvalue()
        output.close()
        return retval


class BaseChunker:
    """yields chunks from text"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def chunk(self, text, *args, **kwargs):
        raise NotImplementedError("chunk not implemented yet...")


class NaiveChunker(BaseChunker):
    """yields chunks from text"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def chunk(self, text, pattern=r"\n", chunk_size=1024):
        """
        Yields chunks of text at a maximum of `chunk_size` characters,
        without breaking individual paragraphs, as defined by a regex pattern.
        Combines multiple paragraphs into a chunk if they fit within the chunk size limit.

        :param text: The long string of sentences and data to be chunked.
        :param pattern: The regex pattern that marks the end of a paragraph.
        :param chunk_size: The maximum size of each chunk in characters.
        """
        paragraph = ""
        pchunk = ""

        logger.info(f"chunking text, {pattern=}, {chunk_size=}")

        for match in re.finditer(pattern, text):
            logger.info(f"{match=}")
            end_of_paragraph = match.end()
            paragraph = text[:end_of_paragraph]
            text = text[end_of_paragraph:]

            if len(pchunk + paragraph) <= chunk_size:
                pchunk += paragraph
            else:
                if pchunk:
                    yield pchunk
                pchunk = paragraph  # Start a new chunk with the current paragraph

        # Yield the last chunk if it's non-empty
        if pchunk:
            yield pchunk
        elif paragraph:
            yield paragraph


class BaseSummarizer:
    """yields summaries from text"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summarize(self, *args, **kwargs):
        raise NotImplementedError("summarize not implemented yet...")


class ContractSummarizer(BaseSummarizer):
    """yields summaries from text"""

    def __init__(self, chunker=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunker = chunker or NaiveChunker()

    def summarize(self, text, *args, **kwargs):
        # orca-mini-7b
        model = llm.get_model("gpt-3.5-turbo")
        model.key = os.environ.get("OPENAI_API_KEY")
        summary = ""

        logger.info(f"summarizing {len(text)=} chars")
        logger.info(f"{text=}\n\n")

        # need a better chunker here
        for i, chunk in enumerate(self.chunker.chunk(text)):
            logging.info(f"summarizing chunk {i}, of length {len(chunk)}")
            logger.info(f"{chunk=}\n\n")
            response = model.prompt(
                "Summarize the following text as an expert paralegal:\n\n" + chunk
            )
            try:
                summary += response.text()+ " \n"
            except ChunkedEncodingError as comm_err:
                logger.error(f"{comm_err=}")
            except openai.error.ServiceUnavailableError as comm_err:
                logger.error(f"{comm_err=}")


        return summary


class BaseEmbedder:
    """yields embeddings from text"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed(self, *args, **kwargs):
        raise NotImplementedError("embed not implemented yet...")


class BaseSearcher:
    """yields search results from embeddings"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search(self, *args, **kwargs):
        raise NotImplementedError("search not implemented yet...")


chat_llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.0,
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
    StockTool(),
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
Unfortunately, Assistant is terrible at math.
   when provided with math questions, no matter how simple, 
   assistant always refers to it's trusty tools and absolutely does not try to answer math questions by itself.
   Assistant will always consult its tools before answering a math question.
When asked about a company, always provide the company's ticker symbol and the quote using the "Stock Quote Tool".
"""
ssytem_prompt = default_prompt.replace("[TOOL_CLAUSE]", TOOL_CLAUSE)
llm_agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # "chat-conversational-react-description",
    #  react -> ai should react to the user
    #  description -> ai should choose tools based upon their .description
    tools=tools,
    llm=chat_llm,
    # verbose=True,
    max_iterations=3,  # how many steps of reasoning, evalultion, and reaction
    early_stopping_method="generate",  # the model will decide when to stop; "convergence",
    memory=conversational_memory,
)
new_prompt = llm_agent.agent.create_prompt(system_message=ssytem_prompt, tools=tools)
llm_agent.agent.llm_chain.prompt = new_prompt


initial_context = {
    "chat_history": [],  # 'conversational_memory': [],
    # 'input': "can you double 3.5?"
    "input": """If I have a triangle to two sides of lenth 51cm and 34cm, 
                what is the length of the hypotenuse?""",
}


def collect():
    for f in DirectoryCollector("./source_docs").collect():
        print(f"{f=}")


def llm_tool_demo():
    res = llm_agent(initial_context)
    print(f"{res=}")

    res = llm_agent({"input": "can you double that?"})
    print(f"{res['output']=}")

    res = llm_agent({"input": "can you tell me about Bob?"})
    print(f"{res['output']=}")

    res = llm_agent({"input": "How can I contact him?"})
    print(f"{res['output']=}")

    res = llm_agent({"input": "How can I contact Jane?"})
    print(f"{res['output']=}")

    res = llm_agent({"input": "How can I contact Glenn?"})
    print(f"{res['output']=}")

    res = llm_agent({"input": "What is Factset?"})
    print(f"{res['output']=}")

    res = llm_agent({"input": "FactSet stock value?"})
    print(f"{res['output']=}")


if __name__ == "__main__":
    LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
    logger.setLevel(LOGLEVEL)
    # logging.basicConfig(level=LOGLEVEL)  # this sets everyone's log level

    # collect()
    llm_tool_demo()
    for f in DirectoryCollector("./source_docs").collect():
        print(f"{f=}")
        extracted_text = PdfPlumberExtractor().extract(f)
        print(extracted_text)

        summarized_text = ContractSummarizer().summarize(extracted_text)
        print(f"\nhere's the summary for {f}\n")
        print(summarized_text)
        print()

