import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict
import time
import dspy
from .callback import BaseCallbackHandler
from .dataclass import RapidInformationTable, RapidInformation
from ...interface import KnowledgeCurationModule, Retriever
from ...utils import ArticleTextProcessing

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    streamlit_connection = True
except ImportError as err:
    streamlit_connection = False

script_dir = os.path.dirname(os.path.abspath(__file__))

class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutlineWithExamples)
        self.engine = engine

    def forward(self, topic: str, brief_intro: str, similar_outlines: dict,
                example_num = 3,
                callback_handler: BaseCallbackHandler = None):

        cleaned_outlines = []
        for title, outline in similar_outlines.items():
            cleaned_outline = ArticleTextProcessing.clean_up_outline(outline)
            cleaned_outlines.append(f"Topic: {title}\nOutline:\n{cleaned_outline}")

        examples = "\n\n".join(cleaned_outlines[:example_num])

        with dspy.settings.context(lm=self.engine):
            draft_outline = self.draft_page_outline(topic=topic,brief_intro=brief_intro,examples=examples).outline
            draft_outline = ArticleTextProcessing.clean_up_outline(draft_outline)

        return dspy.Prediction(outline=draft_outline)

class WritePageOutlineWithExamples(dspy.Signature):
    """Write an outline for a Wikipedia page.
        Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Do not include other information.
        3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    brief_intro=dspy.InputField(prefix="Brief intro of the topic: ", format=str)
    examples = dspy.InputField(prefix="Outlines of similar topics: ", format=str)
    outline = dspy.OutputField(prefix="Write the outline of the topic: ", format=str)

class AttributeExtractor(dspy.Module):
    
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.extract_attribute = dspy.Predict(ExtractAttribute)
        self.engine = engine

    def forward(self, topic: str, outline: str, callback_handler: BaseCallbackHandler = None):
        """
        Extract attributes from the given topic and outline.

        Args:
            topic: The topic of interest.
            outline: The outline from which to extract attributes.
        
        Returns:
            attributes: A list containing the extracted attributes formatted similarly to WriteOutline.
        """
        # Here you would implement the logic to extract attributes from the outline
        # For demonstration, let's assume we simply return a formatted string.
        with dspy.settings.context(lm=self.engine):
            attributes = self.extract_attribute(topic=topic, outline=outline).attributes
        attributes = attributes.strip().split('\n')  # Parse the string into a list
        return dspy.Prediction(attributes=attributes)

class ExtractAttribute(dspy.Signature):
    """Generate attributes for a specified topic with its outline. The generated attributes should summarize all information needed to write the wiki page for this topic. Please avoid creating complex attributes; ensure that each attribute represents a distinct and indivisible aspect.
        Here is the format of your response:
        1. Generate attributes, each on a new line, ensuring no additional tags or formatting are included.
        2. Do not include other information.
        3. Do not include topic name itself in the attribute list.
    """
    
    topic = dspy.InputField(prefix="Topic: ", format=str)
    outline = dspy.InputField(prefix="Outline: ", format=str)
    attributes = dspy.OutputField(prefix="Attributes:\n", format=str)

class InfoMerger(dspy.Module):
    
    def __init__(self, 
                 engine_merge: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 engine_query: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.merge_queries = dspy.Predict(MergeQueries)
        self.generate_queries = dspy.Predict(QuestionToQuery)

        self.engine_merge = engine_merge
        self.engine_query = engine_query

    def _parsse_queries(self, queries):
        return [q.replace("-", "").strip().strip('"').strip('"').strip() for q in queries.split("\n")]


    def forward(self, 
                topic, 
                draft_attributes, 
                attributes_dict, 
                callback_handler: BaseCallbackHandler = None):
        
        with dspy.settings.context(lm=self.engine_query):
            queries_dict = {
                title: self._parsse_queries(
                    self.generate_queries(topic=title, attributes="\n".join(attributes)).queries
                )
                for title, attributes in attributes_dict.items()
            }
            draft_queries = self._parsse_queries(
                self.generate_queries(topic=topic, attributes="\n".join(draft_attributes)).queries
            )

        with dspy.settings.context(lm=self.engine_merge):
            additional_info = ""
            for title,queries in queries_dict.items():
                additional_info += f"Topic: {title}\nQueries:\n"
                for q in queries:
                    additional_info += f"- {q}\n"
                additional_info += "\n"

            inital_qeuries = ""
            for q in draft_queries:
                inital_qeuries += f"- {q}\n"

            merged_queries = self.merge_queries(topic=topic,draft_queries=inital_qeuries, queries_dict=additional_info).output

            merged_queries = self._parsse_queries(merged_queries)
        



        return dspy.Prediction(merged_queries=merged_queries)

class QuestionToQuery(dspy.Signature):
    """You want to search the info of attributes of the topic using Google search. What do you type in the search box?
        Write the queries you will use in the following format:
        - query 1
        - query 2
        ...
        - query n"""

    topic = dspy.InputField(prefix='Topic you are discussing about: ', format=str)
    attributes = dspy.InputField(prefix='The attributes of the topic: ', format=str)
    queries = dspy.OutputField(prefix='Now give your response. Make sure that only queries are output. Do not repeat the input prompt', format=str)

class MergeQueries(dspy.Signature):
    """I want you to act as a researcher gathering information to compose a wiki article on a specific topic. You are now presented with a topic and a list of queries designed to gather information for the topic. Your task is to modify or enhance the query list based on the relevant topics and their queries. Ensure that the final queries comprehensively encompass information beneficial for writing about the topic and are suitable for use in Google searches. Do not repeat the input prompt.

        Here is the format of your response:
        - query 1
        - query 2
        ...
        - query n
    """
    topic = dspy.InputField(prefix="The topic you are discussing about: ", format=str)
    draft_queries = dspy.InputField(prefix="The queries of the topic: ", format=str)
    queries_dict = dspy.InputField(prefix="The similar topics with their queries: \n", format=str)

    output = dspy.OutputField(prefix="The final response of the queries: ", format=str)

class RapidKnowledgeCurationModule(KnowledgeCurationModule):
    """
    The interface for knowledge curation stage. Given topic, return collected information.
    """

    def __init__(self,
                 retriever: Retriever,
                 outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 attribute_extract_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 info_merge_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 search_top_k: int,
                 max_thread_num: int):
        """
        Store args and finish initialization.
        """
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.draft_outline_generator = WriteOutline(engine=outline_gen_lm) 
        self.attribute_extractor = AttributeExtractor(engine=attribute_extract_lm) # 需要用GPT4o，mini只会复读section title
        self.info_merger =InfoMerger(engine_merge=info_merge_lm, engine_query=question_asker_lm)

    def research(self,
                 topic: str,
                 brief_intro: str,
                 similar_outlines: dict,
                 ground_truth_url: str,
                 callback_handler: BaseCallbackHandler,
                 return_conversation_log=False) -> Union[RapidInformationTable, Tuple[RapidInformationTable, Dict]]:
        """
        Curate information and knowledge for the given topic

        Args:
            topic: topic of interest in natural language.
        
        Returns:
            collected_information: collected information in InformationTable type.
        """

        # generate draft outline
        draft_outline = self.draft_outline_generator(topic=topic,
                                                     brief_intro=brief_intro,
                                                     similar_outlines=similar_outlines)
        
        information_table = None

        # analyze attribute & generate attribute for outline
            # extract attribute from similar topics
                # attribute extractor: outline to attribute
        attributes_dict = {}

        for title,outline in similar_outlines.items():
            attributes_dict[title] = self.attribute_extractor(topic=title,outline=outline).attributes


        draft_attributes = self.attribute_extractor(topic=topic,outline=draft_outline.outline).attributes


        merged_queries = self.info_merger(topic, 
                                       draft_attributes, 
                                       attributes_dict).merged_queries

        

        if topic not in merged_queries:
            merged_queries.append(topic)


        searched_results= self.retriever.retrieve(
            list(set(merged_queries)), exclude_urls=[ground_truth_url]
        )

        information_table = RapidInformationTable(search_results=searched_results)

        return information_table, draft_outline, draft_attributes, attributes_dict, merged_queries
