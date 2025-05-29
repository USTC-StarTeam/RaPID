import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .persona_generator import RapidPersonaGenerator
from .dataclass import RapidInformationTable, RapidInformation, WikiDump
from ...interface import TopicClarifyModule, Retriever
from ...utils import ArticleTextProcessing

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    streamlit_connection = True
except ImportError as err:
    streamlit_connection = False

script_dir = os.path.dirname(os.path.abspath(__file__))


class SummarizeTopic(dspy.Signature):
    """You are an expert in utilizing search engines effectively. You are currently compiling information for a wiki page based on a given topic. Now that you have obtained content returned by search engines regarding this topic, please analyze whether there are any ambiguities or multiple concepts associated with it. If the topic is clear, generate a brief introduction based on the search engine content to clarify the concept for subsequent writing, ensuring that the introduction remains within 3 sentences."""

    topic = dspy.InputField(prefix='Topic you are discussing about:', format=str)
    info = dspy.InputField(
        prefix='Gathered information from search engines:\n', format=str)
    answer = dspy.OutputField(
        prefix='Now give your response. Make sure that only the introduction is outputted. Do not include the input content.\n',
        format=str
    )



class TopicClarifier(dspy.Module):

    def __init__(self,engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],retriever, search_top_k):
        self.retriever = retriever
        self.search_top_k = search_top_k
        self.summarize_topic = dspy.Predict(SummarizeTopic)

        self.engine = engine

    def forward(self, topic, ground_truth_url):
        
        # first search
        searched_results = self.retriever.retrieve(topic,
                                                   exclude_urls=[ground_truth_url])
                    
        info = ''
        for n, r in enumerate(searched_results):
            info += '\n'.join(f'[{n + 1}]: {s}' for s in r.snippets[:1])
            info += '\n\n'

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1000)

        with dspy.settings.context(lm=self.engine):
            brief_summary=self.summarize_topic(topic=topic,info=info).answer
        return brief_summary


        

class RapidTopicClarifyModule(TopicClarifyModule):
    """
    The interface for Topic Clarify stage. Given an ambiguious topic, return clarified information.
    """

    def __init__(self,
                 retriever: Retriever,
                 wiki_dump: WikiDump,
                 topic_clarify_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 search_top_k: int = 5,
                 max_thread_num: int = 10):
        """
        Store args and finish initialization.
        """
        self.retriever = retriever
        self.wiki_dump = wiki_dump
        self.topic_clarify_lm = topic_clarify_lm
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.topic_clarifier = TopicClarifier(engine=self.topic_clarify_lm,
                                              retriever=self.retriever,
                                              search_top_k=self.search_top_k)


    def clarify(self, topic, ground_truth_url):
        brief_summary = self.topic_clarifier(topic=topic,
                                             ground_truth_url=ground_truth_url)
        similar_outlines, similar_intros = self.wiki_dump.retrieve(topic=topic,
                                                 top_k=3,
                                                 content=brief_summary)

        
        return brief_summary,similar_outlines,similar_intros