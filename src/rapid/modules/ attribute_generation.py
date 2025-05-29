import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .dataclass import DialogueTurn, RapidInformationTable, RapidInformation
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
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.engine = engine

    def forward(self, topic: str, brief_intro: str, dlg_history, old_outline: Optional[str] = None,
                callback_handler: BaseCallbackHandler = None):


        with dspy.settings.context(lm=self.engine):
            if old_outline is None:
                old_outline = ArticleTextProcessing.clean_up_outline(self.draft_page_outline(topic=topic,brief_intro=brief_intro).outline)


        return dspy.Prediction(outline=outline, old_outline=old_outline)

class QuestionToQuery(dspy.Signature):
    """You want to answer the question using Google search. What do you type in the search box?
        Write the queries you will use in the following format:
        - query 1
        - query 2
        ...
        - query n"""

    topic = dspy.InputField(prefix='Topic you are discussing about: ', format=str)
    question = dspy.InputField(prefix='Question you want to answer: ', format=str)
    queries = dspy.OutputField(prefix='Now give your response. Make sure that only queries are output. Do not repeat the input prompt', format=str)



class RapidKnowledgeCurationModule(KnowledgeCurationModule):
    """
    The interface for knowledge curation stage. Given topic, return collected information.
    """

    def __init__(self,
                 retriever: Retriever,
                 persona_generator: Optional[RapidPersonaGenerator],
                 conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 max_search_queries_per_turn: int,
                 search_top_k: int,
                 max_conv_turn: int,
                 max_thread_num: int):
        """
        Store args and finish initialization.
        """
        self.retriever = retriever
        self.persona_generator = persona_generator
        self.conv_simulator_lm = conv_simulator_lm
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.conv_simulator = ConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn
        )

    def _get_considered_personas(self, topic: str, brief_intro, max_num_persona) -> List[str]:
        return self.persona_generator.generate_persona(topic=topic, brief_intro=brief_intro, max_num_persona=max_num_persona)

    def _run_conversation(self, conv_simulator, topic, ground_truth_url, considered_personas,
                          callback_handler: BaseCallbackHandler) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        Executes multiple conversation simulations concurrently, each with a different persona,
        and collects their dialog histories. The dialog history of each conversation is cleaned
        up before being stored.

        Parameters:
            conv_simulator (callable): The function to simulate conversations. It must accept four
                parameters: `topic`, `ground_truth_url`, `persona`, and `callback_handler`, and return
                an object that has a `dlg_history` attribute.
            topic (str): The topic of conversation for the simulations.
            ground_truth_url (str): The URL to the ground truth data related to the conversation topic.
            considered_personas (list): A list of personas under which the conversation simulations
                will be conducted. Each persona is passed to `conv_simulator` individually.
            callback_handler (callable): A callback function that is passed to `conv_simulator`. It
                should handle any callbacks or events during the simulation.

        Returns:
            list of tuples: A list where each tuple contains a persona and its corresponding cleaned
            dialog history (`dlg_history`) from the conversation simulation.
        """

        conversations = []

        def run_conv(persona):
            return conv_simulator(
                topic=topic,
                ground_truth_url=ground_truth_url,
                persona=persona,
                callback_handler=callback_handler
            )

        max_workers = min(self.max_thread_num, len(considered_personas))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_persona = {executor.submit(run_conv, persona): persona for persona in considered_personas}

            if streamlit_connection:
                # Ensure the logging context is correct when connecting with Streamlit frontend.
                for t in executor._threads:
                    add_script_run_ctx(t)

            for future in as_completed(future_to_persona):
                persona = future_to_persona[future]
                conv = future.result()
                conversations.append((persona, ArticleTextProcessing.clean_up_citation(conv).dlg_history))

        return conversations

    def research(self,
                 topic: str,
                 brief_intro: str,
                 ground_truth_url: str,
                 callback_handler: BaseCallbackHandler,
                 max_perspective: int = 0,
                 disable_perspective: bool = True,
                 return_conversation_log=False) -> Union[RapidInformationTable, Tuple[RapidInformationTable, Dict]]:
        """
        Curate information and knowledge for the given topic

        Args:
            topic: topic of interest in natural language.
        
        Returns:
            collected_information: collected information in InformationTable type.
        """

        # identify personas
        callback_handler.on_identify_perspective_start()
        considered_personas = []
        if disable_perspective:
            considered_personas = [""]
        else:
            considered_personas = self._get_considered_personas(topic=topic, brief_intro=brief_intro, max_num_persona=max_perspective)
        callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # run conversation 
        callback_handler.on_information_gathering_start()
        conversations = self._run_conversation(conv_simulator=self.conv_simulator,
                                               topic=topic,
                                               ground_truth_url=ground_truth_url,
                                               considered_personas=considered_personas,
                                               callback_handler=callback_handler)

        information_table = RapidInformationTable(conversations)
        callback_handler.on_information_gathering_end()
        if return_conversation_log:
            return information_table, RapidInformationTable.construct_log_dict(conversations)
        return information_table
