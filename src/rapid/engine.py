import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy
from .modules.topic_clarify import RapidTopicClarifyModule
from .modules.article_generation import RapidArticleGenerationModule
from .modules.topo_article_generation import TopoArticleGenerationModule
from .modules.article_polish import RapidArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.info_discovery import RapidKnowledgeCurationModule
from .modules.outline_generation import RapidOutlineGenerationModule
from .modules.retriever import RapidRetriever
from .modules.dataclass import RapidInformationTable, RapidArticle
from ..interface import Engine, LMConfigs
from ..lm import OpenAIModel
from ..utils import FileIOHelper, makeStringRed


class RapidWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of Rapid.

    Given that different parts in Rapid framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        self.question_asker_lm = None  # LLM used in question asking.
        self.plan_gen_lm = None 
        self.outline_gen_lm = None  # LLM used in outline generation.
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.

    def init_openai_model(
            self,
            openai_api_key: str,
            openai_type: Literal["openai", "azure"],
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            temperature: Optional[float] = 1.0,
            top_p: Optional[float] = 0.9
    ):

        openai_kwargs = {
            'api_key': openai_api_key,
            'api_provider': openai_type,
            'temperature': temperature,
            'top_p': top_p,
            'api_base': api_base
        }
        if openai_type and openai_type == 'openai':
            self.question_asker_lm = OpenAIModel(model='gpt-3.5-turbo',
                                                 max_tokens=500, **openai_kwargs)
            self.outline_gen_lm = OpenAIModel(model='gpt-4-0125-preview',
                                              max_tokens=400, **openai_kwargs)
            self.article_gen_lm = OpenAIModel(model='gpt-4o-2024-05-13',
                                              max_tokens=700, **openai_kwargs)
            self.article_polish_lm = OpenAIModel(model='gpt-4o-2024-05-13',
                                                 max_tokens=4000, **openai_kwargs)
        else:
            logging.warning('No valid OpenAI API provider is provided. Cannot use default LLM configurations.')

    def set_topic_clarify_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.topic_clarify_lm = model

    def set_attribute_extract_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.attribute_extract_lm = model

    def set_info_merge_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.attribute_merge_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_plan_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.plan_gen_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class RapidWikiRunnerArguments:
    """Arguments for controlling the Rapid Wiki pipeline."""
    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of questions in conversational question asking."},
    )
    max_perspective: int = field(
        default=3,
        metadata={"help": "Maximum number of perspectives to consider in perspective-guided question asking."},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={"help": "Maximum number of threads to use. "
                          "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."},
    )
    wiki_dump_path: str = field(
        default='./wiki_dump/original/combined.jsonl',
        metadata={"help": "Path to the wiki dump file."},
    )
    wiki_encode_path: str = field(
        default='./wiki_dump/encode/merged_encoded_vectors.pkl',
        metadata={"help": "Path to the wiki encode file."},
    )



class RapidWikiRunner(Engine):
    """Rapid Wiki pipeline runner."""

    def __init__(self,
                 args: RapidWikiRunnerArguments,
                 lm_configs: RapidWikiLMConfigs,
                 rm,
                 wiki_dump):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs



        self.retriever = RapidRetriever(rm=rm, k=self.args.retrieve_top_k)
        
        self.Rapid_topic_clarify_module = RapidTopicClarifyModule(
            retriever=self.retriever,
            wiki_dump=wiki_dump,
            search_top_k=self.args.retrieve_top_k,
            topic_clarify_lm=self.lm_configs.topic_clarify_lm
        )
        self.Rapid_knowledge_curation_module = RapidKnowledgeCurationModule(
            retriever=self.retriever,
            attribute_extract_lm=self.lm_configs.attribute_extract_lm,
            info_merge_lm=self.lm_configs.attribute_merge_lm,
            outline_gen_lm=self.lm_configs.outline_gen_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            search_top_k=self.args.search_top_k,
            max_thread_num=self.args.max_thread_num
        )
        self.Rapid_outline_generation_module = RapidOutlineGenerationModule(
            outline_refine_lm=self.lm_configs.outline_gen_lm
        )
        self.Rapid_article_generation = RapidArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num
        )

        self.topo_article_generation = TopoArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            plan_gen_lm=self.lm_configs.plan_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )

        self.Rapid_article_polishing_module = RapidArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_topic_clarify_moudle(self,
                                 topic: str,
                                 ground_truth_url: str,):
        brief_intro,similar_outlines,similar_intros = self.Rapid_topic_clarify_module.clarify(topic=topic,
                                                                                              ground_truth_url=ground_truth_url)
        with open(os.path.join(self.article_output_dir, 'brief_intro.txt'), 'w') as f:
            f.write(brief_intro)

        with open(os.path.join(self.article_output_dir, 'similar_outlines.json'), 'w') as outlines_file:
            json.dump(similar_outlines, outlines_file, ensure_ascii=False, indent=4)

        with open(os.path.join(self.article_output_dir, 'similar_intros.json'), 'w') as intros_file:
            json.dump(similar_intros, intros_file, ensure_ascii=False, indent=4)

        return brief_intro, similar_outlines,similar_intros

    def run_knowledge_curation_module(self,
                                      ground_truth_url: str = "None",
                                      brief_intro: str= "None",
                                      similar_outlines: dict={},
                                      callback_handler: BaseCallbackHandler = None) -> RapidInformationTable:

        information_table, draft_outline, draft_attributes, attributes_dict, merged_queries= self.Rapid_knowledge_curation_module.research(
            topic=self.topic,
            brief_intro=brief_intro,
            similar_outlines=similar_outlines,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            return_conversation_log=True
        )
        article_with_draft_outline_only = RapidArticle.from_outline_str(topic=self.topic,
                                                                        outline_str=draft_outline.outline)
        FileIOHelper.dump_json({"draft_attributes": draft_attributes}, os.path.join(self.article_output_dir, 'draft_attributes.json'))
        FileIOHelper.dump_json(attributes_dict, os.path.join(self.article_output_dir, 'attributes_dict.json'))
        FileIOHelper.dump_json({"merged_queries": merged_queries}, os.path.join(self.article_output_dir, 'merged_queries.json'))
        information_table.dump_url_to_info(os.path.join(self.article_output_dir, 'raw_search_results.json'))
        article_with_draft_outline_only.dump_outline_to_file(os.path.join(self.article_output_dir, "direct_gen_outline.txt"))
        return information_table, article_with_draft_outline_only

    def run_outline_generation_module(self,
                                      information_table: RapidInformationTable,
                                      outline: RapidArticle,
                                      callback_handler: BaseCallbackHandler = None) -> RapidArticle:

        operations, refined_outline = self.Rapid_outline_generation_module.generate_outline(
            topic=self.topic,
            outline=outline,
            information_table=information_table,
            callback_handler=callback_handler
        )
        refined_outline.dump_outline_to_file(os.path.join(self.article_output_dir, 'refined_outline.txt'))
        # Write operations to a file
        operations_file_path = os.path.join(self.article_output_dir, 'outline_operations.txt')
        with open(operations_file_path, 'w') as f:
            f.write(operations)
        return refined_outline

    def run_article_generation_module(self,
                                      outline: RapidArticle,
                                      information_table=RapidInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> RapidArticle:

        draft_article = self.Rapid_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler
        )
        draft_article.dump_article_as_plain_text(os.path.join(self.article_output_dir, 'Rapid_gen_article.txt'))
        draft_article.dump_reference_to_file(os.path.join(self.article_output_dir, 'url_to_info.json'))
        return draft_article
    
    def run_topo_article_generation_module(self,
                                           outline: RapidArticle,
                                           information_table=RapidInformationTable,
                                           callback_handler: BaseCallbackHandler = None) -> RapidArticle:

        draft_article = self.topo_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler
        )
        

        draft_article.dump_plan_to_file(os.path.join(self.article_output_dir, 'plan.json'))
        draft_article.dump_topo_order_to_file(os.path.join(self.article_output_dir, 'topo_order.json'))
        draft_article.dump_article_as_plain_text(os.path.join(self.article_output_dir, 'topo_gen_article.txt'))
        draft_article.dump_reference_to_file(os.path.join(self.article_output_dir, 'url_to_info.json'))
        return draft_article


    def run_article_polishing_module(self,
                                     draft_article: RapidArticle,
                                     remove_duplicate: bool = False) -> RapidArticle:

        polished_article = self.Rapid_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=False
        )
        FileIOHelper.write_str(polished_article.to_string(),
                               os.path.join(self.article_output_dir, 'topo_gen_article_polished.txt'))
        return polished_article

    def post_run(self):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(config_log, os.path.join(self.article_output_dir, 'run_config.json'))

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(os.path.join(self.article_output_dir, 'llm_call_history.jsonl'), 'w') as f:
            for call in llm_call_history:
                if 'kwargs' in call:
                    call.pop('kwargs')  # All kwargs are dumped together to run_config.json.
                f.write(json.dumps(call) + '\n')

    def _load_information_table_from_local_fs(self, information_table_local_path):
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-research argument to prepare the conversation_log.json for this topic.")
        return RapidInformationTable.from_search_results_file(information_table_local_path)

    def _load_outline_from_local_fs(self, topic, outline_local_path):
        assert os.path.exists(outline_local_path), makeStringRed(
            f"{outline_local_path} not exists. Please set --do-generate-outline argument to prepare the Rapid_gen_outline.txt for this topic.")
        return RapidArticle.from_outline_file(topic=topic, file_path=outline_local_path)

    def _load_draft_article_from_local_fs(self, topic, draft_article_path, url_to_info_path):
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} not exists. Please set --do-generate-article argument to prepare the Rapid_gen_article.txt for this topic.")
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} not exists. Please set --do-generate-article argument to prepare the url_to_info.json for this topic.")
        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)
        return RapidArticle.from_string(topic_name=topic, article_text=article_text, references=references)

    def run(self,
            topic: str,
            ground_truth_url: str = '',
            do_clarify: bool = True,
            do_research: bool = True,
            do_generate_outline: bool = True,
            do_generate_article: bool = True,
            do_topo_generation: bool = True,
            do_polish_article: bool = True,
            remove_duplicate: bool = False,
            callback_handler: BaseCallbackHandler = BaseCallbackHandler()):
        """
        Run the Rapid pipeline.

        Args:
            topic: The topic to research.
            ground_truth_url: A ground truth URL including a curated article about the topic. The URL will be excluded.
            do_research: If True, research the topic through information-seeking conversation;
             if False, expect conversation_log.json and raw_search_results.json to exist in the output directory.
            do_generate_outline: If True, generate an outline for the topic;
             if False, expect Rapid_gen_outline.txt to exist in the output directory.
            do_generate_article: If True, generate a curated article for the topic;
             if False, expect Rapid_gen_article.txt to exist in the output directory.
            do_polish_article: If True, polish the article by adding a summarization section and (optionally) removing
             duplicated content.
            remove_duplicate: If True, remove duplicated content.
            callback_handler: A callback handler to handle the intermediate results.
        """
        assert do_clarify or do_research or do_generate_outline or do_generate_article or do_polish_article, \
            makeStringRed(
                "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article")

        self.topic = topic
        self.article_dir_name = topic.replace(' ', '_').replace('/', '_')
        self.article_output_dir = os.path.join(self.args.output_dir, self.article_dir_name)
        os.makedirs(self.article_output_dir, exist_ok=True)

        brief_intro=""

        if do_clarify:
            brief_intro,similar_outlines,similar_intros = self.run_topic_clarify_moudle(topic=topic,ground_truth_url=ground_truth_url)


        # research module
        information_table: RapidInformationTable = None
        outline: RapidArticle = None
        if do_research:
            if brief_intro == "":
                brief_intro = FileIOHelper.load_str(os.path.join(self.article_output_dir, 'brief_intro.txt'))
            information_table, outline = self.run_knowledge_curation_module(ground_truth_url=ground_truth_url,
                                                                   brief_intro=brief_intro,
                                                                   similar_outlines=similar_outlines,
                                                                   callback_handler=callback_handler)
            

        # outline generation module

        if do_generate_outline:
            # load information table if it's not initialized
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, 'raw_search_results.json'))
            if outline is None:
                outline = self._load_outline_from_local_fs(topic=topic,
                                                           outline_local_path=os.path.join(self.article_output_dir,
                                                                                           'direct_gen_outline.txt'))    
                                                                                           
            # information_table.prepare_table_for_retrieval()          
            outline = self.run_outline_generation_module(information_table=information_table,
                                                         outline=outline,
                                                         callback_handler=callback_handler)


        # article generation module
        draft_article: RapidArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, 'raw_search_results.json'))
            if outline is None:
                outline = self._load_outline_from_local_fs(topic=topic,
                                                           outline_local_path=os.path.join(self.article_output_dir,
                                                                                           'direct_gen_outline.txt'))   

            if do_topo_generation:
                draft_article = self.run_topo_article_generation_module(outline=outline,
                                                                        information_table=information_table,
                                                                        callback_handler=callback_handler)
            else:
                draft_article = self.run_article_generation_module(outline=outline,
                                                               information_table=information_table,
                                                               callback_handler=callback_handler)

        # article polishing module
        if do_polish_article:
            if draft_article is None:
                draft_article_path = os.path.join(self.article_output_dir, 'topo_gen_article.txt')
                url_to_info_path = os.path.join(self.article_output_dir, 'url_to_info.json')
                draft_article = self._load_draft_article_from_local_fs(topic=topic,
                                                                       draft_article_path=draft_article_path,
                                                                       url_to_info_path=url_to_info_path)
            self.run_article_polishing_module(draft_article=draft_article, remove_duplicate=False)
