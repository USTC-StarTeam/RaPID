import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union
import queue


import dspy
from .callback import BaseCallbackHandler
from .dataclass import RapidInformationTable, RapidArticle, RapidInformation
from ...interface import ArticleGenerationModule
from ...utils import ArticleTextProcessing


class TopoArticleGenerationModule(ArticleGenerationModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage, 
    """

    def __init__(self,
                 article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel] = None,
                 plan_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel] = None,
                 retrieve_top_k: int = 5,
                 max_thread_num: int = 10):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.plan_gen_lm = plan_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)
        self.plan_gen = TopoOrderGeneration(engine=self.plan_gen_lm)
    

    def generate_section(self, topic, section_name, information_table, section_outline, section_query):
        collected_info: List[RapidInformation] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(queries=section_query,
                                                                    search_top_k=self.retrieve_top_k)
        output = self.section_gen(topic=topic,
                                  outline=section_outline,
                                  section=section_name,
                                  collected_info=collected_info)
        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}

    def generate_section_with_memory(self, topic, section_name, information_table, previous_sections_content, section_outline, section_query):
        collected_info: List[RapidInformation] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(queries=section_query,
                                                                    search_top_k=self.retrieve_top_k)
        output = self.section_gen(topic=topic,
                                  outline=section_outline,
                                  previous_sections_content=previous_sections_content,
                                  section=section_name,
                                  collected_info=collected_info)
        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}



    def generate_article(self,
                         topic: str,
                         information_table: RapidInformationTable,
                         article_with_outline: RapidArticle,
                         callback_handler: BaseCallbackHandler = None) -> RapidArticle:
        """
        Generate article for the topic based on the information table and article outline.

        Args:
            topic (str): The topic of the article.
            information_table (RapidInformationTable): The information table containing the collected information.
            article_with_outline (RapidArticle): The article with specified outline.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the article generation process. Defaults to None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = RapidArticle(topic_name=topic)


        sections_to_write = article_with_outline.get_first_level_section_names()
        outline = article_with_outline.to_string().replace("\n\n","\n")
        
        article = copy.deepcopy(article_with_outline)

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(f'No outline for {topic}. Will directly search with the topic.')
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic]
            )
            section_output_dict_collection = [section_output_dict]
        else:
            
            plan = self.plan_gen(topic,outline,sections_to_write)
            topo_order = None

            if plan is not None:
                topo_order = self._toposort(sections_to_write,plan)

            if topo_order is not None:
                article_content = {}
                for section_title in topo_order:
                    if section_title.lower().strip() == 'introduction':
                            continue
                            # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                                'conclusion') or section_title.lower().strip().startswith('summary'):
                            continue
                    section_query = article_with_outline.get_outline_as_list(root_section_name=section_title,
                                                                                add_hashtags=False)        
                    queries_with_hashtags = article_with_outline.get_outline_as_list(root_section_name=section_title, 
                                                                                     add_hashtags=True)
                    section_outline = "\n".join(queries_with_hashtags)
                    

                    previous_sections = self._get_previous_sections(plan,section_title)

                    if previous_sections != []:
                        previous_sections_content = [article_content[title] for title in previous_sections if title in article_content]

                        current_section_result = self.generate_section_with_memory(topic=topic,
                                                                section_name=section_title,                                                                
                                                                information_table=information_table,
                                                                previous_sections_content=previous_sections_content,
                                                                section_outline=section_outline,
                                                                section_query=section_query)
                    else:
                        current_section_result = self.generate_section(topic=topic,
                                                                section_name=section_title,
                                                                information_table=information_table,
                                                                section_outline=section_outline,
                                                                section_query=section_query)
                    
                    article.update_section(parent_section_name=section_title,
                                            current_section_content=current_section_result["section_content"],
                                            current_section_info_list=current_section_result["collected_info"])
                    article_content[section_title] = current_section_result["section_content"]


            else:
                # fail in generating plan, switch to simple generation mode
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
                    future_to_sec_title = {}
                    for section_title in sections_to_write:
                        # We don't want to write a separate introduction section.
                        if section_title.lower().strip() == 'introduction':
                            continue
                            # We don't want to write a separate conclusion section.
                        if section_title.lower().strip().startswith(
                                'conclusion') or section_title.lower().strip().startswith('summary'):
                            continue
                        section_query = article_with_outline.get_outline_as_list(root_section_name=section_title,
                                                                                add_hashtags=False)
                        queries_with_hashtags = article_with_outline.get_outline_as_list(
                            root_section_name=section_title, add_hashtags=True)
                        section_outline = "\n".join(queries_with_hashtags)
                        future_to_sec_title[
                            executor.submit(self.generate_section,
                                            topic, section_title, information_table, section_outline, section_query)
                        ] = section_title

                    for future in as_completed(future_to_sec_title):
                        section_output_dict_collection.append(future.result())

                article = copy.deepcopy(article_with_outline)
                for section_output_dict in section_output_dict_collection:
                    article.update_section(parent_section_name=topic,
                                        current_section_content=section_output_dict["section_content"],
                                        current_section_info_list=section_output_dict["collected_info"])
                
        article.post_processing()
        article.plan = plan
        article.topo_order=topo_order
        return article
    
    def _toposort(self, vertices, adj) :
        degree = {v: 0 for v in vertices}
        for u, adj_u in adj.items() :
            for v in adj_u :
                degree[v] += 1
        
        Q = queue.Queue()
        for p, d in degree.items() :
            if d == 0 :
                Q.put(p)
        
        topo_order = []
        while not Q.empty() :
            p = Q.get()
            topo_order.append(p)
            for q in adj[p] :
                degree[q] -= 1
                if degree[q] == 0 :
                    Q.put(q)
                
        if len(topo_order) != len(vertices) :
            return None
        
        return topo_order
    def _get_previous_sections(self, plan, current_section):
        previous_sections = []
        for section in plan:
            if current_section in plan[section]:
                previous_sections.append(section)
        return previous_sections

class TopoOrderGeneration(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.GeneratePlan = dspy.Predict(GeneratePlan)
        self.engine = engine

    def forward(self, topic: str, outline: str, sections: List):

        with dspy.settings.context(lm=self.engine):
            plan = self.GeneratePlan(topic=topic, outline=outline).plan

        try:
            parsed_plan = self.parse_plan(plan=plan, sections=sections)
        except Exception as e:
            # Handle any exceptions that occur during parsing
            return None
        return parsed_plan
    
    def parse_plan(self, plan: str, sections: str):

        lines = plan.split('\n')
        lines = [x.lstrip('#') for x in lines if '<-' in x]
        
        graph = {x: [] for x in sections}
        
        def add_edge(u, v):
            if u == 'None':
                return  # Fixed indentation
            if u not in graph.keys() or v not in graph.keys() :
                if u not in graph.keys():
                    raise Exception(f"Failed to find node '{u}'")
                else:
                    raise Exception(f"Failed to find node '{v}'")  # Fixed indentation
            graph[u].append(v)
        
        for line in lines:
            L, R = line.split('<-')
            L = L.strip()
            for r in R.split(','):
                r = r.strip()  # Fixed indentation
                add_edge(r, L)
    
        return graph

class MemoryToSection(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section_with_memory = dspy.Predict(WriteSectionWithMemory)
        self.engine = engine

    def forward(self, topic: str, outline: str, section: str, collected_info: List[RapidInformation]):
        info = ''
        for idx, Rapid_info in enumerate(collected_info):
            info += f'[{idx + 1}]\n' + '\n'.join(Rapid_info.snippets)
            info += '\n\n'

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section_with_memory(topic=topic, info=info, section=section).output)

        return dspy.Prediction(section=section)


class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.write_section_with_memory = dspy.Predict(WriteSectionWithMemory)
        self.filter_info = dspy.Predict(FilterInfo)
        self.engine = engine

    def forward(self, topic: str, outline: str, section: str, 
                collected_info: List[RapidInformation], previous_sections_content=None):
        info = ''
        for idx, Rapid_info in enumerate(collected_info):
            info += f'[{idx + 1}]\n' + f'title: {Rapid_info.title}\n'+ f'url: {Rapid_info.url}\n'+'\n'.join(Rapid_info.snippets)
            info += '\n\n'
        
        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        with dspy.settings.context(lm=self.engine):
            if previous_sections_content is not None:
                # memory = self.format_memory(previous_sections_content=previous_sections_content)
                memory = '\n\n\n'.join(previous_sections_content)
                section = ArticleTextProcessing.clean_up_section(
                    self.write_section_with_memory(topic=topic,outline=outline, info=info, 
                                                   memory=memory, section=section).output)
            else:
                section = ArticleTextProcessing.clean_up_section(
                    self.write_section(topic=topic, outline=outline,info=info, section=section).output)

        return dspy.Prediction(section=section)
    
    def format_memory(self, previous_sections_content: dict ):
        memory = "\n\n".join([f"# {title}\n\n{content}" for title, content in previous_sections_content.items()])
        return memory


class FilterInfo(dspy.Signature):
    """Filter the collected information based on the given section outline. Only generate the id of the information that is helpful for writing the section and do not include any other information. The output format should be like the following:
        - [0]
        - [2]
        - [3]
    """
    topic = dspy.InputField(prefix="The topic of the whole page: ", format=str)
    info = dspy.InputField(prefix="The collected information: ", format=str)
    outline = dspy.InputField(prefix="The outline of the section: ", format=str)
    output = dspy.OutputField(
        prefix="Filter the collected information based on the given outline:\n",
        format=str
    )

class GeneratePlan(dspy.Signature):
    '''You are an experienced wiki writer. I will provide you with a topic with its outline to write. I want you to generate a writing plan for this outline to improve the coherence of the article. The plan defines which sections is needed to be generated before the current section. Try to choose the sections that can help improve the coherence and fluency of the current section. For example, sections like 'Background' don't need extra information while sections like 'Introduction' or 'Conclusion' need all other sections. Please just generate the plan for the first level sections and make sure that the plan is in a valid topological order. If no extra information is needed, generate "None". All the needed sections are connected by '<-' and make sure that they are all from the first level sections of outline. Just output the plan and do not explain.

        Here is an example:

        Topic: 2022_New_York_City_Subway_attack

        Outline:
        # Introduction
        # Background
        # Attack
        # Victims
        # Investigation
        # Perpetrator
        ## Background
        ## Extremist views
        ## Investigation and manhunt
        ## Arrest and charges
        # Aftermath
        # Reactions
        # See also

        Plan:
        # Introduction <- Attack,Victims,Investigation,Perpetrator,Aftermath,Reactions
        # Background <- None
        # Attack <- Background
        # Victims <- Attack
        # Investigation <- Background,Attack,Victims
        # Perpetrator <- Investigation,Attack,Victims
        # Aftermath <- Investigation,Perpetrator
        # Reactions <- Aftermath,Perpetrator
        # See Also <- None
    '''
    topic = dspy.InputField(prefix="Topic: ", format=str)
    outline = dspy.InputField(prefix="Outline: ", format=str)
    plan = dspy.OutputField(
        prefix="Generate the plan of the given topic and outline(do not repeat the outline):\n",
        format=str
    )



class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

        Here is the format of your writing:
            1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
            2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """
    

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    outline = dspy.InputField(prefix="The outline of the section: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str
    )

class WriteSectionWithMemory(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

        Here is the format of your writing:
            1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
            2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """
    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    memory = dspy.InputField(prefix="The other sections of the page: ")
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    outline = dspy.InputField(prefix="The outline of the section: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str
    )



    

    