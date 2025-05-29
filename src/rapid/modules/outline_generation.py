from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .dataclass import RapidInformationTable, RapidArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


class RapidOutlineGenerationModule(OutlineGenerationModule):
    """
    The interface for outline generation stage. Given topic, collected information from knowledge
    curation stage, generate outline for the article.
    """

    def __init__(self,
                 outline_refine_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_refine_lm = outline_refine_lm
        self.outline_refiner = OutlineRefiner(engine=outline_refine_lm)


    def generate_outline(self,
                         topic: str,
                         outline: RapidArticle,
                         information_table: RapidInformationTable,
                         callback_handler: BaseCallbackHandler = None):
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        titles = []
        for url in information_table.url_to_info:
            titles.append(information_table.url_to_info[url].title)

        outline_txt =  outline.to_string().replace("\n\n","\n")

        operations, refined_outline = self.outline_refiner(topic=topic, draft_outline=outline_txt, titles=titles, callback_handler=callback_handler) 


        return operations, refined_outline



class OutlineRefiner(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.engine = engine
        self.generation_operations = dspy.Predict(GenerateOperations)
        self.refine_outline = dspy.Predict(RefineOutline)


    def forward(self, topic: str, 
                draft_outline: str,
                titles,
                callback_handler: BaseCallbackHandler = None):

        draft_outline = ArticleTextProcessing.clean_up_outline(draft_outline)
        titles_input = "\n".join(titles)
        with dspy.settings.context(lm=self.engine):
            
            operations = self.generation_operations(topic=topic, outline=draft_outline, titles=titles_input).operations
    
            refined_outline = self.refine_outline(topic=topic, outline=draft_outline, titles=titles_input, operations=operations).refined_outline
            refined_outline = RapidArticle.from_outline_str(topic=topic,
                                                            outline_str=refined_outline)   
        return operations, refined_outline

    
    def _parse_operations(self, operations):
        operations = operations.split("\n")
        for i in range(len(operations)):
            if operations[i].startswith("[add section]"):
                operations[i] = ("add", operations[i].split(":")[1].strip())
            elif operations[i].startswith("[delete section]"):
                operations[i] = ("delete", operations[i].split(":")[1].strip())
            else:
                operations[i] = ("do nothing", "")
        return operations


class GenerateOperations(dspy.Signature):
    """You are improving an outline for a wiki page. Now I will give you a draft outline and some titles of the searched results. You can do three operations:
        add section
        delete section
        do nothing

    Please list the operations you need to do:
        [add section]: section_title
        [delete section]: section_title

    If nothing is needed to do, please just generate:
    [do nothing]

    Directly write the operations and do not include any other information and the template.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.InputField(prefix="The draft outline: ", format=str)
    titles = dspy.InputField(prefix="Titles of the searched results: ", format=str)

    operations = dspy.OutputField(prefix="Please generate the operations: ", format=str)

class RefineOutline(dspy.Signature):
    """You are improving an outline for a wiki page. Now I will give you a draft outline and some operations like:
        [add section]
        [delete section]
        [do nothing]
    Please proceed with the operations for the outline and then refine the overall outline. Directly write the refined outline and do not include any other information and the template.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.InputField(prefix="The draft outline: ", format=str)
    operations = dspy.InputField(prefix="The operations: \n", format=str)
    refined_outline = dspy.OutputField(prefix="Please generate the refined outline: ", format=str)

