import sys
sys.path.append("/home/hcgu/projects/ACL25/wiki_generation/")

import os
import time
from argparse import ArgumentParser

from src.rapid import RapidWikiRunnerArguments, RapidWikiRunner, RapidWikiLMConfigs
from src.lm import OpenAIModel, AzureOpenAIModel
from src.rm import YouRM, BingSearch, GoogleRM
from src.utils import load_api_key
from src.rapid.modules.dataclass import WikiDump


from concurrent.futures import ThreadPoolExecutor  
from threading import Lock

import pandas as pd
from tqdm import tqdm
import json

def main(args):
    load_api_key(toml_file_path='/home/hcgu/projects/ACL25/wiki_generation/secrets.toml')
    lm_configs = RapidWikiLMConfigs()
    openai_kwargs = {
        'api_key': os.getenv("OPENAI_API_KEY"),
        'temperature': 1.0,
        'top_p': 0.9,
        'api_base': "https://api.starteam.wang/v1/",
    }

    ModelClass = OpenAIModel if os.getenv('OPENAI_API_TYPE') == 'openai' else AzureOpenAIModel


    gpt_4_model_name = 'gpt-4o'

    if os.getenv('OPENAI_API_TYPE') == 'azure':
        openai_kwargs['api_base'] = os.getenv('AZURE_API_BASE')
        openai_kwargs['api_version'] = os.getenv('AZURE_API_VERSION')

    topic_clarify_lm = ModelClass(model=gpt_4_model_name, max_tokens=500, **openai_kwargs)
    attribute_extract_lm = ModelClass(model=gpt_4_model_name, max_tokens=500, **openai_kwargs)
    info_merge_lm = ModelClass(model=gpt_4_model_name, max_tokens=1000, **openai_kwargs)
    question_asker_lm = ModelClass(model=gpt_4_model_name, max_tokens=500, **openai_kwargs)

    plan_gen_lm = ModelClass(model=gpt_4_model_name,max_tokens=1000, **openai_kwargs)
    outline_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=1000, **openai_kwargs)
    article_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
    article_polish_lm = ModelClass(model=gpt_4_model_name, max_tokens=4000, **openai_kwargs)

    lm_configs.set_topic_clarify_lm(topic_clarify_lm)    
    lm_configs.set_attribute_extract_lm(attribute_extract_lm)
    lm_configs.set_info_merge_lm(info_merge_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)

    lm_configs.set_plan_gen_lm(plan_gen_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    # load file in start scipt to save time for many topics
    wiki_dump = WikiDump(model_name="intfloat/e5-large-v2")
    wiki_dump.prepare_for_retrieval(dump_path="./wiki_dump/original/combined.jsonl",
                                    encode_path="./wiki_dump/encode/merged_encoded_vectors.pkl")
    
    engine_args = RapidWikiRunnerArguments(
        output_dir=args.output_dir,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    if args.retriever == 'bing':
        rm = BingSearch(bing_search_api=os.getenv('BING_SEARCH_API_KEY'), k=engine_args.search_top_k)
    elif args.retriever == 'you':
        rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=engine_args.search_top_k)
    elif args.retriever == 'google':
        rm = GoogleRM(google_api_key=os.getenv('GOOGLE_API_KEY'), google_cx=os.getenv('GOOGLE_CX'), k=engine_args.search_top_k)

    runner = RapidWikiRunnerArguments(engine_args, lm_configs, rm,wiki_dump=wiki_dump)

    if args.interface == 'console':
        topic = input('Topic: ')
        # topic = 'Alphafold'
        ground_truth_url = 'https://en.wikipedia.org/wiki/AlphaFold'
        runner.run(
            topic=topic,
            ground_truth_url=ground_truth_url,
            do_clarify=args.do_clarify,
            do_research=args.do_research,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_topo_generation=args.do_topo_generation,
            do_polish_article=args.do_polish_article,
        )
        runner.post_run()
        runner.summary(os.path.join(args.output_dir,topic.replace(' ', '_').replace('/', '_')))  # print usage and save to file

    else:
        with open(args.input_dir, 'r') as f:
            input_file = pd.read_csv(f)

        # add success flag to resume from the last failed topic
        success_flag_file = os.path.join(args.output_dir, 'success_flags.json')

        for topic in tqdm(input_file['topic']):


            # Load existing success flags if the file exists
            if os.path.exists(success_flag_file):
                with open(success_flag_file, 'r') as f:
                    success_flags = json.load(f)
            else:
                success_flags = {}

            
            if topic in success_flags and success_flags[topic]:
                print(f"Skipping already completed topic: {topic}")
                continue

            try:
                ground_truth_url = input_file.loc[input_file['topic'] == topic, 'url'].values[0]
                runner.run(
                    topic=topic,
                    ground_truth_url=ground_truth_url,
                    do_clarify=args.do_clarify,
                    do_research=args.do_research,
                    do_generate_outline=args.do_generate_outline,
                    do_generate_article=args.do_generate_article,
                    do_topo_generation=args.do_topo_generation,
                    do_polish_article=args.do_polish_article,
                )
                success_flags[topic] = True

            except Exception as e:
                print(f"Error processing topic {topic}: {e}")
                success_flags[topic] = False

                        # Save the success flags to file
            with open(success_flag_file, 'w') as f:
                json.dump(success_flags, f)
                
            runner.post_run()  # store log
            runner.summary(os.path.join(args.output_dir,topic.replace(' ', '_').replace('/', '_')))  # print usage and save to file
            time.sleep(5)


if __name__ == '__main__':
    parser = ArgumentParser()


    # global arguments
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to store the outputs.')
    parser.add_argument('--max-thread-num', type=int, default=3,
                        help='Maximum number of threads to use. The information seeking part and the article generation'
                             'part can speed up by using multiple threads. Consider reducing it if keep getting '
                             '"Exceed rate limit" error when calling LM API.')
    parser.add_argument('--retriever', type=str, choices=['bing', 'you', 'google'],
                        help='The search engine API to use for retrieving information.')
    parser.add_argument('--wiki-dump-path', type=str, default='./wiki_generation/wiki_dump/original/combined.jsonl',
                        help='Path to the wiki dump file for article generation.')
    parser.add_argument('--wiki-encode-path', type=str, default='./wiki_dump/encode/merged_encoded_vectors.pkl',
                        help='Path to the encoded topics file for further processing.')

    # stage of the pipeline
    parser.add_argument('--do-clarify', action='store_true',
                        help='If True, clarify the topic before proceeding with research and article generation.')
    parser.add_argument('--do-research', action='store_true',
                        help='If True, simulate conversation to research the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-outline', action='store_true',
                        help='If True, generate an outline for the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-article', action='store_true',
                        help='If True, generate an article for the topic; otherwise, load the results.')
    parser.add_argument('--do-topo-generation', action='store_true',
                        help='If True, perform topological generation of the article.')
    parser.add_argument('--do-polish-article', action='store_true',
                        help='If True, polish the article by adding a summarization section and (optionally) removing '
                             'duplicate content.')
    

    # hyperparameters for the pre-writing stage
    parser.add_argument('--search-top-k', type=int, default=10,
                        help='Top k search results to consider for each search query.')
    

    # hyperparameters for the writing stageß
    parser.add_argument('--retrieve-top-k', type=int, default=10,
                        help='Top k collected references for each section title.')

    parser.add_argument('--interface', type=str, choices=['console', 'file'], default='console')
    parser.add_argument('--input-dir', type=str, default='./FreshWiki-2024/final.csv')

    main(parser.parse_args())
