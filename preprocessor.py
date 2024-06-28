"""
 ****************************************************************************

    @file       preprocessor.py

    @author     Achintha Ihalage

    @brief      This module is responsible for acquiring various Sinhala
                datasets from Hugging Face Datasets Hub and transforming
                them into a common format to be consumed for supervised
                finetuning of LLMs. 

 ****************************************************************************
"""
import re
import json
import pandas as pd
from enum import Enum
from datasets import Dataset, load_dataset, concatenate_datasets

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class SinhalaDatasets(Enum):
    """
    Enum class to identify which Sinhala datasets to be used.
    """
    SINHALA_ELI5 = "9wimu9/sinhala_eli5"
    SINHALA_LLAMA_FORMAT = "Thimira/sinhala-llm-dataset-llama-prompt-format"
    SINHALA_ALPACA = "sahanruwantha/alpaca-sinhala"
    SINHALA_CNN_DAILY_MAIL = "Hamza-Ziyard/CNN-Daily-Mail-Sinhala"
    SINHALA_DATABRICKS_DOLLY = "Suchinthana/databricks-dolly-15k-sinhala"
    SINHALA_DENTAL_QNA = "KanishkaRandunu/SinhalaDentalQnA"


class PreprocessDataset:
    """
    Class to preprocess datasets in various formats.
    """
    def __init__(self,
                dataset_name: str = None,
                transform_all_datasets: bool = False
                ):
        if dataset_name and transform_all_datasets:
            raise ValueError("Please provide only one of the arguments: dataset_name | transform_all_datasets")
        if not dataset_name and not transform_all_datasets:
            raise ValueError("Either dataset_name or transform_all_datasets must be provided!")
        
        ## to check if unknown dataset is provided.
        valid_dataset_type_list = list(
            map(lambda x: x.value, SinhalaDatasets._member_map_.values())
        )
            
        
        if dataset_name == "9wimu9/sinhala_eli5" or "eli5" in dataset_name.lower():
            self.dataset_name = SinhalaDatasets.SINHALA_ELI5
        elif dataset_name == "Thimira/sinhala-llm-dataset-llama-prompt-format" or "sinhala_llama" in dataset_name.lower():
            self.dataset_name = SinhalaDatasets.SINHALA_LLAMA_FORMAT
        elif dataset_name == "sahanruwantha/alpaca-sinhala" or "sinhala_alpaca" in dataset_name.lower():
            self.dataset_name = SinhalaDatasets.SINHALA_ALPACA
        elif dataset_name == "Hamza-Ziyard/CNN-Daily-Mail-Sinhala" or "sinhala_cnn" in dataset_name.lower():
            self.dataset_name = SinhalaDatasets.SINHALA_CNN_DAILY_MAIL
        elif dataset_name == "Suchinthana/databricks-dolly-15k-sinhala" or "sinhala_databricks" in dataset_name.lower():
            self.dataset_name = SinhalaDatasets.SINHALA_DATABRICKS_DOLLY
        elif dataset_name == "KanishkaRandunu/SinhalaDentalQnA" or "sinhala_dental" in dataset_name.lower():
            self.dataset_name = SinhalaDatasets.SINHALA_DENTAL_QNA
        elif dataset_name and dataset_name not in valid_dataset_type_list:
            raise ValueError(
                f"The provided dataset may not be supported. The supported datasets are: {valid_dataset_type_list}"
            )
        
        self.transform_all_datasets = transform_all_datasets

    def concat_multisplit_dataset(self,
                        dataset: Dataset):
        """
        Concatenate multiple splits of a dataset to create a single HF dataset.
        Args:
            datasets_list (Dataset): HF dataset which may have multiple splits
        """
        ## check if dataset has multiple splits
        if isinstance(dataset, dict):
            all_splits = [dataset[split_name] for split_name in dataset.keys()]
            concatenated_dataset = concatenate_datasets(all_splits)
        else:
            concatenated_dataset = dataset

        return concatenated_dataset


    def process_eli5_dataset(self,
                             system_prompt: str = ""):
        """
        Transform sinhala_eli5 dataset into a common instruction
        finetuning dataset format.

        Args:
            system_prompt (str, optional): a system prompt if required
        Returns:
            HF Dataset
        """
        dataset = load_dataset(SinhalaDatasets.SINHALA_ELI5.value)
        # logger.info(f"Dataset before processing: {dataset}")
        concatenated_dataset = self.concat_multisplit_dataset(dataset)

        ## add a specific system_prompt if needed
        concatenated_dataset = concatenated_dataset.map(lambda example: 
                                                        {"question_prompt": system_prompt + example["title"] if example["title"] else ""})
        concatenated_dataset = concatenated_dataset.map(lambda example: 
                                                        {"response_prompt": example["text"][0] if isinstance(example["text"], list) 
                                                         else str(example["text"])})
        ## delete old columns
        concatenated_dataset = concatenated_dataset.remove_columns(dataset["train"].column_names)

        return concatenated_dataset
    
    def preprocess_sinhala_llama_dataset(self):
        """
        Transform sinhala-llm-dataset-llama-prompt-format dataset into a common instruction
        finetuning dataset format.

        Returns:
            HF Dataset
        """
        dataset = load_dataset(SinhalaDatasets.SINHALA_LLAMA_FORMAT.value)
        # logger.info(f"Dataset before processing: {dataset}")
        concatenated_dataset = self.concat_multisplit_dataset(dataset)

        ## remove uninformative samples
        text_to_exclude = "විස්තර සපයන්න" ## these prompts are very short and do not have enough content to answer.
        concatenated_dataset = concatenated_dataset.filter(lambda example: text_to_exclude not in example["text"])
        
        def extract_prompts(example):
            question_match = re.search(r'\[INST\](.*?)\[/INST\]', example['text'], re.DOTALL)
            question_prompt = question_match.group(1).strip() if question_match else ''
            
            ## remove unwanted string "</s>", <<SYS>>, <</SYS>>
            question_prompt = question_prompt.replace('</s>', '').replace('<<SYS>>', '').replace('<</SYS>>', '').strip()

            ## extract text after [/INST]
            response_prompt = example['text'].split('[/INST]')[-1].strip()

            ## remove unwanted string "</s>", <<SYS>>, <</SYS>>
            response_prompt = response_prompt.replace('</s>', '').replace('<<SYS>>', '').replace('<</SYS>>', '').strip()

            return {
                    'question_prompt': question_prompt,
                    'response_prompt': response_prompt
                }
        
        processed_dataset = concatenated_dataset.map(lambda example: extract_prompts(example))
        processed_dataset = processed_dataset.remove_columns(dataset["train"].column_names)

        return processed_dataset
    
    def preprocess_sinhala_alpaca_dataset(self):
        """
        Transform alpaca-sinhala dataset into a common instruction
        finetuning dataset format.

        This dataset is in miscellaneous format. The instruction, input and output are recorded in immediate rows.

        Returns:
            HF Dataset
        """
        dataset = load_dataset(SinhalaDatasets.SINHALA_ALPACA.value)
        # logger.info(f"Dataset before processing: {dataset}")

        ## this dataset is in miscellaneous format. The instruction, input and output are recorded in immediate rows.
        df = dataset['train'].to_pandas()

        question_prompt = []
        response_prompt = []

        # iterate over the DataFrame in steps of 3 to process triplets
        # make sure that the correct instruction, input and output are clubbed
        for i in range(0, len(df), 3):
            if i+2 < len(df):
                instruction = df.iloc[i]['text']
                input_text = df.iloc[i+1]['text']
                output_text = df.iloc[i+2]['text']
                
                if "Translated Instruction" in instruction and "Translated Input" in input_text and "Translated Output" in output_text:
                    instruction = instruction.replace('Translated Instruction: ', '')
                    input_text = input_text.replace('Translated Input: ', '')
                    output_text = output_text.replace('Translated Output: ', '')
                    question_prompt.append(instruction + " " + input_text)
                    response_prompt.append(output_text)
                
                ## counterintuitive. input_text contains output. For some examples, Translated Input is unavailable. Some samples are missed.
                elif "Translated Instruction" in instruction and "Translated Output" in input_text: 
                    instruction = instruction.replace('Translated Instruction: ', '')
                    output_text = input_text.replace('Translated Output: ', '')
                    question_prompt.append(instruction)
                    response_prompt.append(output_text)
                else:
                    continue
        new_df = pd.DataFrame({
                    'question_prompt': question_prompt,
                    'response_prompt': response_prompt
                })
        new_dataset = Dataset.from_pandas(new_df)

        return new_dataset
    
    def preprocess_sinhala_cnn_dataset(self):
        """
        Transform CNN-Daily-Mail-Sinhala dataset into a common instruction
        finetuning dataset format.

        Returns:
            HF Dataset
        """
        dataset = load_dataset(SinhalaDatasets.SINHALA_CNN_DAILY_MAIL.value)
        # logger.info(f"Dataset before processing: {dataset}")
        concatenated_dataset = self.concat_multisplit_dataset(dataset)

        def extract_article_and_summary(example):
            article = example["article_sinhala"]
            summary = example["summary_sinhala"]

            question_prompt = "පහත ඡේදය සාරාංශ කරන්න. " + article
            response_prompt = summary
            
            return {
                    'question_prompt': question_prompt,
                    'response_prompt': response_prompt
                }
        
        processed_dataset = concatenated_dataset.map(lambda example: extract_article_and_summary(example))
        processed_dataset = processed_dataset.remove_columns(dataset["train"].column_names)

        return processed_dataset
    
    def preprocess_sinhala_databricks_dataset(self):
        """
        Transform databricks-dolly-15k-sinhala dataset into a common instruction
        finetuning dataset format.

        Returns:
            HF Dataset
        """
        dataset = load_dataset(SinhalaDatasets.SINHALA_DATABRICKS_DOLLY.value)
        # logger.info(f"Dataset before processing: {dataset}")
        concatenated_dataset = self.concat_multisplit_dataset(dataset)

        concatenated_dataset = concatenated_dataset.map(lambda example: 
                                                        {"question_prompt": example["instruction"] + example[" context"] ## note the space
                                                         if example[" context"] else example["instruction"]})
        concatenated_dataset = concatenated_dataset.map(lambda example: 
                                                        {"response_prompt": example[" response"]})
        ## delete old columns
        concatenated_dataset = concatenated_dataset.remove_columns(dataset["train"].column_names)

        return concatenated_dataset
    
    def preprocess_sinhala_dental_dataset(self):
        """
        Transform SinhalaDentalQnA dataset into a common instruction
        finetuning dataset format.

        Returns:
            HF Dataset
        """
        dataset = load_dataset(SinhalaDatasets.SINHALA_DENTAL_QNA.value)
        # logger.info(f"Dataset before processing: {dataset}")
        concatenated_dataset = self.concat_multisplit_dataset(dataset)

        concatenated_dataset = concatenated_dataset.map(lambda example: 
                                                        {"question_prompt": example["Question"], "response_prompt": example["Answer"]})
        ## delete old columns
        concatenated_dataset = concatenated_dataset.remove_columns(dataset["train"].column_names)

        return concatenated_dataset
        
    def preprocess_datasets(self,
             ):
        """
        Main preprocessor to call individual preprocessing functions.
        """
        if self.transform_all_datasets:
            logger.info("\nProcessing all datasets ...\n\n")
            
            dataset_eli5 = self.process_eli5_dataset()
            logger.info(f"Finished processing eli5 dataset: \n{dataset_eli5}\n\n")

            dataset_llama = self.preprocess_sinhala_llama_dataset()
            logger.info(f"Finished processing llama dataset: \n{dataset_llama}\n\n")

            dataset_alpaca = self.preprocess_sinhala_alpaca_dataset()
            logger.info(f"Finished processing alpaca dataset: \n{dataset_alpaca}\n\n")

            dataset_cnn = self.preprocess_sinhala_cnn_dataset()
            logger.info(f"Finished processing cnn dataset: \n{dataset_cnn}\n\n")

            dataset_databricks = self.preprocess_sinhala_databricks_dataset()
            logger.info(f"Finished processing databricks dataset: \n{dataset_databricks}\n\n")

            dataset_dental = self.preprocess_sinhala_dental_dataset()
            logger.info(f"Finished processing dental dataset: \n{dataset_dental}\n\n")

            dataset_list = [dataset_eli5, dataset_llama, dataset_alpaca, dataset_cnn, dataset_databricks, dataset_dental]

            dataset = concatenate_datasets(dataset_list)

            ## add sample_id column (optional)
            df = dataset.to_pandas()
            df = df.iloc[3:]
            df.insert(0, 'sample_id', range(1, len(df) + 1))
            final_dataset = Dataset.from_pandas(df)

            print(f"\nFinal dataset: {final_dataset}\n")
            # print(final_dataset[-1])
            dataset_file_path = "datasets/sinhala-instruction-finetune-large.jsonl"
            # final_dataset.to_json(dataset_file_path, orient='records', lines=True)
            with open(dataset_file_path, 'w', encoding='utf-8') as f:
                for entry in final_dataset:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        else:
            if self.dataset_name == SinhalaDatasets.SINHALA_ELI5:
                dataset = self.process_eli5_dataset()
                logger.info(f"Dataset after processing: {dataset}")

            if self.dataset_name == SinhalaDatasets.SINHALA_LLAMA_FORMAT:
                dataset = self.preprocess_sinhala_llama_dataset()
                logger.info(f"Dataset after processing: {dataset}")

            if self.dataset_name == SinhalaDatasets.SINHALA_ALPACA:
                dataset = self.preprocess_sinhala_alpaca_dataset()
                logger.info(f"Dataset after processing: {dataset}")

            if self.dataset_name == SinhalaDatasets.SINHALA_CNN_DAILY_MAIL:
                dataset = self.preprocess_sinhala_cnn_dataset()
                logger.info(f"Dataset after processing: {dataset}")

            if self.dataset_name == SinhalaDatasets.SINHALA_DATABRICKS_DOLLY:
                dataset = self.preprocess_sinhala_databricks_dataset()
                logger.info(f"Dataset after processing: {dataset}")


            if self.dataset_name == SinhalaDatasets.SINHALA_DENTAL_QNA:
                dataset = self.preprocess_sinhala_dental_dataset()
                logger.info(f"Dataset after processing: {dataset}")


if __name__ == "__main__":
    preprocessor = PreprocessDataset(dataset_name="", transform_all_datasets=True)
    preprocessor.preprocess_datasets()
    # print(SinhalaDatasets.SINHALA_ALPACA)