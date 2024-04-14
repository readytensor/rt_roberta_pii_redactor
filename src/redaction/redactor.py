import os
import re
import torch
import warnings
import pandas as pd
from config import paths
from tqdm import tqdm
from typing import List, Dict
from processing.dataset import dataset
from torch.utils.data import DataLoader
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from redaction.FakeGenerator import (
    FakeGenerator,
    get_real_fake_name_mapping,
    get_real_fake_entity_mapping,
    get_real_fake_date_mapping,
)
from utils import read_text_files, read_pdf_files, save_text_to_pdf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class Redactor:

    def __init__(
        self,
        model: RobertaForTokenClassification,
        tokenizer: RobertaTokenizerFast,
        label2id: dict,
        batch_size: int,
        random_state: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.batch_size = batch_size
        self.random_state = random_state
        self.fake = FakeGenerator(random_state=random_state)

    def create_overlapping_windows(
        self, token_list: list, window_size: int, overlap: int
    ) -> list:
        """
        Creates overlapping windows from the given list of tokens.

        Args:
            token_list (list): The list of tokens.
            window_size (int): The size of each window.
            overlap (int): The overlap between consecutive windows.

        Returns:
            list: A list of overlapping windows.

        This method iterates through the token list and creates windows of the specified size
        with the specified overlap between consecutive windows.
        """
        windows = []
        step = window_size - overlap
        for i in range(0, len(token_list), step):
            windows.append(token_list[i : i + window_size])
            # Make sure we don't create windows that are smaller than the window size
            if i + window_size >= len(token_list):
                break
        return windows

    def preprocess_for_max_length(self):
        """
        Preprocesses the data to handle maximum sequence length constraints.

        This method applies several preprocessing steps to handle maximum sequence length constraints.
        """
        self.dataframe["tokens"] = self.dataframe["tokens"].apply(lambda x: x[1:-1])
        self.dataframe["offset_mapping"] = self.dataframe["offset_mapping"].apply(
            lambda x: x[1:-1]
        )
        self.dataframe["overlapping_tokens"] = self.dataframe["tokens"].apply(
            lambda x: self.create_overlapping_windows(x, 510, 0)
        )

        self.dataframe["overlapping_offset"] = self.dataframe["offset_mapping"].apply(
            lambda x: self.create_overlapping_windows(x, 510, 0)
        )

        self.dataframe["tokens"] = self.dataframe["tokens"].apply(
            lambda x: ["<s>"] + x + ["</s>"]
        )
        self.dataframe["offset_mapping"] = self.dataframe["offset_mapping"].apply(
            lambda x: [(0, 0)] + x + [(0, 0)]
        )

        overlap_tokens = []
        overlap_offset = []
        document_id = []

        for _, row in self.dataframe.iterrows():
            tokens = row["overlapping_tokens"]
            offset = row["overlapping_offset"]
            document_id += [row["id"]] * len(tokens)
            overlap_tokens += tokens
            overlap_offset += offset

        self.dataframe = pd.DataFrame(
            {
                "id": list(range(len(document_id))),
                "document_id": document_id,
                "tokens": overlap_tokens,
                "offset_mapping": overlap_offset,
            }
        )

    def postprocess_for_max_length(self):
        """
        Performs post-processing to handle padding and merge predictions for documents with the same ID.

        This method applies several post-processing steps to handle padding and merge predictions
        for documents with the same ID in the dataframe.
        """
        remove_special_tokens = lambda x: x[1:-1]

        self.dataframe["tokens"] = self.dataframe["tokens"].apply(remove_special_tokens)
        self.dataframe["offset_mapping"] = self.dataframe["offset_mapping"].apply(
            remove_special_tokens
        )
        self.dataframe["predictions"] = self.dataframe["predictions"].apply(
            remove_special_tokens
        )

        def remove_padding_from_predictions(row):
            tokens = row["tokens"]
            predictions = row["predictions"]
            predictions = predictions[0 : len(tokens)]
            return predictions

        def merge_samples_with_same_id(df: pd.DataFrame):
            offsets = []
            predictions = []
            tokens = []
            for _, row in df.iterrows():
                offsets += row["offset_mapping"]
                predictions += row["predictions"]
                tokens += row["tokens"]
            return offsets, predictions, tokens

        self.dataframe["predictions"] = self.dataframe.apply(
            remove_padding_from_predictions, axis=1
        )

        unique = list(self.dataframe["document_id"].unique())
        prediction_df = pd.DataFrame(
            columns=["document_id", "offset_mapping", "predictions", "tokens"]
        )
        for id in unique:
            temp = self.dataframe[self.dataframe["document_id"] == id].copy()
            offset, predictions, tokens = merge_samples_with_same_id(temp)

            prediction_df.loc[len(prediction_df)] = {
                "document_id": id,
                "offset_mapping": offset,
                "predictions": predictions,
                "tokens": tokens,
            }
        self.dataframe = prediction_df

    def create_dataset(
        self, file_names: list[str], documents_list: list[str]
    ) -> DataLoader:
        """
        Creates a dataset from the provided file names and document texts.

        Args:
            file_names (list[str]): List of file names corresponding to the documents.
            documents_list (list[str]): List of document texts.

        Returns:
            DataLoader: DataLoader containing the created dataset.

        This method tokenizes each document text using the tokenizer, encodes the tokens,
        and retrieves token offsets for each token in the original text.
        The tokenized documents, token offsets, and file names are stored in lists.
        Then, these lists are used to create a DataFrame with columns 'id', 'document', 'tokens', and 'offset_mapping'.
        The 'id' column contains file names, 'document' column contains document texts,
        'tokens' column contains tokenized documents, and 'offset_mapping' column contains token offsets.
        Next, a separate DataFrame containing only 'id' and 'document' columns is created as `self.documents_dataframe`.
        After preprocessing the data for the maximum length, a dataset is created using the created DataFrame.
        Finally, a DataLoader is initialized with the created dataset and returned.
        """
        doc_tokens, doc_offsets, ids = [], [], []
        for file_name, doc in zip(file_names, documents_list):
            tokenized = self.tokenizer.encode_plus(doc, return_offsets_mapping=True)
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
            offset = tokenized["offset_mapping"]
            doc_tokens.append(tokens)
            doc_offsets.append(offset)
            ids.append(file_name)

        self.dataframe = pd.DataFrame(
            {
                "id": ids,
                "document": documents_list,
                "tokens": doc_tokens,
                "offset_mapping": doc_offsets,
            }
        )

        self.documents_dataframe = self.dataframe[["id", "document"]]

        self.preprocess_for_max_length()
        data = dataset(
            dataframe=self.dataframe,
            max_len=512,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            training=False,
        )
        test_params = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": 0,
        }
        data_loader = DataLoader(data, **test_params)
        return data_loader

    def predict(self, data_loader: DataLoader) -> List[str]:
        """
        Predicts entity labels for input sentences using the trained model.

        Args:
            data_loader (DataLoader): DataLoader containing batches of input sentences.

        Returns:
            list: List of predicted labels for each sentence in the input data.

        This method iterates through each batch of input sentences in the DataLoader.
        For each batch, it passes the input sentences through the model and retrieves the logits.
        It then converts the logits to predicted labels using the id-to-label mapping.
        The predictions are stored in a list (`all_predictions`), where each element corresponds to a list of predicted labels for a sentence.
        Finally, the predicted labels are added to the dataframe under the 'predictions' column, and the list of predictions is returned.
        """
        id2label = {k: v for v, k in self.label2id.items()}

        self.model.eval()

        nb_eval_steps = 0
        all_predictions = []

        with torch.no_grad():
            progress_bar = tqdm(total=len(data_loader), desc="Batch progress")
            for _, batch in enumerate(data_loader):
                ids = batch["ids"]
                mask = batch["mask"]

                outputs = self.model(input_ids=ids, attention_mask=mask)
                eval_logits = outputs.logits

                nb_eval_steps += 1

                # Process each sentence in the batch
                for sentence_index in range(ids.size(0)):  # Loop over sentences
                    sentence_predictions = []

                    # Process each token in the sentence
                    for token_index in range(
                        ids.size(1)
                    ):  # Loop over tokens in a sentence
                        if mask[sentence_index, token_index].item() == 0:
                            continue  # Skip masked tokens

                        prediction = torch.argmax(
                            eval_logits[sentence_index, token_index]
                        ).item()

                        sentence_predictions.append(id2label[prediction])

                    all_predictions.append(sentence_predictions)
                progress_bar.update(1)

        self.dataframe["predictions"] = all_predictions
        return all_predictions

    def merge_documents_into_dataframe(self):
        """
        Merges the documents dataframe into the existing dataframe.

        This method performs a left merge between the existing dataframe (`self.dataframe`)
        and the documents dataframe (`self.documents_dataframe`) based on the 'document_id' column
        in the existing dataframe and the 'id' column in the documents dataframe.

        The resulting merged dataframe is assigned back to `self.dataframe`.
        """
        self.dataframe = pd.merge(
            left=self.dataframe,
            right=self.documents_dataframe,
            left_on="document_id",
            right_on="id",
        )

    def get_entity_in_text(
        self,
        text: str,
        tokens: list,
        predictions: list,
        offset_mapping: list,
        entity: str,
    ) -> List[str]:
        """
        Extracts entities of a specific type (e.g., PERSON, LOCATION, DATE) from the given text.

        Args:
            - text (str): The input text containing entities.
            - tokens (list): List of tokens generated from the text.
            - predictions (list): Predictions for each token indicating entity tags.
            - offset_mapping (list): Mapping of token offsets to character offsets in the original text.
            - entity (str): The type of entity to extract.

        Returns:
            List[str]: A list of strings containing the extracted entities of the specified type.
        """
        entity_start = f"B-{entity}"
        entity_middle = f"I-{entity}"
        i = 0
        names = []
        while i < len(tokens):
            if predictions[i] == entity_start:
                name = text[offset_mapping[i][0] : offset_mapping[i][1]]
                j = i + 1
                while j < len(tokens) and predictions[j] == entity_middle:
                    if tokens[j].startswith("Ġ"):
                        name += " "
                    name += text[offset_mapping[j][0] : offset_mapping[j][1]]
                    j += 1
                i = j
                names.append(name)
            else:
                i += 1
        return names

    def merge_entities_into_dataframe(self):
        """
        Merges extracted entities (names, locations, dates) into the dataframe.

        This method iterates through each row of the dataframe and extracts entities (names, locations, dates)
        from the corresponding document text based on the predictions and offset mapping.

        The extracted entities are then stored in separate lists (`names`, `locations`, `dates`).

        Finally, these lists are added as new columns to the dataframe with column names 'names', 'locations', 'dates'.
        """
        names = []
        locations = []
        dates = []
        for _, row in self.dataframe.iterrows():
            text = row["document"]
            tokens = row["tokens"]
            predictions = row["predictions"]
            offset = row["offset_mapping"]

            names.append(
                self.get_entity_in_text(text, tokens, predictions, offset, "PERSON")
            )
            locations.append(
                self.get_entity_in_text(text, tokens, predictions, offset, "LOCATION")
            )
            dates.append(
                self.get_entity_in_text(text, tokens, predictions, offset, "DATE")
            )
        self.dataframe["names"] = names
        self.dataframe["locations"] = locations
        self.dataframe["dates"] = dates

    def redact_phones_in_document(self, document: str) -> str:
        """
        Redacts phone numbers found in the given document by replacing them with fake phone numbers.

        Args:
            document (str): The input document in which phone numbers are to be redacted.

        Returns:
            str: The modified document after redacting phone numbers with fake ones.
        """
        phones = find_phone_numbers_in_text(document)
        if not phones:
            return document
        mapping = self.fake.generate_fake_phone_mapping(phones)
        for real_phone, fake_phone in mapping.items():
            document = document.replace(real_phone, fake_phone)
        return document

    def redact_emails_in_document(self, document: str) -> str:
        """
        Redacts email addresses found in the given document by replacing them with fake email addresses.

        Args:
            document (str): The input document in which email addresses are to be redacted.

        Returns:
            str: The modified document after redacting email addresses with fake ones.
        """
        emails = find_emails_in_text(document)
        if not emails:
            return document
        mapping = self.fake.generate_fake_emails_mapping(emails)
        for real_email, fake_email in mapping.items():
            document = document.replace(real_email, fake_email)

        return document

    def redact_urls_in_document(self, document: str) -> str:
        """
        Redacts URLs found in the given document by replacing them with fake URLs.

        Args:
            document (str): The input document in which URLs are to be redacted.

        Returns:
            str: The modified document after redacting URLs with fake ones.
        """
        urls = find_urls_in_text(document)
        if not urls:
            return document
        mapping = self.fake.generate_fake_urls_mapping(urls)
        for real_url, fake_url in mapping.items():
            document = document.replace(real_url, fake_url)
        return document

    def add_fake_mapping_to_dataframe(self):
        name_real_fake_mapping, location_real_fake_mapping, date_real_fake_mapping = (
            [],
            [],
            [],
        )
        for _, row in self.dataframe.iterrows():
            names = row["names"]
            locations = row["locations"]
            dates = row["dates"]

            fake_names = self.fake.generate_fake_names(num_names=len(names))
            fake_locations = self.fake.generate_fake_locations(
                num_locations=len(locations)
            )

            name_mapping = get_real_fake_name_mapping(
                real_names=names, fake_names=fake_names
            )
            name_real_fake_mapping.append(name_mapping)

            location_mapping = get_real_fake_entity_mapping(
                real_entity=locations, fake_entity=fake_locations
            )
            location_real_fake_mapping.append(location_mapping)

            date_mapping = get_real_fake_date_mapping(real_dates=dates)
            date_real_fake_mapping.append(date_mapping)

        self.dataframe["name_real_fake_mapping"] = name_real_fake_mapping
        self.dataframe["location_real_fake_mapping"] = location_real_fake_mapping
        self.dataframe["date_real_fake_mapping"] = date_real_fake_mapping

    def save_redacted_documents(
        self, path: str = paths.OUTPUTS_DIR, file_format: str = "txt"
    ):
        os.makedirs(path, exist_ok=True)
        for _, row in self.dataframe.iterrows():
            redacted_document = row["redacted_document"]
            id = row["id"]
            if file_format == "txt":
                with open(f"{path}/{id}", "w") as file:
                    file.write(redacted_document)
            elif file_format == "pdf":
                save_text_to_pdf(redacted_document, f"{path}/{id}")

    def redact_from_directory(
        self,
        input_path: str = paths.INPUTS_DIR,
        output_path: str = paths.OUTPUTS_DIR,
        file_format: str = "txt",
    ):
        file_names, documents_list = (
            read_text_files(input_path)
            if file_format == "txt"
            else read_pdf_files(input_path)
        )
        data_loader = self.create_dataset(
            file_names=file_names, documents_list=documents_list
        )

        self.predict(data_loader)
        self.postprocess_for_max_length()
        self.merge_documents_into_dataframe()

        self.merge_entities_into_dataframe()
        self.add_fake_mapping_to_dataframe()

        redacted_docs = []
        progress_bar = tqdm(total=len(self.dataframe), desc="Redacting documents")
        for _, row in self.dataframe.iterrows():
            document = row["document"]
            name_mapping = row["name_real_fake_mapping"]
            location_mapping = row["location_real_fake_mapping"]
            date_mapping = row["date_real_fake_mapping"]
            offset = row["offset_mapping"]
            predictions = row["predictions"]
            redacted_document = map_real_to_fake_with_position(
                text=document,
                mapping_dict=name_mapping,
                offset=offset,
                predictions=predictions,
                entity="PERSON",
            )

            redacted_document = replace_values_in_string(
                redacted_document, location_mapping
            )

            redacted_document = map_real_to_fake_with_position(
                text=redacted_document,
                mapping_dict=date_mapping,
                offset=offset,
                predictions=predictions,
                entity="DATE",
            )
            redacted_docs.append(redacted_document)
            progress_bar.update(1)

        redacted_docs = [self.redact_emails_in_document(i) for i in redacted_docs]
        redacted_docs = [self.redact_phones_in_document(i) for i in redacted_docs]
        redacted_docs = [self.redact_urls_in_document(i) for i in redacted_docs]

        self.dataframe["redacted_document"] = redacted_docs
        self.save_redacted_documents(path=output_path, file_format=file_format)


def replace_values_in_string(text: str, mapping: Dict[str, str]) -> str:
    """
    Replaces values in the given text based on the provided mapping.

    Args:
        text (str): The input text in which values are to be replaced.
        mapping (Dict[str, str]): A dictionary where keys represent values to be replaced and values represent the new values.

    Returns:
        str: The modified text after replacing values based on the mapping.
    """
    for original, new in mapping.items():
        text = text.replace(original, new)
    return text


def map_real_to_fake_with_position(
    text: str,
    mapping_dict: dict,
    offset: pd.Series,
    predictions: pd.Series,
    entity: str,
) -> str:
    """
    Redacts the names in a text based on mapping_dict, offsets, and predictions indicating positions
    and the nature of the real names.

    Args:
        - text (str): Text to be redacted.
        - mapping_dict (dict): Dictionary mapping real names to fake names.
        - offset (pd.Series): Pandas series with positions of names in the text.
        - predictions (pd.Series): Pandas series with prediction labels for each token, must be the same length as offset.
        - entity (str): The type of entity to redact.

    Returns (str): Redacted text.
    """
    # Regular expression pattern to match keys in the dictionary regardless of case
    pattern = re.compile(
        "|".join(re.escape(key) for key in mapping_dict.keys()), re.IGNORECASE
    )

    # Convert all keys in mapping_dict to lowercase to ensure case-insensitive matching
    mapping_dict = {k.lower(): v for k, v in mapping_dict.items()}

    # Replacement function
    def replace_match(match):
        start_index = match.start()  # Get the start index of the matched pattern
        # Check if the start_index is within any of the specified positions and predictions are 'B-Person' or 'I-Person'
        if any(
            start <= start_index < end
            and predictions[i] in {f"B-{entity}", f"I-{entity}"}
            for i, (start, end) in enumerate(offset)
        ):
            key = match.group(
                0
            ).lower()  # Get the matched key in lowercase to lookup in the dictionary
            return mapping_dict.get(
                key, match.group(0)
            )  # Use the original match as fallback
        else:
            return match.group(
                0
            )  # Return the original text if not within specified positions or labels don't match

    # Replace matches in the text
    return pattern.sub(replace_match, text)


def find_urls_in_text(text: str) -> List[str]:
    """
    Finds URLs within the given text.

    Args:
        text (str): The input text in which URLs are to be found.

    Returns:
        list[str]: A list containing the URLs found in the text.
    """
    url_pattern = r"https?://(?:www\.)?[a-zA-Z0-9./\-_?&=%]+"
    urls = re.findall(url_pattern, text)
    return urls


def find_emails_in_text(text: str) -> List[str]:
    """
    Finds email addresses within the given text.

    Args:
        text (str): The input text in which email addresses are to be found.

    Returns:
        List[str]: A list containing the email addresses found in the text.
    """

    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    emails = re.findall(email_pattern, text)
    return emails


def find_phone_numbers_in_text(text: str) -> List[str]:
    """
    Finds phone numbers within the given text.

    Args:
        text (str): The input text in which phone numbers are to be found.

    Returns:
        List[str]: A list containing the phone numbers found in the text.

    The function uses regular expressions to match various phone number formats including:
    - Numbers with country code like +1 234 567 8901
    - A sequence of 10 digits
    - '+1' followed by an optional space and then 10 digits
    - US-style phone numbers with parentheses
    - Phone numbers separated by hyphens or spaces
    - International numbers like +44 20 7946 0857
    - Shorter international numbers like +49 30 1234567
    - Numbers like +1 001-740-326-5423
    - European Format like +44 (0)20 7946 0857
    - Asian Format like +91-1234-567890
    - Australian Format like +61 3 9876 5432
    - Latin American Format like +52 (55) 1234 5678
    - Middle Eastern Format like +971 4 123 4567
    """

    pattern = r"""
        (?:\+\d{1,2}\s?\(?\d{1,4}\)?\s?\d{2,4}\s?\d{2,4}\s?\d{0,4})|  # Matches international phone numbers with or without spaces
        (?:\d{10})|                                                       # Matches a sequence of 10 digits
        (?:\+1\s?\d{10})|                                                 # Matches '+1' followed by an optional space and then 10 digits
        (?:\(\d{3}\)\s?\d{3}-\d{4})|                                      # Matches a US-style phone number with parentheses
        (?:\d{3}[-\s]?\d{3}[-\s]?\d{4})|                                  # Matches a phone number separated by hyphens or spaces
        (?:\+\d{2}\s?\(?\d{1,3}\)?\s?\d{2,4}\s?\d{2,4}\s?\d{0,4})|        # Matches international numbers with or without spaces
        (?:\+\d{2}\s?\d{2,4}\s?\d{5,7})|                                  # Matches shorter international numbers with or without spaces
        (?:\+\d\s?\d{3}-\d{3}-\d{3}-\d{4})|                               # Matches numbers like +1 001-740-326-5423
        (?:\+\d{2}\s?\d{1,2}\s\d{4}\s\d{4})|                              # Matches European phone numbers with or without spaces
        (?:\+\d{2}-\d{4}-\d{6,7})|                                        # Matches Asian phone numbers with or without spaces
        (?:\+\d{2}\s?\d{1,2}\s\d{4}\s\d{4})|                              # Matches Australian phone numbers with or without spaces
        (?:\+\d{2}\s\(\d{2,3}\)\s\d{4}\s\d{4})|                           # Matches Latin American phone numbers with or without spaces
        (?:\+\d{3}\s\d{1,2}\s\d{3}\s\d{4})                                # Matches Middle Eastern phone numbers with or without spaces
        """
    # Find all matches as a list of strings using VERBOSE mode for multi-line regex with comments
    matches = re.findall(pattern, text, re.VERBOSE)
    # Directly return matches as findall with non-capturing groups will return a list of matching strings
    return matches
