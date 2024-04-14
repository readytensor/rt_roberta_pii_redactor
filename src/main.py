import os
from config import paths
from utils import read_json_as_dict
from redaction import Redactor
from transformers import RobertaForTokenClassification, RobertaTokenizerFast


def redact(
    model_path: str = paths.ROBERTA_MODEL_DIR,
    tokenizer_path: str = paths.TOKENIZER_DIR,
    input_dir_path: str = paths.INPUTS_DIR,
    output_dir_path: str = paths.OUTPUTS_DIR,
    label2id_path: str = paths.LABEL2ID_FILE,
    batch_size: int = 8,
    random_state: int = 42,
) -> None:
    """
    Redacts sensitive information from text documents in the input directory and saves the redacted documents in the output directory.

    Args:
        model_path (str): Path to the directory containing the pre-trained model.
        tokenizer_path (str): Path to the directory containing the pre-trained tokenizer.
        input_dir_path (str): Path to the input directory containing text documents to be redacted.
        output_dir_path (str): Path to the output directory where redacted documents will be saved.
        label2id_path (str): Path to the label-to-id mapping file.
        batch_size (int): Batch size for redaction.
        random_state (int): Random state for reproducibility.
    """
    model_config_file = os.path.join(model_path, "config.json")
    if os.path.exists(model_config_file):
        model = RobertaForTokenClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

    else:
        model = RobertaForTokenClassification.from_pretrained("moo3030/roberta-for-pii")
        tokenizer = RobertaTokenizerFast.from_pretrained("moo3030/roberta-for-pii")
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(tokenizer_path, exist_ok=True)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)

    label2id = read_json_as_dict(label2id_path)

    redactor = Redactor(
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=batch_size,
        random_state=random_state,
    )

    for format in ["txt", "pdf"]:
        print(f"Redacting {format} documents...")
        redactor.redact_from_directory(
            input_path=input_dir_path,
            output_path=output_dir_path,
            file_format=format,
        )


if __name__ == "__main__":
    hyperparameters = read_json_as_dict(paths.HYPERPARAMETERS_FILE)
    redact(**hyperparameters)
