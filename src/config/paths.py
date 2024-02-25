import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_DIR = os.path.join(ROOT_DIR, "src")

CONFIG_DIR = os.path.join(SRC_DIR, "config")

MODEL_INPUTS_OUTPUTS_DIR = os.path.join(ROOT_DIR, "model_inputs_outputs")

INPUTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "inputs")

OUTPUTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "outputs")

MODEL_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "model")

HYPERPARAMETERS_FILE = os.path.join(CONFIG_DIR, "hyperparameters.json")

LABEL2ID_FILE = os.path.join(CONFIG_DIR, "label2id.json")

ROBERTA_MODEL_DIR = os.path.join(MODEL_DIR, "roberta_model")

TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")
