# Personally identifiable information (PII) Redactor with RoBERTa

PII Redactor model using RoBERTa. The goal of this model is to remove personally identifiable information from text documents.

PII includes:

- Names
- Dates
- Emails
- Phone numbers
- Addresses
- URLs

---
Here are the highlights of this implementation: <br/>

- **RoBERTa** model trained using **transformers** library. The model is trained to identify names, addresses and dates.
- **Regex** logic to capture phone numbers, emails and URLs.
- **FakeGenerator** module to generate fake information.
- **Redactor** module to replace PII with fake information.

## Project Structure

The following is the directory structure of the project:

- **`model_inputs_outputs/`**: This directory contains files that are either inputs to, or outputs from, the model. This directory is further divided into:
  - **`/inputs/`**: This directory contains the input .txt files to be redacted. 
  - **`/model`**: This directory is used to store the model used for redaction along with the tokenizer used for tokenizing the text files.
  - **`/outputs/`**: The outputs directory will contain the output files after running the model on the input files.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories:
  - **`config/`**: for configuration files for data preprocessing, model hyperparameters, paths, etc.
  - **`redact.py`**: This script is used to run the model on the text files inside **inputs** directory.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains the license for the project.
- **`requirements.txt`** for the main code in the `src` directory.
- **`label2id.json`** This file contains label encoding for the token classes that were used to train the model.
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.

## Usage

- Place the data you want to redact in a .txt format
- Move the .txt files inside the **/model_inputs_outputs/inputs** directory
- Run the **redact.py** script
- Get the result files from **/model_inputs_outputs/outputs** directory


## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## LICENSE

This project is provided under the Apache-2.0 License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. Visit https://www.readytensor.ai/
