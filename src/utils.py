import os
import json
import torch
import random
import pdfplumber
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet


def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
    else:
        raise ValueError(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def read_json_as_dict(input_path: str) -> Dict:
    """
    Reads a JSON file and returns its content as a dictionary.
    If input_path is a directory, the first JSON file in the directory is read.
    If input_path is a file, the file is read.

    Args:
        input_path (str): The path to the JSON file or directory containing a JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.

    Raises:
        ValueError: If the input_path is neither a file nor a directory,
                    or if input_path is a directory without any JSON files.
    """
    if os.path.isdir(input_path):
        # Get all the JSON files in the directory
        json_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ]

        # If there are no JSON files, raise a ValueError
        if not json_files:
            raise ValueError("No JSON files found in the directory")

        # Else, get the path of the first JSON file
        json_file_path = json_files[0]

    elif os.path.isfile(input_path):
        json_file_path = input_path
    else:
        raise ValueError("Input path is neither a file nor a directory")

    # Read the JSON file and return it as a dictionary
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data_as_dict = json.load(file)

    return json_data_as_dict


def read_csv_in_directory(file_dir_path: str) -> pd.DataFrame:
    """
    Reads a CSV file in the given directory path as a pandas dataframe and returns
    the dataframe.

    Args:
    - file_dir_path (str): The path to the directory containing the CSV file.

    Returns:
    - pd.DataFrame: The pandas dataframe containing the data from the CSV file.

    Raises:
    - FileNotFoundError: If the directory does not exist.
    - ValueError: If no CSV file is found in the directory or if multiple CSV files are
        found in the directory.
    """
    if not os.path.exists(file_dir_path):
        raise FileNotFoundError(f"Directory does not exist: {file_dir_path}")

    csv_files = [file for file in os.listdir(file_dir_path) if file.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No CSV file found in directory {file_dir_path}")

    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory {file_dir_path}.")

    csv_file_path = os.path.join(file_dir_path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    return df


def read_pdf_files(dir_path: str) -> List[str]:
    file_names = [f for f in os.listdir(dir_path) if f.endswith(".pdf")]
    file_paths = [os.path.join(dir_path, f) for f in file_names]
    result = []
    for pdf_path in file_paths:
        # Create a string to hold the extracted text
        full_text = ""

        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            # Iterate over each page in the PDF
            for page in pdf.pages:
                # Extract text from the current page
                page_text = page.extract_text()
                # Add the text to the full_text string with a newline
                if page_text:  # If text extraction was successful
                    full_text += page_text + "\n"
        result.append(full_text)

    return file_names, result


def save_text_to_pdf1(text_string, pdf_path):
    # Create a PDF document using SimpleDocTemplate which is more suited for flowables like Paragraph
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []  # This will hold the elements of the document

    # Get the default style and set font size
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    style.fontName = "Helvetica"
    style.fontSize = 12
    style.leading = 15  # Line spacing

    # Split the text into paragraphs by newlines
    paragraphs = text_string.split("\n")

    for paragraph in paragraphs:
        # Create a Paragraph object for each paragraph in the text
        para = Paragraph(paragraph, style)
        story.append(para)

    # Build the PDF using the story which contains all the Paragraph objects
    doc.build(story)


def save_text_to_pdf(text_string, pdf_path):
    # Create a canvas for the PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter  # Get the width and height of the page
    text_object = c.beginText(40, height - 40)  # Start the text object
    text_object.setFont("Helvetica", 12)  # Set the font and size

    # Split the text string into lines
    lines = text_string.split("\n")

    for line in lines:
        # Add each line to the text object
        text_object.textLine(line.strip())

        # Check if we have reached the bottom of the page
        if text_object.getY() < 40:
            # Start a new page
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(40, height - 40)
            text_object.setFont("Helvetica", 12)

    # Draw the text on the canvas
    c.drawText(text_object)
    c.save()


def read_text_files(path: str) -> Tuple[List[str], List[str]]:
    """
    Reads text files from the specified directory.

    Args:
        path (str): The directory path containing the text files.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - file names and corresponding document texts.

    This method lists all files in the specified directory with the '.txt' extension.
    It then reads the content of each text file and appends it to a list along with the corresponding file names.
    Finally, it returns a tuple containing two lists - file names and corresponding document texts.
    """
    data = []
    file_names = [i for i in os.listdir(path) if i.endswith(".txt")]
    file_paths = [os.path.join(path, i) for i in file_names]

    for f in file_paths:
        with open(f, "r") as file:
            text = file.read()
            data.append(text)

    return file_names, data
