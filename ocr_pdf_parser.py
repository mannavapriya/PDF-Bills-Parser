!apt-get update && apt-get install -y tesseract-ocr poppler-utils
!pip install gradio pdf2image pytesseract opencv-python-headless pandas openai

import os
import re
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import gradio as gr
import openai
import json
from getpass import getpass

# CONFIG
openai.api_key = getpass("Enter your open ai API key: ")

# Input folder for bills
INPUT_DIR = './files'
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
    print(f"Put bill files (PDFs/images) into {INPUT_DIR}")

# --------------------------
# helpers
# --------------------------
def load_bill_pages(file_path):
    pages = []
    if file_path.lower().endswith('.pdf'):
        imgs = convert_from_path(file_path, dpi=300)
        for img in imgs:
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            pages.append(arr)
    else:
        img = cv2.imread(file_path)
        pages.append(img)
    return pages

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_image(img):
    return pytesseract.image_to_string(img, config='--psm 6')

# --------------------------
# LLM Extraction
# --------------------------
def clean_and_parse_json(text):
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE)
    text = text.replace('""', '"').strip()
    try:
        return json.loads(text)
    except:
        return {"Error": "Failed to parse JSON", "Raw": text}

def extract_bill_with_llm(ocr_text):
    prompt = f"""
You are an AI assistant. Extract the following fields from the OCR text of a bill:
- Provider
- Bill Number
- Bill Date
- Billing Period
- Due Date
- Total Amount
- Currency
- Account Number

Return strictly in JSON format.

OCR Text:
\"\"\"{ocr_text}\"\"\"
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text_response = response.choices[0].message.content
        return clean_and_parse_json(text_response)
    except Exception as e:
        return {"Error": str(e)}

# --------------------------
# Process all bills in the folder
# --------------------------
def process_bills():
    records = []
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
    if not files:
        return "No files found in 'files' folder.", pd.DataFrame()

    for fname in files:
        path = os.path.join(INPUT_DIR, fname)
        try:
            pages = load_bill_pages(path)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        full_text = ""
        for page in pages:
            try:
                pre = preprocess_image(page)
                txt = ocr_image(pre)
                full_text += "\n" + txt
            except Exception as e:
                print(f"OCR failed for page in {fname}: {e}")
                continue

        # Skip files with no text
        if not full_text.strip():
            print(f"No text found in {fname}, skipping.")
            continue

        try:
            fields = extract_bill_with_llm(full_text)
        except Exception as e:
            print(f"LLM extraction failed for {fname}: {e}")
            fields = {"Error": str(e)}

        fields['Filename'] = fname
        records.append(fields)

    df = pd.DataFrame(records)
    return f"Processed {len(records)} bills.", df

# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks() as demo:
    gr.HTML("""
    <style>
        /* Set background for Gradio container */
        .gradio-container {
            background-color: #f0f8ff !important;  /* Light blue background */
        }
    </style>
    """)
    gr.Markdown("## PDF Bill Parser")
    with gr.Row():
        output_text = gr.Textbox(label="Status")
    with gr.Row():
        output_table = gr.Dataframe(
            headers=["Filename","Provider","Bill Number","Bill Date","Billing Period","Due Date","Total Amount","Currency","Account Number"],
            datatype="str", interactive=False, wrap=True
        )
    process_btn = gr.Button("Process Bills from 'files' folder")
    process_btn.click(fn=process_bills, inputs=[], outputs=[output_text, output_table])

demo.launch()