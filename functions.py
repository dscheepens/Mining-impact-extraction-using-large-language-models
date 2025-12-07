import os
import fitz  # PyMuPDF
from PIL import Image
from qwen_vl_utils import smart_resize
import ast
import json
import pandas as pd 
import numpy as np 

    
def pdf_to_images(study_num, output_dir="extraction_papers/images/", zoom=5):
    """
    Convert each page of a PDF into a PNG image, resize for Qwen2.5-VL, and save locally.

    Parameters:
        study_num (str): Base name (or number) of the PDF in 'extraction_papers/'.
        output_dir (str): Directory where processed images will be saved.
        zoom (float): Scaling factor for higher-resolution rendering.

    Returns:
        list of str: Paths to resized PNG images ready for Qwen
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_path = os.path.join(f"extraction_papers/{study_num}.pdf")
    
    pdf = fitz.open(pdf_path)
    saved_images = []

    for page_number, page in enumerate(pdf, start=1):
        # Render page to image
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Temporary save before resizing
        tmp_path = os.path.join(output_dir, f"{study_num}_{page_number}_raw.png")
        pix.save(tmp_path)

        # Load with PIL
        img = Image.open(tmp_path).convert("RGB")
        width, height = img.size

        # Compute resized dimensions using Qwen's smart_resize
        factor = 32
        min_pixels = 4 * factor * factor
        max_pixels = 16384 * factor * factor
        new_h, new_w = smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)

        # Resize the image
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Save final version (can overwrite or create a new file)
        output_file = os.path.join(output_dir, f"{study_num}_{page_number}.png")
        img_resized.save(output_file)

        saved_images.append(output_file)

        # Optionally remove raw image
        os.remove(tmp_path)

    pdf.close()
    return saved_images




def flatten_cell(x):
    try:
        # Try to parse stringified lists like "['a', 'b']"
        val = ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") and x.endswith("]") else x
    except (ValueError, SyntaxError):
        val = x
    # Join if list, otherwise just return as string
    if isinstance(val, list):
        return ", ".join(map(str, val))
    return str(val)


def get_items(json):
    try: 
        if isinstance(json, dict) and json.get('items',False): 
            json = json['items']
        if isinstance(json, dict) and json.get('data',False): 
            json = json['data']
    except: 
        pass
    return(json)


import re
import json

def sanitize_invalid_unicode_escapes(s):
    """
    Replace invalid escapes with escaped versions
    so json.loads does not fail.
    """
    def fix(match):
        text = match.group(0)
        return ""   # prepend backslash to escape it
    # Match \u not followed by exactly 4 hex digits
    return re.sub(r'\\u(?![0-9a-fA-F]{4})', fix, s)

def wrap_multiple_objects(cleaned):
    objs = []

    # Find all JSON objects with a top-level pattern
    matches = re.findall(r'\{.*?\}', cleaned, re.DOTALL)
    if len(matches) > 1:
        # Return a JSON array
        return "[" + ",".join(matches) + "]"
    return cleaned
    
def string_to_json(json_string):
    """
    Handles cases with markdown code fences (```json ... ```),
    and sanitizes malformed unicode escape sequences before parsing.
    """
    
    if isinstance(json_string, str):
        cleaned = json_string.strip()

        # Remove markdown code fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # SANITIZE invalid \uXXXX sequences BEFORE json.loads
        cleaned = sanitize_invalid_unicode_escapes(cleaned)

        # if text contains multiple JSONs without list, wrap these into one JSON 
        cleaned = wrap_multiple_objects(cleaned)
        
        # Try parsing as JSON
        try:
            json_string = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("Error in string_to_json. Returning None.")
            print("JSONDecodeError:", e)
            print("Cleaned text:", cleaned)
            return None

        # If results wrapped in "items", unwrap it
        json_string = get_items(json_string)

    return json_string


def extract_tables(ollama, model_name, seed, system_prompt, extracted_text, image_paths):
    # Start conversation history, initiate system prompt 
    messages = [{
            "role": "system",
            "content": """
            You are a precise and careful assistant specialized in extracting tabular data from scientific papers.

            Your task:
            - Extract every table that appears in the provided document or page images.
            - Preserve the exact structure of each table, including:
              - All column and row headers
              - Units of measurement
              - Footnotes, superscripts, and symbols (e.g., *, †)
              - Alignment of values (left/right/center)
              - Merged cells or multi-level headers (represent these clearly in Markdown)
            
            Formatting rules:
            1. Output each table in a **Markdown table** using plain text.  
               - If a table is too wide or complex, represent it in **JSON** with keys as column headers and rows as lists of objects.
            2. **Include the table caption or title**, if present, immediately before the table.
            3. If the page contains **no table**, respond with: No tables detected.
            4. **Do not summarize, interpret, or paraphrase** any data — reproduce it exactly as printed.
            5. Correct only obvious OCR errors (e.g., “O”→“0” in numeric cells).
            6. Do not include figures, figure captions, or regular text paragraphs.
            7. Maintain the logical reading order if the table is split across columns or pages.
            
            Your output should be strictly limited to the extracted tables — no commentary, explanation, or extra formatting.
            
            Goal:
            Return all tables from the input as accurately and faithfully as possible in a format suitable for automated data parsing or verification.
            """
    }]
    
    # First turn (with images)
    messages.append({
            "role": "user",
            "content": "Carefully read the following mining-impact study:\n'n" + extracted_text + "\n\nNow carefully extract all tables from the paper.",
            "images": image_paths
    })
    
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True,
        options={"seed":seed,
                 "num_ctx":4096*10,
                 "num_predict":4096*5,
                 #"timeout":3600,
                 "temperature":0
                },
    )

    try:
        # Stream the chunks as they arrive
        response = ""
        for chunk in stream:
            # Each chunk is a dict; content lives here:
            if "message" in chunk and "content" in chunk["message"]:
                text = chunk["message"]["content"]
                #print(text, end="", flush=True)   
                response += text    
    except Exception as e:
        print(f"Error calling Ollama API: {e}.")
        response = None
        
    # Append assistant response to history
    messages.append({
        "role": "assistant",
        "content": response,
    })
    
    return response, messages

def extract_information_formatted(ollama, model_name, seed, system_prompt, user_prompt, prompt_format, image_paths):
    # Start conversation history, initiate system prompt 
    messages = [{
            "role": "system",
            "content": system_prompt
    }]
    
    # First turn (with images)
    messages.append({
            "role": "user",
            "content": user_prompt,
            "images": image_paths
    })
    
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True,
        options={"seed":seed,
                 "num_ctx":4096*10,
                 "num_predict":4096*1,
                 #"timeout":3600,
                 "temperature":0
                },
        format=prompt_format
    )

    try:
        # Stream the chunks as they arrive
        response = ""
        for chunk in stream:
            # Each chunk is a dict; content lives here:
            if "message" in chunk and "content" in chunk["message"]:
                text = chunk["message"]["content"]
                #print(text, end="", flush=True)   
                response += text    
        response_json = json.loads(response)
    except Exception as e:
        print(f"Error calling Ollama API: {e}.")
        response_json = None
        
    # Append assistant response to history
    messages.append({
        "role": "assistant",
        "content": response,
    })
    
    return response_json, messages
    
def extract_information(ollama, model_name, seed, system_prompt, images=False, image_paths="", prompt=""):
    # Start conversation history, initiate system prompt 
    messages = [{
            "role": "system",
            "content": system_prompt
    }]

    if images: 
        messages.append({
                "role": "user",
                "content": prompt,
                "images": image_paths
        })
    else: 
        messages.append({
                "role": "user",
                "content": prompt,
        })
    
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True,
        options={"seed":seed,
                 "think":True,
                 "num_ctx":4096*10,
                 "num_predict":4096*5,
                 #"timeout":3600,
                 "temperature":0
                },
    )

    try:
        # Stream the chunks as they arrive
        response = ""
        for chunk in stream:
            # Each chunk is a dict; content lives here:
            if "message" in chunk and "content" in chunk["message"]:
                text = chunk["message"]["content"]
                #print(text, end="", flush=True)   
                response += text    
    except Exception as e:
        print(f"Error calling Ollama API: {e}.")
        response = None

    # Append assistant response to history
    messages.append({
        "role": "assistant",
        "content": response,
    })

    return response, messages


def get_json_table(ollama, model_name, seed, text_response, prompt_format):

    messages = [{"role": "system", "content": "You are an expert JSON generator and are to return JSON responses only."},
                {"role": "user", "content": f"Carefully read this information:\n\n{text_response}\n\nReturn this information according to the following JSON schema and output JSON only: {prompt_format}"}
               ]
    
    # messages.append({
    #         "role": "user",
    #         "content": prompt
    # })
    
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        #format=prompt_format,
        stream=True,
        options={"seed":seed,
                 "think":True,
                 "num_ctx":4096*10,
                 "num_predict":4096*5,
                 #"timeout":3600,
                 "temperature":0
                },
    )

    try: 
        # Stream the chunks as they arrive
        response = ""
        for chunk in stream:
            # Each chunk is a dict; content lives here:
            if "message" in chunk and "content" in chunk["message"]:
                text = chunk["message"]["content"]
                #print(text, end="", flush=True)   
                response += text  

        json_table = string_to_json(response) 
        #json_table = json.loads(response)
    except Exception as e:
        print(f"Error calling Ollama API: {e}.")
        return None 
    
    return json_table


def get_df_metrics(json_metrics, response_metrics): 
    cols = ["Metric", "Metric category", "Impact direction", "Significance", "P-value", "Effect size"]
    
    try: 
        df = pd.DataFrame([[""] * len(cols)],columns=cols) # empty dataframe
        for metric in json_metrics: 
            df_row = pd.DataFrame([[""] * len(cols)],columns=cols) # empty dataframe
            df_row['Metric'] = metric.get('metric', "")
            df_row['Metric category'] = metric.get('metric_category', "")
            df_row['Impact direction'] = metric.get('impact_direction', "")
            df_row['Significance'] = metric.get('significance', "")
            df_row['P-value'] = metric.get('p_value', "")
            df_row['Effect size'] = metric.get('effect_size', "")
            df = pd.concat([df, df_row], ignore_index=True)
    except Exception as e:
        print(f"Error in get_metric_row(): {e}.")
        print('metric:',metric)
    df = df.map(flatten_cell)
    return df[1:]

def add_remaining_rows(df, json_response): 

    df["Scale"] = json_response.get("scale","")
    df["Country/Continent"] = json_response.get("country_or_continent","")
    df["Specific location"] = json_response.get("specific_location","")
    df["Mined commodities"] = json_response.get("mined_commodities","")
    df["Mining stage"] = str(json_response.get("mining_stage",""))
    df["Type of mining activity"] = str(json_response.get("type_of_mining_activity",[""]))
    df["Experimental design"] = str(json_response.get("experimental_design",[""]))
    df["Taxonomic groups"] = json_response.get("taxonomic_groups","")
    df["Biomes of assessment"] = str(json_response.get("biomes_of_assessment",[""]))
    df["Method of assessment"] = str(json_response.get("method_of_assessment",[""]))
    df["Specific methodology"] = json_response.get("specific_methodology","")
    df["Temporal scale"] = json_response.get("temporal_scale","")
    df["Type of impact or pressure"] = json_response.get("type_of_impact_or_pressure","")
    df["Impact pathway"] = str(json_response.get("impact_pathway",[""]))
    df["Description of impact or pressure"] = json_response.get("description_of_impact_or_pressure","")
    df["Spatial scale"] = json_response.get("spatial_scale","")

    df = df.map(flatten_cell)
    return df



def extract_and_return_as_json(ollama, model, seed, system_prompt, user_prompt, prompt_format, images=False, image_paths=""):

    # 1. generate text response 
    text_response, _ = extract_information(ollama, model, seed, system_prompt, images, image_paths, prompt=user_prompt)
    if text_response is None:
        print("1. Generating text response is None. Trying again...")
        text_response, _ = extract_information(ollama, model, seed, system_prompt, images, image_paths, prompt=user_prompt)
        if text_response is None: # probable API error or failure to generate 
            print("Failed.") 
            
    # 2. return response as json: 
    json_response = get_json_table(ollama, model, seed, text_response, prompt_format)
    if json_response is None:
        print("2. Generating json is None. Trying again...")
        json_response = get_json_table(ollama, model, seed, text_response, prompt_format)
        if json_response is None: # probable API error or failure to generate 
            print("Failed.") 

    return(text_response, json_response)

    

            

def get_full_response_json(ollama, 
                           model_vl, 
                           model_lang, 
                           seed, 
                           study_id, 
                           system_prompt, 
                           text_prompt, 
                           text_prompt_diversity_metrics, 
                           text_prompt_diversity_metrics_2, 
                           text_tables, 
                           prompt_format, 
                           prompt_format_metrics, 
                           prompt_format_single_metric, 
                           image_paths, 
                           extracted_text, 
                           markdown_text,
                           prompt_prefix=""): 
    
    # 1. get text response for all columns except the metrics (supply extracted raw text): 
    user_prompt_1 = "Carefully read this mining-impact study:\n\n" + markdown_text + "\n\nRaw text: " + extracted_text + "\n\n" + prompt_prefix + text_prompt
    
    response, json_response = extract_and_return_as_json(ollama, model_lang, seed, system_prompt, user_prompt_1, prompt_format, images=False)

    # 2. get text response for the metrics (supply images, text and extracted tables): 
    user_prompt_2 = "Carefully read this mining-impact study:\n\n" + markdown_text + "\n\nRaw text: " + extracted_text + "\n\nTables from the text: " + text_tables  + "\n\n" + prompt_prefix + text_prompt_diversity_metrics
    
    response_metrics, json_metrics = extract_and_return_as_json(ollama, model_lang, seed, system_prompt, user_prompt_2, prompt_format_metrics, images=False)
    
    if json_metrics is not None: 
        
        cols = ["Metric", "Metric category", "Impact direction", "Significance", "P value", "Effect size", "Response metric"]
        df = pd.DataFrame([[""] * len(cols)],columns=cols) # empty dataframe
        for metric in json_metrics: 
            df_row = pd.DataFrame([[""] * len(cols)],columns=cols) # empty dataframe
            df_row['Metric'] = metric.get('metric', "")
            df_row['Metric category'] = metric.get('metric_category', "")
            df = pd.concat([df, df_row], ignore_index=True)
        df = df[1:] # first row is empty 
        
        for m in df['Metric']: 
            print(f"Extracting information for metric: {m}")
            
            # setup prompt for metric
            user_prompt_3 = "Carefully read this mining-impact study:\n\n" + markdown_text + "\n\nRaw text: " + extracted_text + "\n\nTables from the text: " + text_tables + "\n\n" + prompt_prefix + f"For the metric {m} specifically, determine:\n" + text_prompt_diversity_metrics_2
            
            # extract json 
            response_metric, json_metric = extract_and_return_as_json(ollama, model_lang, seed, system_prompt, user_prompt_3, prompt_format_single_metric, images=False)
            
            # add to df 
            df.loc[df['Metric'] == m, 'Response metric']  = response_metric # save text response
            if json_metric is not None: 
                if isinstance(json_metric, list): # the metric was split into a list of individual metrics  
                    for metric_i in json_metric:
                        df_i = pd.DataFrame([[""] * len(cols)],columns=cols)
                        df_i['Metric']           = metric_i.get("metric", "")
                        df_i['Metric category']  = df.loc[df['Metric'] == m, 'Metric category'].iloc[0]
                        df_i['Impact direction'] = metric_i.get("impact_direction", "")
                        df_i['Significance']     = metric_i.get("significance", "")
                        df_i['P value']          = metric_i.get("p_value", "")
                        df_i['Effect size']      = metric_i.get("effect_size", "")
                        df_i['Response metric']  = response_metric # save text response
                        df = pd.concat([df, df_i], ignore_index=True)
                    df = df[df["Metric"] != m] # remove the earlier metric (which has been replaced) 
                else: 
                    df.loc[df['Metric'] == m, 'Impact direction'] = json_metric.get("impact_direction", "")
                    df.loc[df['Metric'] == m, 'Significance']     = json_metric.get("significance", "")
                    df.loc[df['Metric'] == m, 'P value']          = json_metric.get("p_value", "")
                    df.loc[df['Metric'] == m, 'Effect size']      = json_metric.get("effect_size", "")
                    
    # add previous columns to df 
    df = add_remaining_rows(df, json_response)

    df["Response study"] = response
    df["Extracted tables"] = text_tables
    df["Study ID"] = study_id
    
    return df

    