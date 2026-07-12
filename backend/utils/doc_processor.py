#!/home/preritubuntu/miniconda3/envs/langchain_env/bin/python
import os
import sys
import uuid
import json
import base64
import requests
from pathlib import Path

# Safe import checking with user instructions
try:
    import docling
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DocItemLabel, PictureItem, TableItem, TextItem
except ImportError as e:
    print(f"\n[Error] ModuleNotFoundError: {e}")
    print("Please ensure you are using the correct Conda environment ('langchain_env').")
    print("Activate it with:")
    print("  conda activate langchain_env")
    print("Or run the script using the direct Python interpreter path:")
    print("  /home/preritubuntu/miniconda3/envs/langchain_env/bin/python backend/utils/doc_processor.py\n")
    sys.exit(1)


import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
def clean_unicode_text(text: str) -> str:
    if not text:
        return text
    unicode_map = {
        0x201c: '"',  # “ (Left Double Quote)
        0x201d: '"',  # ” (Right Double Quote)
        0x2018: "'",  # ‘ (Left Single Quote)
        0x2019: "'",  # ’ (Right Single Quote / Smart Apostrophe)
        0x2013: "-",  # – (En Dash)
        0x2014: "-",  # — (Em Dash)
        0x2212: "-",  # − (Minus Sign)
        0xa0:   " ",  # Non-breaking space
    }
    return text.translate(unicode_map)

class LangChainVLMClient:
    def __init__(self, api_base="http://127.0.0.1:1234/v1", model="google/gemma-3-4b"):
        # We use ChatOpenAI because LM Studio is OpenAI-compatible
        self.llm = ChatOpenAI(
            base_url=api_base,
            openai_api_key="lm-studio",  # A placeholder key is required by LangChain
            model=model,
            temperature=0.2,
            max_tokens=512
        )
    def describe_image(self, image_path, prompt="Please analyze this image from a scientific/technical document and provide a detailed description of its contents, including any text, trends, data points, or patterns.Do not add your own interpretation or assumptions, Please use utf-8 characters only"):
        if not os.path.exists(image_path):
            return f"[VLM Error: Image path '{image_path}' not found]"
        try:
            # Encode image to base64
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            return f"[VLM Error: Failed to encode image: {e}]"
        # Construct a multimodal human message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        # Pass base64 data inline using the data URI scheme
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        )
        try:
            # Invoke the VLM model
            response = self.llm.invoke([message])
            text = response.content
            # Fix Mojibake (Latin-1 vs UTF-8 decoding mismatch from local HTTP responses)
            try:
                text = text.encode('latin-1').decode('utf-8')
            except Exception:
                pass
            return clean_unicode_text(text.strip())
        except Exception as e:
            return f"[VLM Error: Request failed: {e}]"

def bbox_to_list(prov):
    """Converts a Docling provenance bbox to a [l, t, r, b] float list."""
    if not prov or not getattr(prov[0], 'bbox', None):
        return []
    bbox = prov[0].bbox
    return [bbox.l, bbox.t, bbox.r, bbox.b]


def resolve_reference_text(doc, ref):
    """Resolves a Docling RefItem cref pointer (e.g. '#/texts/25') to its text content."""
    if not ref or not getattr(ref, 'cref', None):
        return ""
    try:
        parts = ref.cref.split("/")
        if len(parts) >= 3 and parts[1] == "texts":
            idx = int(parts[2])
            if idx < len(doc.texts):
                return doc.texts[idx].text
    except Exception as e:
        print(f"Error resolving reference {ref.cref}: {e}")
    return ""


def process_document(file_path, output_json_path, vlm_url=None, run_vlm_on_tables=False):
    """
    Parses a PDF using Docling, extracts tables as Markdown, crops figures,
    groups items under parent headings, runs the VLM if active, and saves to JSON.
    """
    file_path = Path(file_path)
    output_json_path = Path(output_json_path)
    doc_id = file_path.stem
    filename = file_path.name

    # Create directories for crops
    images_dir = output_json_path.parent / "extracted_images"
    tables_dir = output_json_path.parent / "extracted_tables"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Initialize local VLM client if URL is provided
    vlm_client = LangChainVLMClient(api_base=vlm_url) if vlm_url else None

    # Setup Docling converter options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = run_vlm_on_tables

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print(f"Converting document: {filename} with Docling...")
    conv_res = converter.convert(str(file_path))
    doc = conv_res.document

    # Containers for parsing
    sections = []
    
    # Initialize front matter section for text/figures before the first heading
    current_section = {
        "doc_id": doc_id,
        "filename": filename,
        "page_start": 1,
        "page_end": 1,
        "heading": "Front Matter / Document Header",
        "text": "",
        "figures": [],
        "tables": [],
        "footnotes": []
    }

    picture_counter = 0
    table_counter = 0

    print("Grouping document items sequentially...")
    # iterate_items yields elements in reading order
    for element, _ in doc.iterate_items():
        # Get page number for the element
        item_page = 1
        if element.prov:
            item_page = element.prov[0].page_no

        # Check label type
        label = element.label

        if label == DocItemLabel.SECTION_HEADER:
            # If current section has content, commit it
            if current_section["text"].strip() or current_section["tables"] or current_section["figures"]:
                sections.append(current_section)
            
            # Start new section
            current_section = {
                "doc_id": doc_id,
                "filename": filename,
                "page_start": item_page,
                "page_end": item_page,
                "heading": element.text.strip(),
                "text": "",
                "figures": [],
                "tables": [],
                "footnotes": []
            }

        elif label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM):
            text_val = element.text.strip()
            if text_val:
                current_section["text"] += text_val + "\n"
                current_section["page_end"] = max(current_section["page_end"], item_page)

        elif label == DocItemLabel.FORMULA:
            # Append formula to the main text block
            formula_val = element.text.strip()
            if formula_val:
                current_section["text"] += f"\n[Formula: {formula_val}]\n"
                current_section["page_end"] = max(current_section["page_end"], item_page)

        elif label == DocItemLabel.TABLE:
            table_counter += 1
            table_md = ""
            try:
                table_md = element.export_to_markdown(doc)
            except Exception as e:
                print(f"Error exporting table {table_counter} to markdown: {e}")

            # Get caption text
            caption_text = ""
            if getattr(element, 'captions', None):
                caption_text = " ".join([resolve_reference_text(doc, ref) for ref in element.captions]).strip()

            # Handle image cropping for table if requested
            table_img_path = ""
            summary = ""
            if run_vlm_on_tables:
                try:
                    table_img_path = tables_dir / f"table_{table_counter}.png"
                    element.get_image(doc).save(table_img_path)
                    
                    if vlm_client:
                        print(f"Generating local VLM description for Table {table_counter}...")
                        summary = vlm_client.describe_image(
                            str(table_img_path),
                            prompt="Summarize this table's structure and the key info it represents."
                        )
                except Exception as e:
                    print(f"Error cropping table {table_counter}: {e}")

            table_entry = {
                "page": item_page,
                "bbox": bbox_to_list(element.prov),
                "markdown": table_md,
                "caption": caption_text,
                "summary": summary,
                "image_path": str(table_img_path) if table_img_path else ""
            }
            current_section["tables"].append(table_entry)
            current_section["page_end"] = max(current_section["page_end"], item_page)

        elif label == DocItemLabel.PICTURE:
            picture_counter += 1
            img_path = images_dir / f"picture_{picture_counter}.png"
            
            # Crop image
            try:
                element.get_image(doc).save(img_path)
            except Exception as e:
                print(f"Error saving picture crop {picture_counter}: {e}")
                img_path = ""

            # Get caption
            caption_text = ""
            if getattr(element, 'captions', None):
                caption_text = " ".join([resolve_reference_text(doc, ref) for ref in element.captions]).strip()

            # Ask local VLM to generate image description
            description = ""
            if img_path and vlm_client:
                print(f"Generating local VLM description for Figure {picture_counter}...")
                description = vlm_client.describe_image(str(img_path))

            figure_entry = {
                "page": item_page,
                "bbox": bbox_to_list(element.prov),
                "image_path": str(img_path) if img_path else "",
                "caption": caption_text,
                "description": description
            }
            current_section["figures"].append(figure_entry)
            current_section["page_end"] = max(current_section["page_end"], item_page)

        elif label == DocItemLabel.FOOTNOTE:
            current_section["footnotes"].append({
                "page": item_page,
                "text": element.text.strip()
            })

    # Commit the last section if not empty
    if current_section["text"].strip() or current_section["tables"] or current_section["figures"]:
        sections.append(current_section)

    # Save to output file
    print(f"Saving structured JSON to: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {filename}. Extracted:")
    print(f" - {len(sections)} sections/headings")
    print(f" - {table_counter} tables")
    print(f" - {picture_counter} figures/pictures")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Docling Structured Parser with Local VLM Integration.")
    parser.add_argument("--input", required=True, help="Path to input PDF file.")
    parser.add_argument("--output", required=True, help="Path to output JSON file.")
    parser.add_argument("--vlm-url", default=None, help="Base API URL for local LM Studio VLM (e.g. http://localhost:1234/v1).")
    parser.add_argument("--run-vlm-on-tables", action="store_true", help="Generate table image crops and ask local VLM to summarize them.")
    
    args = parser.parse_args()
    process_document(args.input, args.output, vlm_url=args.vlm_url, run_vlm_on_tables=args.run_vlm_on_tables)
