import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_docling_json(json_path, chunk_size=1000, chunk_overlap=150):
    """
    Loads the structured Docling JSON output and converts it into a list of 
    LangChain Document objects. It splits prose text safely while keeping 
    tables and figure descriptions intact as standalone documents.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)
        
    documents = []
    
    # Text splitter for standard prose paragraphs only
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    global_text_counter = 0
    global_table_counter = 0
    global_figure_counter = 0
    
    for section_idx, section in enumerate(sections):
        doc_id = section.get("doc_id", "")
        filename = section.get("filename", "")
        heading = section.get("heading", "")
        page_start = section.get("page_start", 1)
        page_end = section.get("page_end", 1)
        
        # 1. Split and add prose text
        section_text = section.get("text", "").strip()
        if section_text:
            text_chunks = text_splitter.split_text(section_text)
            for idx, chunk in enumerate(text_chunks):
                global_text_counter += 1
                metadata = {
                    "type": "text",
                    "doc_id": doc_id,
                    "filename": filename,
                    "heading": heading,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_id": f"{doc_id}_txt_{global_text_counter}"
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
                
        # 2. Add Tables as standalone Documents (keeps structure intact)
        for table in section.get("tables", []):
            global_table_counter += 1
            caption = table.get("caption", "").strip()
            markdown = table.get("markdown", "").strip()
            summary = table.get("summary", "").strip()
            page = table.get("page", page_start)
            image_path = table.get("image_path", "")
            
            # Formulate structural table presentation
            table_content = f"Table Caption: {caption}\n\n{markdown}"
            if summary:
                table_content += f"\n\nTable Summary: {summary}"
                
            metadata = {
                "type": "table",
                "doc_id": doc_id,
                "filename": filename,
                "heading": heading,
                "page": page,
                "image_path": image_path,
                "chunk_id": f"{doc_id}_tbl_{global_table_counter}_pg_{page}"
            }
            documents.append(Document(page_content=table_content, metadata=metadata))
            
        # 3. Add Figures as standalone Documents (keeps VLM description intact)
        for figure in section.get("figures", []):
            global_figure_counter += 1
            caption = figure.get("caption", "").strip()
            description = figure.get("description", "").strip()
            page = figure.get("page", page_start)
            image_path = figure.get("image_path", "")
            
            figure_content = f"Figure Caption: {caption}"
            if description:
                figure_content += f"\n\nFigure Description: {description}"
                
            metadata = {
                "type": "figure",
                "doc_id": doc_id,
                "filename": filename,
                "heading": heading,
                "page": page,
                "image_path": image_path,
                "chunk_id": f"{doc_id}_fig_{global_figure_counter}_pg_{page}"
            }
            documents.append(Document(page_content=figure_content, metadata=metadata))
            
    print(f"Loaded and partitioned {json_path.name} into {len(documents)} chunks:")
    counts = {"text": 0, "table": 0, "figure": 0}
    for doc in documents:
        counts[doc.metadata["type"]] += 1
    print(f" - Text chunks: {counts['text']}")
    print(f" - Table chunks: {counts['table']}")
    print(f" - Figure chunks: {counts['figure']}")
    
    return documents
