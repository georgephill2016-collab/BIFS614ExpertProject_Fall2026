from pathlib import Path
from docx import Document


project_folder = Path(__file__).resolve().parent

input_folder = project_folder / "course_content" / "original_docx" / "lectures"
output_folder = project_folder / "course_content" / "extracted_txt" / "lectures"

output_folder.mkdir(parents=True, exist_ok=True)

docx_files = sorted(input_folder.glob("*.docx"))

if not docx_files:
    print(f"No Word documents were found in: {input_folder}")

for docx_file in docx_files:
    document = Document(docx_file)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    output_file = output_folder / f"{docx_file.stem}.txt"
    output_file.write_text("\n\n".join(paragraphs), encoding="utf-8")

    print(f"Converted: {docx_file.name} -> {output_file.name}")

print(f"Finished converting {len(docx_files)} lecture document(s).")
