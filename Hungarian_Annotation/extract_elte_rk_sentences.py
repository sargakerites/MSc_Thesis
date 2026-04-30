import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "Elte_Regenykorpusz"
OUTPUT_DIR = PROJECT_ROOT / "Elte_Rk_Sentences"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def extract_paragraphs(xml_path: Path):
    """Extract paragraph texts from a TEI XML file and return them as a list of strings."""

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        paragraphs = []
        for p in root.findall(".//tei:p", NS):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        return paragraphs

    except Exception as e:
        print(f"Error in {xml_path}: {e}")
        return []


def split_into_sentences(text: str):
    """Normalize whitespace and split paragraph text into sentences longer than 10 characters."""

    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    return sentences


def process_file(xml_path: Path, out_path: Path):
    """Extract paragraphs from one XML file, split them into sentences, and save them as JSON."""

    paragraphs = extract_paragraphs(xml_path)

    sentences = []
    for p in paragraphs:
        sentences.extend(split_into_sentences(p))

    data = [{"sentence": s} for s in sentences]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    """Process all XML files from the input directory and save sentence-level JSON files."""

    xml_files = list(INPUT_DIR.glob("*.xml"))

    print(f"Found {len(xml_files)} XML files")

    for xml_file in tqdm(xml_files):
        out_file = OUTPUT_DIR / (xml_file.stem + ".json")
        process_file(xml_file, out_file)


if __name__ == "__main__":
    main()