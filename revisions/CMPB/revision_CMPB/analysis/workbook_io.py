"""Read scientific CSVs and verify their serialized values in a review workbook."""
import csv
import gzip
import json
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def value(text, field):
    if text == "":
        return None
    if field in {"source_gene", "target_gene", "gene_a", "gene_b", "sample", "cohort"}:
        return text
    if text in {"True", "False"}:
        return text == "True"
    if re.fullmatch(r"-?(?:0|[1-9]\d*)", text):
        n = int(text)
        return n if abs(n) < 2**53 else text
    if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", text):
        n = float(text)
        if math.isfinite(n):
            return n
    return text


def snapshot(sources, folder):
    for i, spec in enumerate(sources):
        path = Path(spec["path"])
        with (gzip.open if path.suffix == ".gz" else open)(path, "rt", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            headers = next(reader)
            if not headers or any(not field for field in headers) or len(set(headers)) != len(headers):
                raise ValueError(f"Empty or duplicate field in {path.name}")
            exact = spec.get("specification")
            spec_col = headers.index("specification") if exact else None
            rows = [headers]
            for row in reader:
                if len(row) != len(headers):
                    raise ValueError(f"Malformed record in {path.name}")
                if exact and row[spec_col] != exact:
                    continue
                rows.append([value(text, field) for text, field in zip(row, headers)])
        if len(rows) < 2 and not spec.get("allowEmpty", False):
            raise ValueError(f"Empty result in {path.name}")
        (folder / f"{i}.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def equivalent(left, right):
    if left is None or right is None or type(left) in (str, bool) or type(right) in (str, bool):
        return type(left) is type(right) and left == right
    return math.isclose(left, right, rel_tol=2e-14, abs_tol=0.0)


def column(reference):
    v = 0
    for ch in reference:
        if not ch.isalpha():
            break
        v = v * 26 + ord(ch.upper())-64
    return v-1


def excel_rows(archive, part, strings):
    with archive.open(part) as src:
        for _, element in ET.iterparse(src, events=("end",)):
            if element.tag != NS+"row":
                continue
            row = {}
            for cell in element.findall(NS+"c"):
                if cell.find(NS+"f") is not None or cell.get("t") == "e":
                    raise ValueError(f"Unexpected formula/error {part} {cell.get('r')}")
                val = cell.findtext(NS+"v")
                t = cell.get("t")
                if t == "inlineStr":
                    val = "".join(x.text or "" for x in cell.iter(NS+"t"))
                elif t == "s" and val is not None:
                    val = strings[int(val)]
                elif t == "b" and val is not None:
                    val = val == "1"
                elif t != "str" and val is not None:
                    val = float(val)
                if val is not None:
                    row[column(cell.get("r"))] = val
            yield int(element.get("r")), row
            element.clear()


def verify(sources, folder, xlsx):
    expected_names = ["Guide"]+[source["name"] for source in sources]
    with ZipFile(xlsx) as book:
        main = ET.fromstring(book.read("xl/workbook.xml"))
        relations = {r.get("Id"):r.get("Target") for r in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))}
        sheets = main.find(NS+"sheets")
        assert [s.get("name") for s in sheets] == expected_names
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            for item in ET.fromstring(book.read("xl/sharedStrings.xml")):
                strings.append("".join(x.text or "" for x in item.iter(NS+"t")))
        total = 0
        for j, sheet in enumerate(sheets):
            expected = json.loads((folder/("guide.json" if j==0 else f"{j-1}.json")).read_text())
            dest = relations[sheet.get(REL+"id")]
            part = dest[1:] if dest.startswith("/") else "xl/"+dest
            seen = set()
            for ix, actual in excel_rows(book,part,strings):
                line = expected[ix-1] if ix <= len(expected) else []
                if actual or any(x is not None for x in line):
                    for c in set(range(len(line)))|set(actual):
                        val = line[c] if c<len(line) else None
                        if not equivalent(val,actual.get(c)):
                            raise ValueError(f"{sheet.get('name')} row={ix}, col={c+1}: {val!r} vs {actual.get(c)!r}")
                    seen.add(ix)
            if not all(i in seen for i,row in enumerate(expected,1) if any(x is not None for x in row)):
                raise ValueError(f"Missing values in {sheet.get('name')}")
            total += len(expected)
    print(f"Verified {len(expected_names)} sheets and {total:,} rows against the source snapshot.")


if __name__ == "__main__":
    action, directory = sys.argv[1:3]
    base = Path(directory)
    specs = json.loads((base/"sources.json").read_text())
    if action == "snapshot":
        snapshot(specs,base)
    elif action == "verify":
        verify(specs,base,sys.argv[3])
    else:
        raise ValueError(action)
