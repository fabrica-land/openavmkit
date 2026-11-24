"""Caller output contracts for accepted ENG-6008 dependency updates."""

from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd

from openavmkit.reports import _markdown_to_html
from openavmkit.utilities.excel import write_to_excel


def test_markdown_report_renders_ssh_autolink():
    html = _markdown_to_html("<ssh://example.com/reports>")
    assert (
        '<p><a href="ssh://example.com/reports">ssh://example.com/reports</a></p>'
        in html
    )


def test_markdown_report_preserves_valueless_hidden_attribute():
    html = _markdown_to_html("Internal report\n{: hidden}")
    assert "<p hidden>Internal report</p>" in html


def test_excel_export_preserves_function_like_string_literals(tmp_path):
    literal_formula = '=IF(1,"FILTER(""a"")","XLOOKUP(""b"")")'
    dynamic_formula = "=FILTER(B1:B2,B1:B2>0)"
    df = pd.DataFrame({"formula": [literal_formula, dynamic_formula]})
    path = tmp_path / "formulas.xlsx"
    write_to_excel(df, str(path), {})
    with ZipFile(path) as workbook:
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    literal = sheet.find(".//s:c[@r='A2']/s:f", ns)
    assert literal is not None
    assert literal.text == literal_formula[1:]
    assert literal.attrib == {}
    dynamic = sheet.find(".//s:c[@r='A3']/s:f", ns)
    assert dynamic is not None
    assert dynamic.text == "_xlfn._xlws.FILTER(B1:B2,B1:B2>0)"
    assert dynamic.get("t") == "array"
