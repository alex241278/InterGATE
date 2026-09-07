#!/usr/bin/env python3
"""Apply typography/wording-only edits. No training or evaluation is executed.

The patch is validated in memory, backs up original bytes, and writes an audit.
Use --apply to write. Without it, only the planned changes are displayed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

TAG = 'CMPB_REVIEW_20260907'


def replace_required(text, old, new):
    if old not in text:
        raise ValueError(f'Expected source fragment not found: {old[:100]!r}')
    return text.replace(old, new)


def patch_notebook(text: str, kind: str) -> str:
    nb = json.loads(text)
    if any(TAG in ''.join(c.get('source', [])) for c in nb['cells']):
        return text
    changed = 0
    for cell in nb['cells']:
        old = ''.join(cell.get('source', []))
        s = old
        if kind == 'main':
            s = s.replace("OUTPUT_ROOT = RESULTS_ROOT / '10_manuscript_figures'",
                          "OUTPUT_ROOT = RESULTS_ROOT / '10_manuscript_figures_large_fonts'")
            if 'STYLE = {' in s:
                s, n = re.subn(r'STYLE = \{.*?\n\}',
                    'from intergate.publication_style import load_style, apply_style, prepare_figure\n'
                    f'# {TAG}: all publication sizes are in configs/publication_style.json\n'
                    'STYLE = load_style(PROJECT_ROOT)\napply_style(STYLE)', s, count=1, flags=re.S)
                if n != 1:
                    raise ValueError('Could not identify the main style dictionary.')
            s = s.replace("    for extension in ('pdf', 'svg', 'png'):\n",
                          "    prepare_figure(fig, STYLE)\n    for extension in ('pdf', 'svg', 'png'):\n")
            s = s.replace("A  Cross-class strict-core specificity", "A  Signature coverage")
            s = s.replace("ax_a.set_ylabel('Strict core class')", "ax_a.set_ylabel('Consensus network class')")
            s = s.replace("ax_a.set_xlabel('Directional annotation class')", "ax_a.set_xlabel('Signature annotation')")
            s = s.replace("B  Directional Basal-like recovery", "B  Basal-like consensus")
            s = s.replace("C  Functional coherence beyond preserved topology", "C  Hallmark coherence in consensus networks")
            s = s.replace("FIG3_TITLE = 'Subtype-specific recovery and functional coherence'",
                          "FIG3_TITLE = 'Biological recovery in consensus networks'")
            if cell['cell_type'] == 'code':
                s = re.sub(r'fontsize\s*=\s*(\d+(?:\.\d+)?)',
                    lambda m: ("fontsize=STYLE['annotation_size']" if float(m[1]) < 12
                               else ("fontsize=STYLE['figure_title_size']" if float(m[1]) >= 15 else m[0])), s)
                s = re.sub(r'(SUPP_CONNECT_(?:SELECTED|AUX|ALL)_LABEL_SIZE\s*=\s*)\d+(?:\.\d+)?',
                           r"\1STYLE['gene_label_size']", s)
                if "    os.replace(temporary, target)" in s:
                    s = s.replace("    os.replace(temporary, target)",
                        "    os.replace(temporary, target)\n"
                        "        aliases = {'Figure_structural_recovery': 'Figure2',\n"
                        "                   'Figure_Basal_strict_core': 'Figure3',\n"
                        "                   'Figure_biological_validation': 'Figure4',\n"
                        "                   'Figure_external_structural_concordance': 'Figure5',\n"
                        "                   'Figure_controls_sensitivity': 'FigureS8'}\n"
                        "        if stem in aliases:\n"
                        "            import shutil\n"
                        "            shutil.copyfile(target, FIGURES / f'{aliases[stem]}.{extension}')")
        elif kind == 'supp':
            s = s.replace('"13_supplementary_figures"', '"13_supplementary_figures_large_fonts"')
            if 'plt.rcParams.update({' in s:
                s = ('import sys\n'
                     '_style_root = locate_project_root()\n'
                     'if _style_root is None:\n'
                     '    raise FileNotFoundError("Run inside the patched InterGATE project.")\n'
                     'if str(_style_root) not in sys.path:\n'
                     '    sys.path.insert(0, str(_style_root))\n'
                     'from intergate.publication_style import load_style, apply_style, prepare_figure\n'
                     f'# {TAG}\n'
                     'PUBLICATION_STYLE = load_style(_style_root)\n' + s)
                s = s.replace('\n\ndef export_figure(', '\n\napply_style(PUBLICATION_STYLE)\n\ndef export_figure(')
                s = s.replace('    paths: dict[str, Path] = {}',
                              '    prepare_figure(fig, PUBLICATION_STYLE)\n    paths: dict[str, Path] = {}')
            if cell['cell_type'] == 'code':
                s = re.sub(r'fontsize\s*=\s*(\d+(?:\.\d+)?)',
                    lambda m: ("fontsize=PUBLICATION_STYLE['annotation_size']" if float(m[1]) < 12
                               else "fontsize=PUBLICATION_STYLE['figure_title_size']"), s)
        if s != old:
            changed += 1
            cell['source'] = s.splitlines(keepends=True)
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
    if not changed:
        raise ValueError(f'No expected edits found in {kind} notebook.')
    nb['cells'].insert(1, {'cell_type': 'markdown', 'metadata': {},
        'id': 'cmpb-review-note', 'source': [
        '## Revisión CMPB: presentación, sin reentrenamiento\n',
        'Los tamaños se controlan en `configs/publication_style.json`. Las figuras se guardan '
        'en una carpeta nueva terminada en `_large_fonts`. No vuelva a ejecutar 06 ni 07 '
        'para regenerar figuras. Los cambios de fuente no alteran los resultados bloqueados.\n',
        'El panel de validación representa consensos Exact-K, no núcleos estrictos 5/5. '
        'La fracción bruta de firmas se denomina cobertura; no es una especificidad ajustada '
        'por el tamaño de la firma.\n']})
    return json.dumps(nb, ensure_ascii=False, indent=1) + '\n'


def patch_module(text: str) -> str:
    if TAG in text:
        return text
    text = replace_required(text, 'from .posthoc_biology_common import load_locked_signature_resources',
        'from .posthoc_biology_common import load_locked_signature_resources\n'
        'from .publication_style import load_style, apply_style, prepare_figure\n'
        f'# {TAG}\nPUBLICATION_STYLE = load_style()\napply_style(PUBLICATION_STYLE)')
    text = text.replace('MANUSCRIPT_OUT = RESULTS / "10_manuscript_figures"',
                        'MANUSCRIPT_OUT = RESULTS / "10_manuscript_figures_large_fonts"')
    text = replace_required(text, '    for ext in ("pdf", "svg"):',
                            '    FIGURES.mkdir(parents=True, exist_ok=True)\n'
                            '    prepare_figure(fig, PUBLICATION_STYLE)\n'
                            '    for ext in ("pdf", "svg"):')
    text = text.replace('A  Cross-class locked-network specificity', 'A  Signature coverage')
    text = text.replace('A  Cross-class strict-core specificity', 'A  Signature coverage')
    text = text.replace('ax.set_ylabel("Locked network class")', 'ax.set_ylabel("Consensus network class")')
    text = text.replace('ax.set_ylabel("Strict core class")', 'ax.set_ylabel("Consensus network class")')
    text = text.replace('B  Directional Basal-like recovery', 'B  Basal-like consensus')
    text = text.replace('C  Functional coherence beyond preserved topology', 'C  Hallmark coherence in consensus networks')
    text = text.replace('"Subtype-specific recovery and functional coherence"', '"Biological recovery in consensus networks"')
    text = text.replace('    summary, strict_edges = structural_tables()',
                        '    apply_style(PUBLICATION_STYLE)\n    summary, strict_edges = structural_tables()')
    text = re.sub(r'fontsize\s*=\s*(\d+(?:\.\d+)?)',
        lambda m: ("fontsize=PUBLICATION_STYLE['annotation_size']" if float(m[1]) < 12
                   else "fontsize=PUBLICATION_STYLE['figure_title_size']"), text)
    return text


def run(root: Path, apply: bool = False) -> dict:
    specs = {'intergate/final_figures.py': patch_module,
             'notebooks/10_figuras_definitivas.ipynb': lambda t: patch_notebook(t, 'main'),
             'notebooks/13_generate_supplementary_figures_S2_S3.ipynb': lambda t: patch_notebook(t, 'supp')}
    pending = []
    for rel, patch in specs.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_bytes()
        edited = patch(original.decode('utf-8')).encode('utf-8')
        if original != edited:
            if path.suffix == '.py':
                compile(edited, str(path), 'exec')
            else:
                for i, cell in enumerate(json.loads(edited)['cells']):
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), f'{path}:cell{i}', 'exec')
            pending.append((rel, original, edited))
    audit = {'revision': TAG, 'applied': apply, 'training_modified': False,
             'normalization_modified': False, 'changes': [
                 {'path': rel, 'before_sha256': hashlib.sha256(old).hexdigest(),
                  'after_sha256': hashlib.sha256(new).hexdigest()} for rel, old, new in pending]}
    if apply:
        backup = root / '_backup_before_cmpb_review'
        for rel, old, new in pending:
            b = backup / rel
            b.parent.mkdir(parents=True, exist_ok=True)
            if b.exists() and b.read_bytes() != old:
                raise FileExistsError(f'Refusing to overwrite a different backup: {b}')
        for rel, old, new in pending:
            b = backup / rel
            b.write_bytes(old)
            target = root / rel
            temp = target.with_suffix(target.suffix + '.cmpb-tmp')
            temp.write_bytes(new)
            temp.replace(target)
        audit_path = root / 'cmpb_patch_audit.json'
        if pending or not audit_path.exists():
            audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.apply), indent=2))
