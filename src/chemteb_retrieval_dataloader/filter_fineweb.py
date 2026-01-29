from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from chemteb_retrieval_dataloader.chunking import chunk_text


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def _iter_stream(dataset: str, config: str, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset, name=config, split=split, streaming=True)
    for row in ds:
        yield row


def run_filter_fineweb(
    *,
    dataset: str,
    config: str,
    split: str,
    out_dir: str,
    max_rows: int,
    max_kept: int,
    keywords: list[str],
    min_weak_keyword_hits: int,
    min_strong_keyword_hits: int,
    url_contains: list[str],
    url_regex: str,
    disable_domain_filter: bool,
    chunk_tokens: int,
    chunk_overlap: int,
    resume: bool,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs_path = out / "docs.jsonl"
    chunks_path = out / "chunks.jsonl"
    stats_path = out / "stats.json"
    progress_path = out / "progress.json"

    n_seen = 0
    n_kept = 0

    resume_seen = 0
    if resume:
        for p in (stats_path, progress_path):
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    resume_seen = int(d.get("n_seen") or 0)
                    n_kept = int(d.get("n_kept") or 0)
                except Exception:
                    resume_seen = 0
                break

    if resume_seen > 0:
        n_seen = resume_seen

    url_contains_norm = [s.lower() for s in (url_contains or []) if s]
    url_re = re.compile(url_regex, re.IGNORECASE) if url_regex else None

    # Keyword filtering:
    # We keep two pools:
    # - strong keywords: "chemistry paper markers" (InChI/SMILES/NMR + synthesis/characterization phrasing)
    # - weak keywords: paper-ish chem phrasing (experimental section, spectral data, etc.)
    #
    # Default behavior in keyword mode:
    # - require >= min_keyword_hits total hits across (strong+weak)
    # - require >= min_strong_keyword_hits hits from strong pool
    #
    # If the user passes --keyword, we treat *all* provided keywords as strong markers.
    # "Strong" should mean: common in chemistry papers (esp. synthesis/materials),
    # and relatively uncommon in generic bio/med/news pages.
    default_strong_keywords = [
        # Identifiers / representations
        "inchi=",
        "inchi key",
        "smiles:",
        "smiles=",
        "canonical smiles:",
        "canonical smiles=",
        "cas number",
        "iupac name",
        # Spectroscopy / analytical methods
        "1h nmr",
        "13c nmr",
        "19f nmr",
        "31p nmr",
        "chemical shift",
        # NMR solvents / reporting conventions
        "cdcl3",
        "dmso-d6",
        "d2o",
        "acetone-d6",
        "methanol-d4",
        "cd3od",
        "thf-d8",
        "tetramethylsilane",
        "brucker",  # common misspelling
        "bruker",
        "jeol",
        # MS reporting (chem-heavy when paired with other strong markers)
        "hrms",
        "esi-ms",
        "esi-hrms",
        "maldi",
        # Crystallography / structure determination
        "scxrd",
        "single-crystal x-ray",
        "single crystal x-ray",
        "x-ray crystallography",
        "x ray crystallography",
        # Separations
        "column chromatography",
        "flash chromatography",
        "thin layer chromatography",
        "silica gel",
        # Synthesis / lab language
        "anhydrous",
        "inert atmosphere",
        "under argon",
        "argon atmosphere",
        "under nitrogen",
        "nitrogen atmosphere",
        "glovebox",
        "schlenk",
        "degassed",
        # High-precision chemistry paper phrases
        "chemical synthesis",
        "reaction yield",
        "molecular structure determination",
        "x-ray crystal structure",
        "x ray crystal structure",
        "calcd for",
        "found:",
        "elemental analysis",
        "spectral data",
        "general procedure",
        "experimental section",
        # Even more strong markers (more recall)
        "chemrxiv",
        "rxiv",
        "supporting information:",
        "supplementary information:",
        # Synthesis-specific reagents/notations
        "pd/c",
        "tmscl",
        "tbaf",
        "n-buli",
        "nbuli",
        "lihm ds",
        "lihmds",
        "lithium hexamethyldisilazide",
        # Structural identifiers & registries
        "mol file",
        "sdf file",
        "structure deposited",
        "ccdc",
        "cambridge crystallographic data centre",
        "crystallographic data have been deposited",
        "cif file",
        # Expanded NMR reporting language
        "δ ppm",
        "δ (ppm)",
        "multiplet",
        "doublet",
        "triplet",
        "quartet",
        "singlet",
        "br s",
        "br d",
        "br t",
        "coupling constant",
        "j =",
        "j-value",
        "j coupling",
        # More solvent-specific deuterated forms
        "c6d6",
        "cd2cl2",
        "acetonitrile-d3",
        "pyridine-d5",
        "chloroform-d",
        # High-precision mass spec conventions
        "found for",
        "[m+h]+",
        "[m+na]+",
        "[m-k]+",
        "[m-h]-",
        "exact mass",
        "isotopic pattern",
        # Crystal structure language (chem-exclusive)
        "thermal ellipsoids",
        "asymmetric unit",
        "refinement converged",
        "r1 value",
        "wr2 value",
        "space group",
        "unit cell parameters",
        # Synthetic chemistry operations (very SI-specific)
        "added dropwise",
        "stirred overnight",
        "under reflux",
        "cooled in ice bath",
        "quenched with",
        "organic layer was separated",
        "aqueous layer was extracted",
        "combined organic extracts",
        "dried over na2so4",
        "concentrated in vacuo",
        "under reduced pressure",
        # Reagent shorthand chemists use constantly
        "nbs",
        "ddq",
        "pcc",
        "pdc",
        "dibal-h",
        "lah",
        "nab h4",
        "tfa",
        "tea",
        "dipea",
        "hünig’s base",
        "hunig's base",
        # Paper-specific chemistry phrases
        "all reactions were carried out under",
        "commercial reagents were used without further purification",
        "yields refer to isolated products",
        "spectra are consistent with literature",
        # Named chemistry-exclusive data types
        "single-crystal structure",
        "crystal structure analysis",
        "nmr spectra are shown",
        "hrms data",
        "ftir spectrum",
    ]
    # "Weak" keywords can be chemistry-related, but are common across many sciences.
    default_weak_keywords = [
        # Cross-domain analytical terms (bio/med also uses them heavily)
        "doi:",
        "doi.org/",
        "chromatography",
        "mass spectrometry",
        "ms/ms",
        "hplc",
        "gc-ms",
        "lc-ms",
        "lc–ms",
        "uv-vis spectroscopy",
        "ftir spectroscopy",
        "raman spectroscopy",
        "x-ray diffraction",
        "xray diffraction",
        "powder x-ray diffraction",
        "pxrd",
        "xps spectrum",
        "ir spectroscopy",
        "spectral analysis",
        "ultraviolet-visible spectroscopy",
        "ultraviolet visible spectroscopy",
        "uv-vis",
        "uv–vis",
        "ftir",
        "raman",
        "xps",
        "xrd",
        "m/z",
        # Common chem paper / SI vocabulary (often appears alongside strong markers)
        "synthesis",
        "reaction",
        "reaction mechanism",
        "reaction kinetics",
        "rate constant",
        "reaction rate",
        "equilibrium constant",
        "activation energy",
        "arrhenius equation",
        "kinetics",
        "thermodynamics",
        "stoichiometry",
        "catalyst",
        "catalysis",
        "ligand",
        "reagent",
        "solvent",
        "precursor",
        "derivative",
        "functional group",
        "oxidation",
        "reduction",
        "purification",
        "crystallography",
        "compound",
        "molecule",
        "synthesized",
        "prepared",
        "reaction mixture",
        "stirred",
        "heated",
        "reflux",
        "cooled to room temperature",
        "quenched",
        "workup",
        "extracted",
        "washed with",
        "dried over",
        "filtered",
        "concentrated under reduced pressure",
        "evaporated under reduced pressure",
        "purified by",
        "gradient elution",
        # Units / quantities that show up in procedures
        "equiv",
        "mmol",
        "mol%",
        "mg",
        "ml",
        "°c",
        "rt",
        "k",
        "kelvin",
        "mol/l",
        "m (molarity)",
        "molarity",
        "ppm",
        "wt%",
        "ph",
        "yield %",
        "mhz",
        # Common solvents/reagents (weak: appear in many contexts, but useful in combination)
        "dcm",
        "dichloromethane",
        "ethyl acetate",
        "hexanes",
        "toluene",
        "acetonitrile",
        "methanol",
        "ethanol",
        "thf",
        "dmf",
        "water",
        "brine",
        "na2so4",
        "mgso4",
        "silica",
        # Chemical entities / structure language (often chemistry, but not paper-exclusive)
        "chemical formula",
        "molecular formula",
        "ionic species",
        "benzene ring",
        "aromatic",
        "alkyl chain",
        "polymer backbone",
        "monomer",
        "crosslinking",
        "coordination complex",
        "coordination chemistry",
        "metal complex",
        "transition metal",
        "metal oxide",
        "semiconductor",
        "thin film",
        "nanomaterial",
        "nanostructure",
        "nanoparticle",
        "crystal lattice",
        "crystalline phase",
        "lattice structure",
        "dopant",
        # Organic chemistry terms / named reactions (chem-heavy, but can appear in educational text)
        "alkane",
        "alkene",
        "alkyne",
        "ester",
        "amide",
        "ketone",
        "aldehyde",
        "carboxylic acid",
        "substitution reaction",
        "addition reaction",
        "elimination reaction",
        "grignard reaction",
        "suzuki coupling",
        "polymerization",
        # Analytical chemistry terms
        "limit of detection",
        "quantification",
        "sensitivity",
        "selectivity",
        "standard solution",
        "concentration",
        "concentration gradient",
        "assay",
        "sample preparation",
        "calibration curve",
        # More chemistry vocabulary (weak; helps recall)
        "reaction pathway",
        "mechanistic study",
        "mechanistic studies",
        "catalyst loading",
        "turnover frequency",
        "turnover number",
        "conversion",
        "isolated yield",
        "crude",
        "purity",
        "gcms",
        # More named reactions / common chemistry terms
        "suzuki",
        "heck",
        "sonogashira",
        "buchwald-hartwig",
        "stille",
        "negishi",
        "kumada",
        "click reaction",
        "diels-alder",
        "friedel-crafts",
        # More solvents/reagents abbreviations
        "meoh",
        "etoh",
        "mecn",
        "dioxane",
        "diethyl ether",
        "chloroform",
        "acetone",
        # More workup / purification language
        "in vacuo",
        "under vacuum",
        "reduced pressure",
        "rotavap",
        "tlc",
        "rf",
        # Paper-ish phrases
        "experimental section",
        "general procedure",
        "supporting information",
        "supplementary information",
        "characterization",
        "yield:",
        "yield",
        "melting point",
        "boiling point",
        # Registries / author metadata (paper-ish, not chemistry-specific)
        "orcid",
        # Crystal structure measurement language (weak; appears beyond chemistry)
        "angstrom",
        "å",
    ]

    if keywords:
        strong_kw_list = [k.lower() for k in keywords if k]
        weak_kw_list: list[str] = []
    else:
        strong_kw_list = [k.lower() for k in default_strong_keywords]
        weak_kw_list = [k.lower() for k in default_weak_keywords]

    # Ensure strong/weak are actually separate to avoid double-counting.
    strong_set = set(strong_kw_list)
    weak_kw_list = [w for w in weak_kw_list if w not in strong_set]

    def _compile_keywords(kw_list: list[str]) -> tuple[list[tuple[str, re.Pattern[str]]], list[str]]:
        # For short alpha-numeric keywords (e.g. "hplc"), require word boundaries
        # to avoid accidental substring matches in normal prose.
        rx_list: list[tuple[str, re.Pattern[str]]] = []
        substr_list: list[str] = []
        for k in kw_list:
            if k.replace("_", "").isalnum() and len(k) <= 4:
                rx_list.append((k, re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE)))
            else:
                substr_list.append(k)
        return rx_list, substr_list

    strong_rx, strong_sub = _compile_keywords(strong_kw_list)
    weak_rx, weak_sub = _compile_keywords(weak_kw_list)

    def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    excluded_domain_substrings = [
        # not research paper sources (very often encyclopedic/bloggy)
        "wikipedia.org",
        "wikia.com",
        "blogspot.",
        "wordpress.",
        "medium.com",
        "stackexchange.com",
        "quora.com",
    ]

    def _write_progress() -> None:
        _atomic_write_json(
            progress_path,
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "n_seen": n_seen,
                "n_kept": n_kept,
                "mode": "keyword",
                "min_weak_keyword_hits": min_weak_keyword_hits,
                "min_strong_keyword_hits": min_strong_keyword_hits,
            },
        )

    stream_iter: Iterator[dict[str, Any]] = _iter_stream(dataset, config, split)
    if resume_seen > 0:
        stream_iter = itertools.islice(stream_iter, resume_seen, None)

    last_ckpt_seen = n_seen
    last_ckpt_time = time.time()

    for row in stream_iter:
        n_seen += 1
        if max_rows and n_seen > max_rows:
            break

        text = (row.get("text") or "").strip()
        if not text:
            continue

        # Basic metadata (FineWeb-Edu rows usually have these fields).
        url = str(row.get("url") or "")
        dump = str(row.get("dump") or "")
        file_path = str(row.get("file_path") or "")
        row_id = str(row.get("id") or "")

        if url:
            ul = url.lower()
            if url_contains_norm and not all(s in ul for s in url_contains_norm):
                continue
            if url_re and not url_re.search(url):
                continue

        if (not disable_domain_filter) and url:
            u = url.lower()
            if any(s in u for s in excluded_domain_substrings):
                continue

        doc_id = _stable_id(row_id, url, dump, file_path)

        # Keyword filter (only mode we keep in this simplified codebase).
        lower = text.lower()
        hits = 0
        matched: list[str] = []
        strong_hits = 0
        strong_matched: list[str] = []
        weak_hits = 0
        weak_matched: list[str] = []

        for k, rx in strong_rx:
            if rx.search(text):
                hits += 1
                matched.append(k)
                strong_hits += 1
                strong_matched.append(k)
        for k in strong_sub:
            if k and k in lower:
                hits += 1
                matched.append(k)
                strong_hits += 1
                strong_matched.append(k)

        for k, rx in weak_rx:
            if rx.search(text):
                hits += 1
                matched.append(k)
                weak_hits += 1
                weak_matched.append(k)
        for k in weak_sub:
            if k and k in lower:
                hits += 1
                matched.append(k)
                weak_hits += 1
                weak_matched.append(k)

        # New default logic:
        # keep if (strong_hits >= min_strong_keyword_hits) OR (weak_hits >= min_weak_keyword_hits)
        if (strong_hits < max(0, int(min_strong_keyword_hits))) and (weak_hits < max(0, int(min_weak_keyword_hits))):
            continue

        # Keep doc
        doc_obj: dict[str, Any] = {
            "doc_id": doc_id,
            "source_row_id": row_id,
            "url": url,
            "dump": dump,
            "file_path": file_path,
            "language": row.get("language"),
            "language_score": row.get("language_score"),
            "fineweb_score": row.get("score"),
            "token_count": row.get("token_count"),
            "text": text,
            "keyword_filter": True,
            "keywords_matched": hits,
            "matched_keywords": matched[:20],
            "strong_keywords_matched": strong_hits,
            "matched_strong_keywords": strong_matched[:20],
            "weak_keywords_matched": weak_hits,
            "matched_weak_keywords": weak_matched[:20],
        }
        write_jsonl(docs_path, doc_obj)

        # Chunks (for retrieval indexing)
        for ch in chunk_text(doc_id=doc_id, text=text, chunk_tokens=chunk_tokens, overlap=chunk_overlap):
            write_jsonl(
                chunks_path,
                {
                    "chunk_id": ch.chunk_id,
                    "doc_id": ch.doc_id,
                    "start_token": ch.start_token,
                    "end_token": ch.end_token,
                    "text": ch.text,
                    "url": url,
                    "dump": dump,
                    "file_path": file_path,
                },
            )

        n_kept += 1
        if max_kept and n_kept >= max_kept:
            break

        if n_kept % 50 == 0:
            _write_stats(
                stats_path,
                dataset=dataset,
                config=config,
                split=split,
                n_seen=n_seen,
                n_kept=n_kept,
                mode="keyword",
                min_weak_keyword_hits=min_weak_keyword_hits,
                min_strong_keyword_hits=min_strong_keyword_hits,
            )

        # Resume checkpoints (by seen rows / time). This is what enables "continue where it left off".
        if (n_seen - last_ckpt_seen) >= 5000 or (time.time() - last_ckpt_time) >= 60:
            _write_progress()
            last_ckpt_seen = n_seen
            last_ckpt_time = time.time()

    _write_stats(
        stats_path,
        dataset=dataset,
        config=config,
        split=split,
        n_seen=n_seen,
        n_kept=n_kept,
        mode="keyword",
        min_weak_keyword_hits=min_weak_keyword_hits,
        min_strong_keyword_hits=min_strong_keyword_hits,
    )
    _write_progress()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_stats(path: Path, **data: Any) -> None:
    _atomic_write_json(path, dict(data))

