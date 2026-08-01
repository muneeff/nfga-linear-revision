from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


EXPECTED_TABLE_LABELS = (
    "tab:final_forecast",
    "tab:nfga_ablation",
    "tab:detector_macro",
    "tab:resource_cost",
)


def check_pdf(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "pdf_header_valid": False}

    size = path.stat().st_size
    with path.open("rb") as file:
        header = file.read(5)

    return {
        "exists": True,
        "size_bytes": size,
        "pdf_header_valid": header == b"%PDF-",
    }


def check_png(path: Path) -> dict:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "width_px": None,
            "height_px": None,
            "dpi_x": None,
            "dpi_y": None,
            "image_valid": False,
        }

    size = path.stat().st_size

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (None, None))
            dpi_x = dpi[0] if isinstance(dpi, tuple) else None
            dpi_y = dpi[1] if isinstance(dpi, tuple) else None

        valid = width > 0 and height > 0 and size > 0
    except Exception:
        width = height = dpi_x = dpi_y = None
        valid = False

    return {
        "exists": True,
        "size_bytes": size,
        "width_px": width,
        "height_px": height,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "image_valid": valid,
    }


def build_contact_sheets(
    figure_paths: list[Path],
    output_dir: Path,
    columns: int = 2,
    thumb_width: int = 900,
    thumb_height: int = 600,
    title_height: int = 55,
    page_rows: int = 3,
) -> list[Path]:
    output_paths: list[Path] = []
    figures_per_page = columns * page_rows

    for page_index in range(
        math.ceil(len(figure_paths) / figures_per_page)
    ):
        page_paths = figure_paths[
            page_index * figures_per_page :
            (page_index + 1) * figures_per_page
        ]

        page_width = columns * thumb_width
        page_height = page_rows * (thumb_height + title_height)
        sheet = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(sheet)

        for index, path in enumerate(page_paths):
            row = index // columns
            column = index % columns
            x0 = column * thumb_width
            y0 = row * (thumb_height + title_height)

            with Image.open(path).convert("RGB") as image:
                contained = ImageOps.contain(
                    image,
                    (thumb_width - 30, thumb_height - 30),
                )
                x_image = x0 + (thumb_width - contained.width) // 2
                y_image = y0 + title_height + (
                    thumb_height - contained.height
                ) // 2
                sheet.paste(contained, (x_image, y_image))

            draw.text((x0 + 15, y0 + 17), path.stem, fill="black")

        output_path = (
            output_dir
            / f"manuscript_figure_contact_sheet_{page_index + 1}.png"
        )
        sheet.save(output_path, dpi=(150, 150))
        output_paths.append(output_path)

    return output_paths


def audit_tex(tex_path: Path) -> dict:
    if not tex_path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "table_count": 0,
            "balanced_table_environments": False,
            "balanced_tabular_environments": False,
            "required_labels_present": False,
            "literal_hline_found": False,
            "unmatched_brace_count": None,
        }

    text = tex_path.read_text(encoding="utf-8")
    table_begin = len(re.findall(r"\\begin\{table\*?\}", text))
    table_end = len(re.findall(r"\\end\{table\*?\}", text))
    tabular_begin = text.count(r"\begin{tabular}")
    tabular_end = text.count(r"\end{tabular}")

    brace_balance = 0
    for char in text:
        if char == "{":
            brace_balance += 1
        elif char == "}":
            brace_balance -= 1

    return {
        "exists": True,
        "size_bytes": tex_path.stat().st_size,
        "table_count": table_begin,
        "balanced_table_environments": table_begin == table_end,
        "balanced_tabular_environments": tabular_begin == tabular_end,
        "required_labels_present": all(
            label in text for label in EXPECTED_TABLE_LABELS
        ),
        "literal_hline_found": "hline" in text.replace(r"\hline", ""),
        "unmatched_brace_count": abs(brace_balance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Stage 14 manuscript figures and LaTeX tables, "
            "and create visual contact sheets."
        )
    )
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = assets_dir / "figures"
    tables_dir = assets_dir / "tables"
    manifest_path = assets_dir / "figure_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    if "figure" not in manifest.columns:
        raise ValueError("figure_manifest.csv is missing the 'figure' column")

    rows: list[dict] = []
    png_paths: list[Path] = []

    for figure_name in manifest["figure"].astype(str):
        png_path = figures_dir / f"{figure_name}.png"
        pdf_path = figures_dir / f"{figure_name}.pdf"

        png_info = check_png(png_path)
        pdf_info = check_pdf(pdf_path)
        png_paths.append(png_path)

        dpi_ok = (
            png_info["dpi_x"] is None
            or abs(float(png_info["dpi_x"]) - 300.0) <= 2.0
        )

        rows.append(
            {
                "figure": figure_name,
                "png_exists": png_info["exists"],
                "png_valid": png_info["image_valid"],
                "png_size_bytes": png_info["size_bytes"],
                "width_px": png_info["width_px"],
                "height_px": png_info["height_px"],
                "dpi_x": png_info["dpi_x"],
                "dpi_y": png_info["dpi_y"],
                "dpi_approximately_300": dpi_ok,
                "pdf_exists": pdf_info["exists"],
                "pdf_header_valid": pdf_info["pdf_header_valid"],
                "pdf_size_bytes": pdf_info["size_bytes"],
                "figure_pass": bool(
                    png_info["exists"]
                    and png_info["image_valid"]
                    and pdf_info["exists"]
                    and pdf_info["pdf_header_valid"]
                    and png_info["size_bytes"] > 0
                    and pdf_info["size_bytes"] > 0
                ),
            }
        )

    figure_audit = pd.DataFrame(rows)
    figure_audit.to_csv(
        output_dir / "figure_asset_audit.csv",
        index=False,
    )

    missing_pngs = [path for path in png_paths if not path.exists()]
    contact_sheets: list[Path] = []

    if not missing_pngs:
        contact_sheets = build_contact_sheets(
            png_paths,
            output_dir,
        )

    tex_path = tables_dir / "manuscript_tables.tex"
    tex_audit = audit_tex(tex_path)

    overall_pass = bool(
        figure_audit["figure_pass"].all()
        and tex_audit["exists"]
        and tex_audit["balanced_table_environments"]
        and tex_audit["balanced_tabular_environments"]
        and tex_audit["required_labels_present"]
        and not tex_audit["literal_hline_found"]
        and tex_audit["unmatched_brace_count"] == 0
    )

    summary = {
        "assets_dir": str(assets_dir),
        "manifest_figure_count": int(len(manifest)),
        "figures_passing": int(figure_audit["figure_pass"].sum()),
        "figures_failing": int((~figure_audit["figure_pass"]).sum()),
        "all_figures_pass": bool(figure_audit["figure_pass"].all()),
        "latex_table_audit": tex_audit,
        "contact_sheets": [str(path) for path in contact_sheets],
        "overall_structural_pass": overall_pass,
        "manual_review_required": [
            "Check that text is readable at journal column width.",
            "Check that axes, legends, and labels are not clipped.",
            "Check that numerical annotations do not overlap bars.",
            "Check that the Electricity Prophet/XGBoost tie note is visible.",
            "Check that all table captions and footnotes match manuscript claims.",
        ],
    }

    (output_dir / "manuscript_asset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown = [
        "# Manuscript asset audit",
        "",
        f"- Manifest figures: {summary['manifest_figure_count']}",
        f"- Figures passing structural checks: {summary['figures_passing']}",
        f"- Figures failing structural checks: {summary['figures_failing']}",
        f"- Overall structural pass: {summary['overall_structural_pass']}",
        "",
        "## LaTeX table checks",
        "",
    ]
    for key, value in tex_audit.items():
        markdown.append(f"- {key}: {value}")

    markdown.extend(
        ["", "## Manual visual checks still required", ""]
    )
    for item in summary["manual_review_required"]:
        markdown.append(f"- {item}")

    (output_dir / "manuscript_asset_audit.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print("\n===== Stage 14 asset audit completed =====")
    print(figure_audit.to_string(index=False))
    print("\nLaTeX audit:")
    print(json.dumps(tex_audit, indent=2))
    print("\nOverall structural pass:", overall_pass)
    print("\nContact sheets:")
    for path in contact_sheets:
        print(path)


if __name__ == "__main__":
    main()
