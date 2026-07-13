from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import gradio as gr


SUPPORTED_SUFFIXES = {".vcf", ".csv", ".txt"}


def _as_paths(files: str | Iterable[str] | None) -> list[Path]:
    """Normalise the values returned by Gradio's single/multiple file inputs."""
    if not files:
        return []
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    return [Path(file) for file in files]


def _variant_count(path: Path) -> int:
    """Return a lightweight row count without loading a potentially large file."""
    suffix = path.suffix.lower()
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            if suffix == ".vcf" and line.startswith("#"):
                continue
            count += 1

    if suffix == ".vcf":
        return count

    # CSV/TXT inputs may have either a conventional header or a VEP-style
    # header beginning with '#'. In both cases the first content row is a header.
    return max(count - 1, 0)


def review_uploads(
    pathogenic_files: str | list[str] | None,
    benign_files: str | list[str] | None,
) -> str:
    """Validate the two labelled cohorts and show a compact upload summary."""
    cohorts = {
        "Pathogenic": _as_paths(pathogenic_files),
        "Benign": _as_paths(benign_files),
    }
    problems: list[str] = []
    summaries: list[str] = []

    for label, paths in cohorts.items():
        if not paths:
            problems.append(f"Add at least one {label.lower()} file.")
            continue

        unsupported = [path.name for path in paths if path.suffix.lower() not in SUPPORTED_SUFFIXES]
        if unsupported:
            problems.append(f"{label}: unsupported file type — {', '.join(unsupported)}")
            continue

        try:
            count = sum(_variant_count(path) for path in paths)
        except OSError as exc:
            problems.append(f"{label}: could not read the upload ({exc}).")
            continue

        file_word = "file" if len(paths) == 1 else "files"
        variant_word = "variant row" if count == 1 else "variant rows"
        summaries.append(
            f"<strong>{label}:</strong> {len(paths)} {file_word} · "
            f"{count:,} {variant_word}"
        )

    if problems:
        return "### Almost there\n" + "\n".join(f"- {problem}" for problem in problems)

    return (
        '<div class="upload-success">'
        '<span class="success-mark">✓</span>'
        '<div><strong>Files are ready</strong><br>'
        + "<br>".join(summaries)
        + "</div></div>"
    )


CSS = """
:root {
  --dogma-ink: #202123;
  --dogma-muted: #6b6f76;
  --dogma-line: #dedede;
  --dogma-soft: #f7f7f8;
  --dogma-accent: #10a37f;
}

.gradio-container {
  background: #ffffff;
  color: var(--dogma-ink);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#dogma-shell {
  max-width: 780px;
  margin: 0 auto;
  padding: clamp(48px, 10vh, 110px) 24px 48px;
}

#dogma-hero { text-align: center; margin-bottom: 34px; }
#dogma-hero h1 {
  margin: 0 0 12px;
  font-size: clamp(34px, 6vw, 50px);
  font-weight: 600;
  letter-spacing: -0.04em;
  line-height: 1.08;
}
#dogma-hero p {
  max-width: 570px;
  margin: 0 auto;
  color: var(--dogma-muted);
  font-size: 16px;
  line-height: 1.6;
}

#upload-row { gap: 14px; }
.upload-card {
  border: 1px solid var(--dogma-line) !important;
  border-radius: 18px !important;
  background: #fff !important;
  box-shadow: none !important;
  overflow: hidden;
}
.upload-card:hover { border-color: #b8bbbe !important; }
.upload-card label { font-weight: 600 !important; }
.upload-card .wrap { min-height: 158px; }

#continue-button {
  margin-top: 12px;
  min-height: 48px;
  border: 0;
  border-radius: 12px;
  background: var(--dogma-ink);
  color: white;
  font-weight: 600;
}
#continue-button:hover { background: #000; }

#upload-status { margin-top: 14px; }
#upload-status h3 { text-align: center; font-size: 17px; }
.upload-success {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 14px 18px;
  border-radius: 12px;
  background: #f1faf7;
  color: #185c4b;
  line-height: 1.5;
}
.success-mark {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 50%;
  background: var(--dogma-accent);
  color: white;
  font-weight: 700;
}

#format-note {
  margin-top: 24px;
  text-align: center;
  color: var(--dogma-muted);
  font-size: 13px;
}
#format-note p { margin: 0; }

@media (max-width: 640px) {
  #dogma-shell { padding: 42px 16px 28px; }
  #upload-row { flex-direction: column; }
}
"""


with gr.Blocks(title="DOGMA", css=CSS, theme=gr.themes.Base()) as demo:
    with gr.Column(elem_id="dogma-shell"):
        gr.Markdown(
            """
# Drop your variant files

Add your known **pathogenic** and **benign** variants. DOGMA accepts `.vcf`,
`.csv`, and `.txt` files.
""",
            elem_id="dogma-hero",
        )

        with gr.Row(elem_id="upload-row"):
            pathogenic_files = gr.File(
                label="Pathogenic variants",
                file_count="multiple",
                file_types=[".vcf", ".csv", ".txt"],
                type="filepath",
                elem_classes=["upload-card"],
            )
            benign_files = gr.File(
                label="Benign variants",
                file_count="multiple",
                file_types=[".vcf", ".csv", ".txt"],
                type="filepath",
                elem_classes=["upload-card"],
            )

        review_button = gr.Button("Continue", elem_id="continue-button")
        upload_status = gr.Markdown(elem_id="upload-status")

        gr.Markdown(
            "A VCF is not automatically pathogenic or benign. If it has no "
            "clinical-significance annotation, DOGMA uses the upload box as its label.",
            elem_id="format-note",
        )

        review_button.click(
            fn=review_uploads,
            inputs=[pathogenic_files, benign_files],
            outputs=upload_status,
            api_name=False,
            show_api=False,
        )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("DOGMA_HOST", "127.0.0.1"),
        server_port=int(os.getenv("DOGMA_PORT", "7860")),
        show_error=True,
        show_api=False,
    )
