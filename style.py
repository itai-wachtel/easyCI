"""The look of the page: CSS and small pieces of HTML. No maths and no logic.

The colours and fonts of the built-in widgets come from .streamlit/config.toml.
This file adds what the theme cannot do: the header, numbered sections, the
verdict card and the footer. Colours here are see-through (rgba) wherever
possible, so the same CSS works on the light and the dark background.
"""

from html import escape

import streamlit as st

CSS = """
<style>
/* Page width and the empty band Streamlit leaves at the top. */
[data-testid="stMainBlockContainer"] { max-width: 880px; padding-top: 2.5rem; }
[data-testid="stHeader"] { background: transparent; }

/* --- Header ------------------------------------------------------------ */
.eci-hero {
  position: relative; overflow: hidden;
  padding: 2.2rem 2.2rem 1.9rem; margin-bottom: 1.6rem; border-radius: 1.25rem;
  background:
    radial-gradient(circle at 12% 0%, rgba(99, 102, 241, 0.28), transparent 55%),
    radial-gradient(circle at 95% 100%, rgba(236, 72, 153, 0.20), transparent 50%),
    linear-gradient(135deg, rgba(79, 70, 229, 0.10), rgba(14, 165, 233, 0.06));
  border: 1px solid rgba(99, 102, 241, 0.25);
}
.eci-eyebrow {
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: #6366f1; margin-bottom: 0.4rem;
}
.eci-title {
  font-size: 3.4rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.03em;
  background: linear-gradient(90deg, #4f46e5, #8b5cf6 45%, #ec4899);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  display: inline-block;
}
.eci-tagline { font-size: 1.35rem; font-weight: 650; margin: 0.6rem 0 0.35rem; }
.eci-lead { font-size: 1rem; opacity: 0.75; margin: 0 0 1.1rem; max-width: 38rem; }
.eci-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.eci-badges span {
  font-size: 0.8rem; font-weight: 500; padding: 0.3rem 0.75rem; border-radius: 999px;
  background: rgba(127, 127, 160, 0.12); border: 1px solid rgba(127, 127, 160, 0.25);
}

/* --- Numbered section titles -------------------------------------------- */
.eci-section { display: flex; align-items: center; gap: 0.75rem; margin: 2.2rem 0 0.4rem; }
.eci-step {
  display: inline-flex; align-items: center; justify-content: center;
  width: 2rem; height: 2rem; border-radius: 0.6rem; flex: none;
  font-weight: 700; font-size: 0.95rem; color: white;
  background: linear-gradient(135deg, #4f46e5, #8b5cf6);
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
}
.eci-section-title { font-size: 1.6rem; font-weight: 750; letter-spacing: -0.02em; }
.eci-section-note { font-size: 0.92rem; opacity: 0.7; margin: 0 0 0.8rem 2.75rem; }

/* --- Verdict card -------------------------------------------------------- */
.eci-verdict {
  display: flex; gap: 1rem; align-items: flex-start;
  padding: 1.25rem 1.4rem; margin: 0.4rem 0 1.2rem; border-radius: 1rem;
  border: 1px solid var(--eci-edge); background: var(--eci-tint);
  border-left: 6px solid var(--eci-accent);
}
.eci-real { --eci-accent: #12b76a; --eci-edge: rgba(18, 183, 106, 0.35); --eci-tint: rgba(18, 183, 106, 0.09); }
.eci-borderline { --eci-accent: #f59e0b; --eci-edge: rgba(245, 158, 11, 0.4); --eci-tint: rgba(245, 158, 11, 0.09); }
.eci-none { --eci-accent: #8189a8; --eci-edge: rgba(129, 137, 168, 0.4); --eci-tint: rgba(129, 137, 168, 0.09); }
.eci-icon {
  flex: none; width: 2.6rem; height: 2.6rem; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem; font-weight: 800; color: white; background: var(--eci-accent);
}
.eci-verdict-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem; }
.eci-verdict-reason { opacity: 0.85; line-height: 1.5; }
.eci-cautions { margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid var(--eci-edge); }
.eci-cautions b { font-size: 0.9rem; }
.eci-cautions ul { margin: 0.35rem 0 0; padding-left: 1.2rem; }
.eci-cautions li { font-size: 0.92rem; opacity: 0.85; margin: 0.2rem 0; }

/* --- Number cards (st.metric inside bordered columns) --------------------- */
[data-testid="stMetricLabel"] p { font-size: 0.85rem; font-weight: 600; opacity: 0.75; }
[data-testid="stMetricValue"] { font-weight: 750; letter-spacing: -0.02em; }

/* --- Sub-headings of the results ------------------------------------------ */
[data-testid="stMarkdownContainer"] h3 { font-weight: 700; letter-spacing: -0.01em; }

/* --- Footer ---------------------------------------------------------------- */
.eci-footer {
  margin-top: 3.5rem; padding-top: 1.2rem; text-align: center;
  font-size: 0.85rem; opacity: 0.6; border-top: 1px solid rgba(127, 127, 160, 0.25);
}
</style>
"""

ICONS = {"real": "✓", "borderline": "!", "none": "○"}


def _html(text):
    """Show raw HTML. The pieces are joined without line breaks, because
    Markdown would turn indented lines into code blocks."""
    st.markdown(text, unsafe_allow_html=True)


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    _html(
        '<div class="eci-hero">'
        '<div class="eci-eyebrow">Statistics for ML experiments</div>'
        '<div class="eci-title">easyCI</div>'
        '<p class="eci-tagline">Is your model\'s improvement real, or just luck?</p>'
        '<p class="eci-lead">Give easyCI the results of two models, and it tells you '
        "how much of the difference between them you can trust.</p>"
        '<div class="eci-badges">'
        "<span>Permutation tests</span><span>Exact McNemar</span>"
        "<span>Confidence intervals</span><span>Validated by simulation</span>"
        "</div></div>"
    )


def section(number, title, note=None):
    """A numbered section title, like "① Scores", with an optional line under it."""
    _html(
        f'<div class="eci-section"><span class="eci-step">{number}</span>'
        f'<span class="eci-section-title">{escape(title)}</span></div>'
        + (f'<p class="eci-section-note">{escape(note)}</p>' if note else "")
    )


def verdict_card(verdict, cautions):
    """The verdict as a card with an icon, and any cautions inside it."""
    caution_html = ""
    if cautions:
        items = "".join(f"<li>{escape(caution)}</li>" for caution in cautions)
        caution_html = f'<div class="eci-cautions"><b>Before you rely on this:</b><ul>{items}</ul></div>'
    _html(
        f'<div class="eci-verdict eci-{verdict.kind}">'
        f'<div class="eci-icon">{ICONS[verdict.kind]}</div>'
        f'<div><div class="eci-verdict-title">{escape(verdict.title)}</div>'
        f'<div class="eci-verdict-reason">{escape(verdict.reason)}</div>'
        f"{caution_html}</div></div>"
    )


def footer():
    _html(
        '<div class="eci-footer">easyCI · built by Itai Wachtel · '
        "every method checked on thousands of simulated experiments</div>"
    )
