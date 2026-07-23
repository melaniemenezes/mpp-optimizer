"""Generate a comprehensive, beginner-friendly project guide as a PDF.

Teaches the whole project from the ground up: the biology, the data, the maths
(normal distribution -> Gaussian Processes -> Bayesian optimization), the software,
how to run it, and how it was built with Claude Code.

    python scripts/make_guide_pdf.py     ->  docs/MPP_Optimizer_Guide.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, ListFlowable, ListItem, PageBreak, Paragraph,
    Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGS = DOCS / "_figs"
DOCS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
OUT = DOCS / "MPP_Optimizer_Guide.pdf"

# ----------------------------------------------------------------- palette
INK = colors.HexColor("#18181b")
MUTED = colors.HexColor("#52525b")
ACCENT = colors.HexColor("#4f46e5")
ACCENT2 = colors.HexColor("#0ea5e9")
BORDER = colors.HexColor("#e4e4e7")
CARD = colors.HexColor("#f4f4f5")
GOOD = colors.HexColor("#16a34a")
RED = colors.HexColor("#dc2626")

# ----------------------------------------------------------------- fonts
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DJ", str(_ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-B", str(_ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-I", str(_ttf / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DJ-M", str(_ttf / "DejaVuSansMono.ttf")))
registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I", boldItalic="DJ-B")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#a1a1aa",
    "axes.labelcolor": "#18181b", "text.color": "#18181b", "xtick.color": "#52525b",
    "ytick.color": "#52525b", "axes.linewidth": 0.9, "figure.dpi": 150,
})

# =================================================================== figures
def _finish(fig, name):
    path = FIGS / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def fig_normal():
    x = np.linspace(-4, 4, 500)
    pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    a = ax[0]
    a.plot(x, pdf, color=ACCENT.hexval() if False else "#4f46e5", lw=2)
    for lo, hi, c, lab in [(-1, 1, "#4f46e5", "68%"), (-2, -1, "#818cf8", "95%"), (1, 2, "#818cf8", None),
                           (-3, -2, "#c7d2fe", "99.7%"), (2, 3, "#c7d2fe", None)]:
        m = (x >= lo) & (x <= hi)
        a.fill_between(x[m], pdf[m], color=c, alpha=0.55)
    a.axvline(0, color="#18181b", lw=1, ls="--")
    a.text(0, 0.02, " μ (mean)", fontsize=9)
    a.annotate("", xy=(1, 0.05), xytext=(0, 0.05), arrowprops=dict(arrowstyle="<->", color="#18181b"))
    a.text(0.5, 0.065, "σ", ha="center", fontsize=10)
    a.set_title("One bell curve: 68–95–99.7 rule", fontsize=11)
    a.set_yticks([]); a.set_xlabel("value")
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
    b = ax[1]
    for sig, c in [(0.6, "#0ea5e9"), (1.0, "#4f46e5"), (1.8, "#a855f7")]:
        b.plot(x, np.exp(-x**2 / (2 * sig**2)) / (sig * np.sqrt(2 * np.pi)), color=c, lw=2, label=f"σ = {sig}")
    b.set_title("Bigger σ = more spread", fontsize=11)
    b.set_yticks([]); b.set_xlabel("value"); b.legend(frameon=False, fontsize=9)
    for s in ("top", "right", "left"): b.spines[s].set_visible(False)
    return _finish(fig, "normal.png")


def fig_gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    rng = np.random.default_rng(3)
    f = lambda x: np.sin(1.6 * x) + 0.3 * x
    Xtr = np.array([-3.5, -2.0, -0.5, 1.0, 2.5]).reshape(-1, 1)
    ytr = f(Xtr).ravel() + rng.normal(0, 0.06, len(Xtr))
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01),
        normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(Xtr, ytr)
    xs = np.linspace(-4.2, 4.2, 300).reshape(-1, 1)
    mu, sd = gp.predict(xs, return_std=True)
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    ax.plot(xs, f(xs), color="#a1a1aa", ls="--", lw=1.5, label="true hidden function")
    ax.fill_between(xs.ravel(), mu - 2 * sd, mu + 2 * sd, color="#c7d2fe", alpha=0.7, label="model uncertainty (±2σ)")
    ax.plot(xs, mu, color="#4f46e5", lw=2, label="model's best guess (mean)")
    ax.scatter(Xtr, ytr, color="#18181b", zorder=5, s=45, label="experiments done")
    ax.set_title("A Gaussian Process: a prediction AND its uncertainty everywhere", fontsize=11)
    ax.set_xlabel("a formulation setting (input)"); ax.set_ylabel("outcome")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.annotate("narrow where we\nhave data", xy=(-0.5, f(-0.5)), xytext=(-1.2, 1.7),
                fontsize=8, color="#16a34a", arrowprops=dict(arrowstyle="->", color="#16a34a"))
    ax.annotate("wide where we\ndon't", xy=(3.9, gp.predict([[3.9]])[0]), xytext=(2.4, -1.9),
                fontsize=8, color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626"))
    return _finish(fig, "gp.png")


def fig_ei():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from scipy.stats import norm
    rng = np.random.default_rng(1)
    f = lambda x: -(np.sin(1.3 * x) + 0.25 * x)
    Xtr = np.array([-3.2, -1.5, 0.2, 2.0, 3.4]).reshape(-1, 1)
    ytr = f(Xtr).ravel() + rng.normal(0, 0.05, len(Xtr))
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01),
                                  normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(Xtr, ytr)
    xs = np.linspace(-4, 4.2, 300).reshape(-1, 1)
    mu, sd = gp.predict(xs, return_std=True)
    best = ytr.max()
    z = (mu - best) / np.maximum(sd, 1e-9)
    ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
    ei = np.maximum(ei, 0)
    fig, ax = plt.subplots(2, 1, figsize=(9.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    a = ax[0]
    a.fill_between(xs.ravel(), mu - 2 * sd, mu + 2 * sd, color="#c7d2fe", alpha=0.7)
    a.plot(xs, mu, color="#4f46e5", lw=2, label="model mean")
    a.scatter(Xtr, ytr, color="#18181b", zorder=5, s=40, label="done")
    a.axhline(best, color="#16a34a", ls="--", lw=1.2, label="best so far")
    a.set_ylabel("outcome (higher better)"); a.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=3)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    b = ax[1]
    b.fill_between(xs.ravel(), 0, ei, color="#4f46e5", alpha=0.35)
    b.plot(xs, ei, color="#4f46e5", lw=1.8)
    star = xs.ravel()[int(np.argmax(ei))]
    b.axvline(star, color="#dc2626", lw=1.4)
    b.text(star, b.get_ylim()[1] * 0.6, "  try here next", color="#dc2626", fontsize=9)
    b.set_ylabel("Expected\nImprovement"); b.set_xlabel("a formulation setting (input)"); b.set_yticks([])
    for s in ("top", "right", "left"): b.spines[s].set_visible(False)
    return _finish(fig, "ei.png")


def fig_pareto():
    rng = np.random.default_rng(7)
    o1 = rng.uniform(0.1, 1, 40); o2 = rng.uniform(0.1, 1, 40)
    o2 = np.clip(o2 + 0.25 * (1 - o1) + rng.normal(0, 0.05, 40), 0, 1.05)
    pts = np.column_stack([o1, o2])
    nd = np.ones(len(pts), bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and np.any(pts[j] > pts[i]):
                nd[i] = False; break
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(o1[~nd], o2[~nd], color="#a1a1aa", s=35, label="dominated (something beats it)")
    front = pts[nd][np.argsort(pts[nd, 0])]
    ax.plot(front[:, 0], front[:, 1], color="#dc2626", lw=1.3, ls="--", zorder=1)
    ax.scatter(o1[nd], o2[nd], color="#dc2626", s=60, zorder=3, label="Pareto-optimal (best trade-offs)")
    ax.annotate("better", xy=(0.95, 0.98), xytext=(0.62, 0.62), fontsize=10, color="#16a34a",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.6))
    ax.set_xlabel("Objective 1  (e.g. mucus penetration →)")
    ax.set_ylabel("Objective 2  (e.g. cargo retention →)")
    ax.set_title("Pareto front: the best possible compromises", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    return _finish(fig, "pareto.png")


def fig_msd():
    t = np.linspace(0.2, 12, 200)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for a, c, lab in [(0.5, "#dc2626", "α = 0.5  subdiffusive (trapped)"),
                      (1.0, "#4f46e5", "α = 1.0  Brownian (free)"),
                      (1.5, "#16a34a", "α = 1.5  superdiffusive (directed)")]:
        ax.plot(t, t**a, color=c, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time  (log scale)"); ax.set_ylabel("MSD — how far it has wandered  (log scale)")
    ax.set_title("The exponent α is the slope on a log–log plot", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    return _finish(fig, "msd.png")


def fig_loop():
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    steps = [("1. Define\ncampaign", 5, 9), ("2. Suggest\nformulations", 9, 6),
             ("3. Run in\nthe lab", 7.2, 2), ("4. Upload\nresults", 2.8, 2), ("5. Model\nlearns", 1, 6)]
    cx, cy = [], []
    for txt, x, y in steps:
        ax.add_patch(plt.Rectangle((x - 1.15, y - 0.8), 2.3, 1.6, facecolor="#eef2ff",
                                   edgecolor="#4f46e5", lw=1.4, zorder=2, joinstyle="round"))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9.5, zorder=3)
        cx.append(x); cy.append(y)
    for i in range(len(steps)):
        j = (i + 1) % len(steps)
        ax.add_patch(FancyArrowPatch((cx[i], cy[i]), (cx[j], cy[j]), connectionstyle="arc3,rad=0.22",
                     arrowstyle="-|>", mutation_scale=16, color="#a1a1aa", lw=1.4,
                     shrinkA=26, shrinkB=26, zorder=1))
    ax.text(5, 5.4, "active-learning\nloop", ha="center", va="center", fontsize=10, color="#52525b", style="italic")
    return _finish(fig, "loop.png")


def fig_liposome():
    fig, ax = plt.subplots(figsize=(5.4, 4.0)); ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5)
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#4f46e5", lw=16, alpha=0.35))  # bilayer
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#4f46e5", lw=1))
    ax.add_patch(Circle((0, 0), 0.78, facecolor="#eff6ff", edgecolor="#93c5fd", lw=1))
    ax.text(0, 0, "drug\ncargo", ha="center", va="center", fontsize=9, color="#2563eb")
    for ang in np.linspace(0, 2 * np.pi, 22, endpoint=False):  # PEG brush
        x, y = np.cos(ang), np.sin(ang)
        ax.plot([x * 1.06, x * 1.35], [y * 1.06, y * 1.35], color="#0ea5e9", lw=1.2)
    ax.annotate("PEG 'stealth' coat", xy=(1.28, 0.35), xytext=(1.4, 1.25), fontsize=8.5, color="#0ea5e9",
                ha="right", arrowprops=dict(arrowstyle="->", color="#0ea5e9"))
    ax.annotate("lipid bilayer wall", xy=(-0.95, 0.4), xytext=(-1.6, 1.25), fontsize=8.5, color="#4f46e5",
                arrowprops=dict(arrowstyle="->", color="#4f46e5"))
    ax.set_title("A liposome (a fat bubble ~100 nm across)", fontsize=10)
    return _finish(fig, "liposome.png")


FIGURES = {
    "normal": fig_normal, "gp": fig_gp, "ei": fig_ei, "pareto": fig_pareto,
    "msd": fig_msd, "loop": fig_loop, "liposome": fig_liposome,
}
print("Rendering figures…")
FIG_PATHS = {k: fn() for k, fn in FIGURES.items()}

# =================================================================== styles
ss = getSampleStyleSheet()
def _st(name, **kw):
    base = dict(fontName="DJ", textColor=INK, leading=15, fontSize=10.5)
    base.update(kw)
    return ParagraphStyle(name, **base)

S = {
    "title": _st("t", fontName="DJ-B", fontSize=26, leading=30, textColor=INK, alignment=TA_LEFT),
    "subtitle": _st("st", fontSize=13, leading=18, textColor=MUTED),
    "h1": _st("h1", fontName="DJ-B", fontSize=17, leading=21, textColor=INK, spaceBefore=16, spaceAfter=6),
    "h2": _st("h2", fontName="DJ-B", fontSize=13, leading=17, textColor=ACCENT, spaceBefore=12, spaceAfter=4),
    "body": _st("b", alignment=TA_JUSTIFY, spaceAfter=6),
    "bul": _st("bul", alignment=TA_LEFT, leftIndent=4, spaceAfter=3),
    "cap": _st("cap", fontSize=8.6, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8),
    "code": _st("code", fontName="DJ-M", fontSize=8.8, leading=12, textColor=INK, backColor=CARD),
    "callh": _st("callh", fontName="DJ-B", fontSize=10, leading=13, textColor=INK),
    "toc": _st("toc", fontSize=10.5, leading=17),
    "small": _st("sm", fontSize=8.6, leading=11, textColor=MUTED),
}

story = []
def P(t, s="body"): story.append(Paragraph(t, S[s]))
def H1(t): story.append(Paragraph(t, S["h1"]))
def H2(t): story.append(Paragraph(t, S["h2"]))
def SP(h=6): story.append(Spacer(1, h))
def CAP(t): story.append(Paragraph(t, S["cap"]))
def RULE(): story.append(HRFlowable(width="100%", color=BORDER, thickness=0.8, spaceBefore=8, spaceAfter=8))

def BUL(items):
    story.append(ListFlowable([ListItem(Paragraph(t, S["bul"]), leftIndent=12, value="•") for t in items],
                              bulletType="bullet", start="•", leftIndent=10))
    SP(4)

def IMG(key, width_cm=15):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(FIG_PATHS[key]).getSize()
    w = width_cm * cm
    story.append(Image(FIG_PATHS[key], width=w, height=w * ih / iw))

def CODE(text):
    story.append(Preformatted(text, S["code"]))
    SP(6)

def CALLOUT(title, body, bg=CARD, bar=ACCENT):
    inner = [Paragraph(f"<b>{title}</b>", S["callh"]), Spacer(1, 3), Paragraph(body, S["body"])]
    t = Table([[inner]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
    ]))
    story.append(t); SP(8)

def TABLE(rows, widths, header=True):
    t = Table(rows, colWidths=[w * cm for w in widths])
    st = [("FONT", (0, 0), (-1, -1), "DJ", 9), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
          ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
          ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONT", (0, 0), (-1, 0), "DJ-B", 9)]
    t.setStyle(TableStyle(st)); story.append(t); SP(8)

def cell(t):
    return Paragraph(t, S["small"])

# =================================================================== content
# ---- cover
SP(40)
story.append(HRFlowable(width="42%", color=ACCENT, thickness=3, spaceAfter=18, hAlign="LEFT"))
P("MPP Optimizer", "title")
SP(4)
P("A complete, beginner-friendly guide to the machine-learning system for designing "
  "muco-penetrating liposomal nanoparticles — the science, the algorithm from scratch, "
  "the software, and how it was built.", "subtitle")
SP(26)
TABLE([
    [cell("<b>Project</b>"), cell("ML-driven design of muco-penetrating liposomal nanoparticles (MPPs)")],
    [cell("<b>What it does</b>"), cell("Learns from experiments and proposes the next formulations to test (Bayesian optimization)")],
    [cell("<b>Built with</b>"), cell("Python, Streamlit, scikit-learn — and Claude Code (model: Claude Opus 4.8)")],
    [cell("<b>Code</b>"), cell("github.com/melaniemenezes/mpp-optimizer  ·  github.com/JitheshVijay/mpp-optimizer")],
], widths=[3.2, 13.2], header=False)
SP(10)
P("Read this front to back and you will understand every idea the system uses — including the "
  "normal distribution — without needing any prior maths background.", "small")
story.append(PageBreak())

# ---- contents
H1("Contents")
toc = [
    "Part 1 — The big picture", "Part 2 — The biology, simply", "Part 3 — The data",
    "Part 4 — The maths you actually need (from zero)", "Part 5 — The model: Gaussian Processes",
    "Part 6 — The algorithm: Bayesian optimization", "Part 7 — Understanding the results",
    "Part 8 — The software", "Part 9 — Running it yourself",
    "Part 10 — How this was built: using Claude Code", "Glossary", "Appendix — the configured study and FAQ",
]
for i, t in enumerate(toc, 1):
    story.append(Paragraph(f"{i}.&nbsp;&nbsp;{t}", S["toc"]))
story.append(PageBreak())

# ---- Part 1
H1("Part 1 — The big picture")
P("Some medicines work best if they are delivered as tiny carriers called <b>nanoparticles</b> — "
  "far smaller than a cell. This project is about a particular kind: <b>liposomes</b>, which are "
  "microscopic bubbles made of fat molecules (lipids) that can carry a drug inside them.")
P("The challenge is that many drugs need to get <b>through mucus</b> — the sticky protective gel "
  "lining our airways and gut. Mucus is a net that traps most particles. A carrier that can slip "
  "through it is called <b>muco-penetrating</b> (hence <b>MPP</b>: muco-penetrating particle).")
P("Whether a liposome can penetrate mucus depends on its <b>recipe</b> — which lipids it is made of "
  "and in what proportions — plus a couple of physical properties. There are millions of possible "
  "recipes, and testing each one in the lab is slow and expensive. So we built a piece of software "
  "that uses <b>machine learning</b> to be smart about which recipes to try next.")
CALLOUT("In one sentence",
        "The app learns from the experiments you have already done and predicts which new formulations "
        "are most worth testing — so you reach a good design in far fewer lab experiments.")

# ---- Part 2
story.append(PageBreak())
H1("Part 2 — The biology, simply")
H2("Mucus: the body's sticky net")
P("Mucus is a mesh of long, sticky molecules. Small, slippery things pass through the gaps; large "
  "or sticky things get caught. In diseased lungs the mesh is even denser, so getting a drug through "
  "is harder. Our goal is to design carriers that behave like the slippery things.")
H2("What is a liposome?")
P("Picture a soap bubble, but its skin is made of two layers of fat molecules (a <b>bilayer</b>). "
  "That hollow bubble can hold a drug inside. Think of it like building a ball out of LEGO bricks, "
  "where each brick is a different lipid:")
BUL([
    "a <b>structural</b> lipid forms the wall of the bubble;",
    "<b>cholesterol</b> stiffens the wall so it does not leak;",
    "a <b>PEG-lipid</b> adds a slippery ‘stealth’ coat that stops mucus sticking;",
    "<b>charged</b> lipids (positive or negative) tune the surface charge.",
])
IMG("liposome", 8.5)
CAP("A liposome: a fat bubble with a drug inside and a slippery PEG coat outside.")
H2("What makes a particle muco-penetrating?")
P("Three things help a particle slip through mucus: it should be <b>small</b> enough to fit the mesh, "
  "have a surface charge <b>close to zero</b> (so it is not sticky — we call this ‘muco-inert’), and "
  "usually carry a <b>PEG coat</b>. The recipe controls all of these.")
H2("The recipe = the composition")
P("A <b>composition</b> is just the list of proportions of each lipid, which always add up to 100%. "
  "For example: 55% structural, 35% cholesterol, 7% PEG-lipid, 3% charged. Change those numbers and "
  "you get a different particle that behaves differently in mucus.")

# ---- Part 3
story.append(PageBreak())
H1("Part 3 — The data")
P("A machine-learning model needs a tidy table: each row is one formulation, with some columns we "
  "control (<b>inputs</b>) and some columns we measure (<b>outputs</b>).")
H2("Inputs — what describes a formulation")
BUL([
    "the proportions of each lipid (e.g. DDAB, DSPG, HSPC, Cholesterol, mPEG);",
    "two measured physical properties used as inputs: <b>liposome size</b> (in nanometres) and "
    "<b>zeta potential</b> (the surface charge, in millivolts).",
])
H2("Outputs — how well it moves in mucus")
P("To measure movement, scientists film individual particles in mucin (the main mucus protein) and "
  "track their paths. From those paths they compute a few numbers. The core quantity is the "
  "<b>MSD</b> (mean squared displacement): on average, how far has a particle wandered from its "
  "start after a given time.")
TABLE([
    [cell("<b>Readout</b>"), cell("<b>What it means (plain English)</b>")],
    [cell("D (>5 s)"), cell("Overall speed of spreading, from long tracks. Higher = faster = better penetration.")],
    [cell("D₁ (1 s)"), cell("Speed measured over a short 1-second window (simple ‘free-motion’ model).")],
    [cell("Dα (10 s)"), cell("Speed over a longer 10-second window (allows for hindered motion).")],
    [cell("α (alpha)"), cell("The ‘mobility fingerprint’. ≈1 = free; below 1 = trapped/hindered; above 1 = directed.")],
    [cell("net-to-path"), cell("Straight-line distance ÷ total wiggly distance. Near 1 = travels; near 0 = wiggles in place.")],
], widths=[2.6, 13.8])
H2("The exponent α, visualised")
P("If you plot MSD against time on a log–log chart, you get a straight line whose <b>slope is α</b>. "
  "A shallow slope (α&lt;1) means the particle keeps getting stuck; a slope of 1 is ordinary free "
  "diffusion; a steep slope (α&gt;1) means it is being carried in a direction.")
IMG("msd", 11)
CAP("α is simply the steepness of the MSD line — the single best number for telling ‘stuck’ from ‘mobile’.")

# ---- Part 4
story.append(PageBreak())
H1("Part 4 — The maths you actually need (from zero)")
P("You only need a few ideas. We build them up gently; nothing here assumes prior maths.")
H2("Average (mean) and spread (standard deviation)")
P("The <b>mean</b> (written μ, the Greek letter ‘mu’) is just the average of some numbers. The "
  "<b>standard deviation</b> (written σ, ‘sigma’) measures how spread out they are: a small σ means "
  "the numbers huddle close to the average; a large σ means they are scattered widely.")
H2("The normal distribution — the bell curve")
P("Measure something many times with a bit of random error — a particle’s size, a lab reading — and "
  "the values usually pile up in a symmetric <b>bell shape</b> called the <b>normal</b> (or "
  "<b>Gaussian</b>) distribution. It is the most important shape in statistics, and our whole "
  "algorithm leans on it.")
P("The bell is described by just two numbers you already met: its centre <b>μ</b> (where the peak "
  "is) and its width <b>σ</b> (how fat the bell is). A famous rule of thumb, the "
  "<b>68–95–99.7 rule</b>, says:")
BUL([
    "about <b>68%</b> of values fall within 1σ of the mean (one step either side of the peak);",
    "about <b>95%</b> fall within 2σ;",
    "about <b>99.7%</b> fall within 3σ.",
])
IMG("normal", 15)
CAP("Left: the 68–95–99.7 rule. Right: a bigger σ makes a wider, flatter bell.")
P("The exact formula (you do not need to memorise it) is:")
CODE("height(x) = 1 / (σ·√(2π))  ·  e^( -(x - μ)² / (2σ²) )")
P("Read it as: the curve is tallest at x = μ and falls away smoothly the further x gets from μ, at a "
  "rate set by σ. That is all.")
H2("Probability = area under the curve")
P("The <b>area</b> under a chunk of the bell tells you how <b>likely</b> a value is to land in that "
  "range (the whole area is 100%). The running total of area from the far left up to some point is "
  "so useful it has a name — the <b>cumulative</b> function, written <b>Φ</b> (‘Phi’). "
  "Φ(z) answers: ‘what fraction of the bell lies to the left of here?’ We will use Φ in the "
  "algorithm to turn ‘how far above a threshold’ into ‘how probable’.")
CALLOUT("Why this matters",
        "The one big idea to carry forward: instead of guessing a single number, we describe what we "
        "know as a <b>bell curve</b> — a best guess (μ) plus an honest uncertainty (σ). Everything the "
        "algorithm does is built on comparing these bells.")

# ---- Part 5
story.append(PageBreak())
H1("Part 5 — The model: Gaussian Processes")
P("An ordinary model fits a single line through your data. But with only a handful of expensive "
  "experiments, a single line is over-confident — it says nothing about how unsure it is. We want a "
  "model that also tells us <b>where it is guessing</b>.")
P("A <b>Gaussian Process</b> (GP) does exactly that. At every possible formulation it gives you a "
  "bell curve: a best guess (μ) <b>and</b> an uncertainty (σ). Near formulations you have already "
  "tested, it is confident (narrow bell). Far from any data, it admits it does not know (wide bell).")
IMG("gp", 15)
CAP("The GP’s best guess (line) with its uncertainty band. The band is tight near real experiments "
    "and widens where we have none — an honest ‘I’m not sure here’.")
H2("The one knob: the lengthscale")
P("A GP has a setting called the <b>lengthscale</b> for each input, which captures ‘how far does a "
  "change in this input reach before the outcome changes’. A <b>short</b> lengthscale means the "
  "outcome is very sensitive to that input; a <b>long</b> one means that input barely matters. Later "
  "we read these lengthscales directly to say <b>which ingredients drive the result</b>.")
CALLOUT("Why a GP here",
        "Lab experiments are scarce and costly, so we cannot afford a data-hungry model. A GP squeezes "
        "the most out of a few points and, crucially, reports its own uncertainty — which is the fuel "
        "the next part needs.")

# ---- Part 6
story.append(PageBreak())
H1("Part 6 — The algorithm: Bayesian optimization")
P("Now we can state the method. We want the best formulation, each experiment is expensive, and the "
  "search space is huge. <b>Bayesian optimization</b> is the strategy of using the GP model to spend "
  "each experiment as wisely as possible.")
H2("The loop")
BUL([
    "Fit the GP to every experiment done so far.",
    "Use it to <b>score</b> every untested formulation by how promising it is.",
    "Suggest the top few; run them in the lab.",
    "Add the results and repeat. Each round the model gets sharper.",
])
IMG("loop", 12)
CAP("The active-learning loop the whole app is built around.")
H2("Exploration vs exploitation")
P("A good ‘promise’ score must balance two urges: <b>exploit</b> (try formulations the model already "
  "thinks are good) and <b>explore</b> (try uncertain regions that might hide something better). Lean "
  "too far either way and you waste experiments.")
H2("Expected Improvement — the promise score")
P("The score we use is <b>Expected Improvement</b> (EI). In words: <i>for a candidate formulation, "
  "how much better than our current best do we expect it to be — averaged over the model’s "
  "uncertainty?</i> Because the model’s belief is a bell curve, this is a small calculation on that "
  "bell, using the very Φ (area) function from Part 4:")
CODE("EI  =  (how far the mean beats the best) × Φ(z)   +   (uncertainty σ) × φ(z)")
P("The first term rewards a high predicted mean (exploitation); the second rewards high uncertainty "
  "(exploration). EI is large where the formulation is either predicted to be good, or uncertain "
  "enough that it <i>could</i> be good. We simply pick the formulations with the biggest EI.")
IMG("ei", 15)
CAP("Top: the model and the current best. Bottom: the EI score. The red line marks the most "
    "promising place to experiment next — note it favours a spot that is both high and uncertain.")
H2("Many goals at once: the Pareto front")
P("We rarely have a single goal. We might want high penetration <b>and</b> good cargo retention "
  "<b>and</b> a target size — and these fight each other. There is no single winner; instead there is "
  "a set of <b>best compromises</b>, called the <b>Pareto front</b>: formulations where you cannot "
  "improve one goal without sacrificing another.")
IMG("pareto", 11)
CAP("Each dot is a formulation. Red dots are the best available trade-offs; grey dots are beaten by "
    "some red dot on every goal.")
P("To search several goals at once, the app uses a method called <b>ParEGO</b>: each round it rolls "
  "random ‘importance weights’ for the goals, blends them into a single temporary score, and finds "
  "the best formulation for that blend. Over many rounds the random blends trace out the whole Pareto "
  "front, giving a <b>spread</b> of good compromises rather than one narrow answer.")
H2("Respecting constraints")
P("Some requirements are hard limits, not goals — for example ‘surface charge must stay near zero’ "
  "or ‘size below 200 nm’. For each such limit the app builds a small GP and uses the bell curve to "
  "compute the <b>probability the limit is satisfied</b>, then multiplies the EI score by it. "
  "Formulations likely to break a rule are quietly pushed down the list.")
H2("The cold start")
P("With no data at all, there is nothing to learn from, so the first batch is chosen by a "
  "<b>space-filling design</b> (a tidy way to spread points evenly across all recipes, called a "
  "Latin hypercube). This seeds the first 96-well plate broadly; the Bayesian smarts switch on once "
  "enough results are in.")

# ---- Part 7
story.append(PageBreak())
H1("Part 7 — Understanding the results")
P("A good tool does not just give an answer — it explains itself. The app offers three views.")
H2("Which inputs matter (sensitivity)")
P("Reading the GP’s lengthscales (Part 5) tells us how strongly each ingredient or property moves a "
  "chosen outcome. This is shown as a simple bar chart of <b>importances</b> — at a glance you see, "
  "say, that surface charge and PEG dominate mucus mobility.")
H2("Partial-dependence curves")
P("These answer ‘as I dial this one input up, what happens to the outcome (holding the others "
  "steady)?’ — a plain line you can read off, e.g. ‘more PEG helps up to a point, then plateaus’.")
H2("Predicted vs observed (a trust check)")
P("To test whether the model can be believed, we hide each experiment in turn, predict it from the "
  "others, and plot predicted against actual. Points hugging the diagonal mean the model generalises "
  "well; scattered points mean ‘collect more data before trusting it’.")

# ---- Part 8
story.append(PageBreak())
H1("Part 8 — The software")
P("The whole thing is a single <b>Streamlit</b> web app (Streamlit turns Python scripts into web "
  "pages) backed by a local database file. There is no cloud service and no heavy AI hardware needed.")
H2("The five pages")
TABLE([
    [cell("<b>Page</b>"), cell("<b>What you do there</b>")],
    [cell("Campaign Setup"), cell("Choose the lipids, ranges, input features, objectives and constraints.")],
    [cell("Suggest Experiments"), cell("Get the next batch of formulations; export a 96-well worklist CSV.")],
    [cell("Upload Results"), cell("Type in the measured readouts and attach raw files (Excel, PDF, images).")],
    [cell("Dataset Browser"), cell("View, filter and export the whole dataset; preview attachments.")],
    [cell("Model &amp; Insights"), cell("Pareto front, recommended recipes, and the three explanation views.")],
], widths=[3.6, 12.8])
H2("Where the data lives")
P("Everything is stored locally in one small database file (SQLite) plus a folder of uploaded files. "
  "You can export the full table to Excel or CSV at any time.")
H2("The engine, and why it is lightweight")
P("The Bayesian optimization runs on <b>scikit-learn</b> Gaussian Processes (plus SciPy for the bell-"
  "curve maths). This avoids heavy deep-learning libraries, so it installs in seconds and runs on an "
  "ordinary laptop with no graphics card.")

# ---- Part 9
story.append(PageBreak())
H1("Part 9 — Running it yourself")
P("Three steps in a terminal, from the project folder:")
CODE("# 1) set up a private environment and install everything\n"
     "python3 -m venv .venv\n"
     ".venv/bin/python -m pip install -r requirements.txt\n\n"
     "# 2) (optional) fill it with synthetic demo data so you can explore immediately\n"
     ".venv/bin/python scripts/seed_diffusion_demo.py\n\n"
     "# 3) launch the app (opens in your web browser)\n"
     ".venv/bin/python -m streamlit run app.py")
P("The app opens at a local web address (usually http://localhost:8501). On Windows, use "
  "<font face='DJ-M'>.venv\\Scripts\\python</font> instead of <font face='DJ-M'>.venv/bin/python</font>.")
P("The code lives in two GitHub repositories (identical content):")
BUL(["github.com/melaniemenezes/mpp-optimizer", "github.com/JitheshVijay/mpp-optimizer"])

# ---- Part 10
story.append(PageBreak())
H1("Part 10 — How this was built: using Claude Code")
P("This entire application — the code, the tests, this very PDF — was built by working with "
  "<b>Claude Code</b>, an AI coding assistant, in plain conversation. This section explains what that "
  "is and how to use it, in case you want to keep developing the project.")
H2("What is Claude Code?")
P("Claude Code is an <b>AI agent</b> that runs in your terminal or code editor. You describe what you "
  "want in ordinary English; it reads and writes files, runs commands, fixes errors, runs tests, and "
  "can even manage GitHub — pausing for your approval on important actions. It is like pairing with a "
  "tireless programmer who explains its reasoning.")
H2("What is ‘the model’?")
P("Behind Claude Code is a <b>large language model (LLM)</b> — here, <b>Claude Opus 4.8</b>, made by "
  "Anthropic. An LLM is a system trained on enormous amounts of text and code so that, given a "
  "request, it can predict and produce a sensible, detailed response — writing code, explaining "
  "concepts, or planning a task. It is the same family of technology as a chatbot, but wired up so it "
  "can actually act on your computer (with permission) rather than only chat.")
H2("How we worked with it — the workflow")
BUL([
    "<b>Describe the goal</b> in plain language (‘build a tool that suggests nanoparticle recipes’).",
    "<b>Plan first</b>: Claude Code can enter a ‘plan mode’ where it researches and proposes an "
    "approach for you to approve before writing any code.",
    "<b>It builds and tests</b>: it writes the files, runs the test suite, and launches the app to "
    "check it actually works — reporting what passed or failed.",
    "<b>You review and steer</b>: you read its summaries, answer its questions, and ask for changes.",
    "<b>Version control</b>: it commits the work to Git and pushes to GitHub when you ask.",
])
H2("Tips for working with it")
BUL([
    "Be specific about the goal and any constraints; vague requests get vague results.",
    "Let it use <b>plan mode</b> for anything non-trivial, and read the plan before approving.",
    "Ask it to run the tests and the app — insist on evidence that a change works.",
    "Review important or irreversible actions (deleting files, publishing, pushing) before approving.",
    "Keep secrets (passwords, tokens) out of the chat where you can; rotate any you do share.",
])
CALLOUT("Getting started with Claude Code",
        "Install it from Anthropic (available as a terminal command and as editor extensions for "
        "VS Code and JetBrains, plus a web version). Open it in your project folder and start typing "
        "what you want. Type ‘/’ to see built-in commands. That is genuinely all it takes to begin.")

# ---- Glossary
story.append(PageBreak())
H1("Glossary")
gl = [
    ("Liposome", "A tiny fat bubble (lipid bilayer) that can carry a drug."),
    ("MPP", "Muco-penetrating particle — a carrier engineered to slip through mucus."),
    ("Composition", "The proportions of each lipid in a formulation; they sum to 100%."),
    ("Zeta potential", "A particle’s surface charge; near zero is ‘muco-inert’ (non-sticky)."),
    ("MSD", "Mean squared displacement — how far a particle wanders over time."),
    ("α (alpha)", "The MSD slope; ≈1 free, <1 trapped, >1 directed."),
    ("Mean (μ)", "The average of some numbers."),
    ("Std. dev. (σ)", "How spread out the numbers are."),
    ("Normal distribution", "The bell curve; described by μ and σ."),
    ("Gaussian Process (GP)", "A model giving a prediction and an uncertainty at every point."),
    ("Lengthscale", "A GP setting showing how much an input matters."),
    ("Bayesian optimization", "Using the model to choose the most useful next experiment."),
    ("Expected Improvement", "A score for how promising a candidate is."),
    ("Pareto front", "The set of best possible trade-offs among competing goals."),
    ("Constraint", "A hard requirement (e.g. size limit) the search must respect."),
    ("LLM", "Large language model — the AI behind Claude Code (Claude Opus 4.8)."),
]
TABLE([[cell("<b>Term</b>"), cell("<b>Meaning</b>")]] + [[cell(f"<b>{t}</b>"), cell(d)] for t, d in gl],
      widths=[4.2, 12.2])

# ---- Appendix
story.append(PageBreak())
H1("Appendix — the configured study and FAQ")
H2("The configured diffusion study")
P("The app ships with a ready-made campaign matching the study design agreed with the supervisor:")
BUL([
    "<b>Inputs (7):</b> DDAB, DSPG, HSPC (structural), Cholesterol and mPEG proportions, plus "
    "liposome size and zeta potential.",
    "<b>Outputs (5):</b> D (&gt;5 s), D₁ (1 s), Dα (10 s), α, and net-to-path.",
])
P("On the Model &amp; Insights page you can pick any output and see which inputs drive it, how it "
  "responds, and how well it is predicted — the ‘characterise and distinguish mobility’ goal.")
H2("A few common questions")
CALLOUT("Is this a Bayesian network?",
        "No — it is Bayesian <b>optimization</b> (choosing the best next experiment). A Bayesian "
        "<b>network</b> (a diagram of cause-and-effect probabilities) is a different, complementary "
        "tool that could be added later.")
CALLOUT("Should size and zeta be inputs or outputs?",
        "They are used as inputs here to <b>characterise</b> mobility, as specified. Since they are "
        "themselves consequences of the recipe, a future version could also predict them from the "
        "composition to help <b>design</b> brand-new formulations.")
CALLOUT("Where is my data — is it private?",
        "Entirely local: one database file and a folder of your uploaded files. Nothing is sent to "
        "any external service.")

# =================================================================== build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DJ", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, "MPP Optimizer — Project Guide")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.setStrokeColor(BORDER)
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="MPP Optimizer — Project Guide", author="MPP Optimizer project")
print("Building PDF…")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
