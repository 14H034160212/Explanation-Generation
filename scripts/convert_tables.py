import re

def markdown_to_latex_table(md_lines):
    if not md_lines: return ""
    
    # Process headers
    header_line = md_lines[0].strip('| ')
    headers = [h.strip() for h in header_line.split('|')]
    
    # Process alignment (assuming mostly left/center/right depending on content)
    cols = ['c'] * len(headers)
    cols[0] = 'l' # Usually model name
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\resizebox{\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{" + "|".join(cols) + "}\n"
    latex += "\\toprule\n"
    latex += " & ".join(headers).replace('%', '\\%').replace('↑', '$\\uparrow$').replace('↓', '$\\downarrow$') + " \\\\\n"
    latex += "\\midrule\n"
    
    # Process rows
    for line in md_lines[2:]:
        if not line.strip(): continue
        row = line.strip('| ').split('|')
        row_str = " & ".join([r.strip() for r in row])
        row_str = row_str.replace('%', '\\%').replace('—', '---')
        row_str = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', row_str)
        row_str = row_str.replace('_', '\\_')
        latex += row_str + " \\\\\n"
        
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "}\n"
    latex += "\\caption{Experimental Results}\n"
    latex += "\\end{table}\n\n"
    return latex

with open('README_RL.md', 'r') as f:
    content = f.read()

# Extract sections
lines = content.split('\n')
tables = []
current_table = []
in_table = False

for line in lines:
    if line.strip().startswith('|') and not line.strip().startswith('|-'):
        in_table = True
        current_table.append(line)
    elif line.strip().startswith('|-'):
        if in_table:
            current_table.append(line)
    else:
        if in_table:
            tables.append(current_table)
            current_table = []
            in_table = False

out_tex = "\\section{Comprehensive Experimental Results}\n\n"
out_tex += "In this section, we present the full tables of experiments from our codebase, detailing performance on Cardiff, Sydney, Medicine, and Law datasets, as well as the GPT pairwise evaluations.\n\n"

for i, t in enumerate(tables):
    # Only pick evaluation tables and avoid pipeline small tables
    if len(t) > 3 and "Model" in t[0] or "Dataset" in t[0] or "Model A" in t[0]:
        out_tex += markdown_to_latex_table(t)

tikz_diagram = r"""
\section{System Architecture: Hybrid-DPO}

The Hybrid-DPO methodology explicitly fuses factual determinism (NLI) with linguistic fluency (Verifier) into a dual-signal reward. The architecture is detailed in Figure \ref{fig:hybrid_dpo}.

\begin{figure}[h]
\centering
\begin{tikzpicture}[
    node distance=1.5cm and 2cm,
    box/.style={draw, rectangle, rounded corners, align=center, minimum height=1cm, fill=blue!5},
    eval/.style={draw, rectangle, rounded corners, align=center, minimum height=1cm, fill=orange!10},
    alignbox/.style={draw, rectangle, rounded corners, align=center, minimum height=1cm, fill=green!10},
    arrow/.style={->, thick}
]

% Nodes
\node[box] (input) {Context\\(Question + Ground Truth)};
\node[box, right=of input] (policy) {Explanation Policy Model\\(SFT Generator)};
\node[box, below=of policy] (cands) {Candidate Explanations\\(N samples)};

\node[eval, below left=of cands] (logic) {Logical Evaluator\\(DeBERTa-v3 NLI)};
\node[eval, below right=of cands] (fluency) {Linguistic Verifier\\(Alpaca-7B)};

\node[alignbox, below=3cm of cands] (rank) {Dual-Signal Preference Ranking\\(Hybrid Score = $0.5 \times$ NLI $+ 0.5 \times$ Verifier)};
\node[alignbox, below=of rank] (dpo) {Direct Preference Optimization (DPO)};

% Arrows
\draw[arrow] (input) -- (policy);
\draw[arrow] (policy) -- node[right] {Sampling} (cands);
\draw[arrow] (cands) -| (logic);
\draw[arrow] (cands) -| (fluency);
\draw[arrow, dashed] (input) |- (logic);

\draw[arrow] (logic) |- node[above left] {Logic Signal (P)} (rank);
\draw[arrow] (fluency) |- node[above right] {Fluency Signal (0-5)} (rank);

\draw[arrow] (rank) -- node[right] {(Chosen, Rejected) Pairs} (dpo);
\draw[arrow, dashed, bend left=60] (dpo) to node[left] {Policy Update} (policy);

\end{tikzpicture}
\caption{The Hybrid-DPO Alignment Architecture.}
\label{fig:hybrid_dpo}
\end{figure}
"""

out_tex += tikz_diagram

with open("paper_draft/extra_results.tex", "w") as f:
    f.write(out_tex)

print("Latex generation complete! extra_results.tex created.")
