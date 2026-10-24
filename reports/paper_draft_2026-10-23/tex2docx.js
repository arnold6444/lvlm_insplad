// Convert the InsPLAD-QA LaTeX draft (a small, known LaTeX subset) into a .docx with docx-js.
// Usage: node tex2docx.js <main.tex> <out.docx> <en|ko>
const fs = require("fs");
const path = require("path");
const D = require("docx");

const [, , texPath, outPath, lang] = process.argv;
const ROOT = path.dirname(texPath);
const KO = lang === "ko";
const L = KO
  ? { abstract: "요약", references: "참고문헌", table: "표", appendix: "부록" }
  : { abstract: "Abstract", references: "References", table: "Table", appendix: "Appendix" };
const FONT = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "Malgun Gothic", cs: "Times New Roman" };
const PAGE_W = 11906, MARGIN = 1300, CONTENT_W = PAGE_W - 2 * MARGIN;
// Korean text with long unbreakable number groups looks gappy when justified; Malgun Gothic also runs tall.
const BODY_ALIGN = KO ? D.AlignmentType.LEFT : D.AlignmentType.JUSTIFIED, LINE = KO ? 250 : 276;

const stripComments = (s) => s.split(/\r?\n/).map((l) => l.replace(/(^|[^\\])%.*$/, "$1")).join("\n");
const read = (p) => stripComments(fs.readFileSync(p, "utf8"));

// ---------- brace helpers ----------
function braced(s, i) { // s[i] === '{' -> [content, indexAfter]
  if (s[i] !== "{") throw new Error("expected { at " + i + ": " + s.slice(i, i + 30));
  let depth = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === "\\") { j++; continue; }
    if (s[j] === "{") depth++;
    else if (s[j] === "}" && --depth === 0) return [s.slice(i + 1, j), j + 1];
  }
  throw new Error("unbalanced braces");
}
function splitTop(s, sep) { // split on sep outside braces, ignoring escaped chars
  const out = []; let depth = 0, cur = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "\\" && s.startsWith(sep, i) && sep === "\\\\" && depth === 0) { out.push(cur); cur = ""; i++; continue; }
    if (c === "\\") { cur += c + (s[i + 1] || ""); i++; continue; }
    if (c === "{") depth++;
    if (c === "}") depth--;
    if (depth === 0 && s.startsWith(sep, i)) { out.push(cur); cur = ""; i += sep.length - 1; continue; }
    cur += c;
  }
  out.push(cur);
  return out;
}

// ---------- bibliography ----------
function parseBib(text) {
  const entries = {};
  const re = /@(\w+)\s*\{\s*([^,\s]+)\s*,/g;
  let m;
  while ((m = re.exec(text))) {
    let i = re.lastIndex, depth = 1, j = i;
    for (; j < text.length && depth; j++) { if (text[j] === "{") depth++; else if (text[j] === "}") depth--; }
    const body = text.slice(i, j - 1), fields = {};
    const fre = /(\w+)\s*=\s*\{/g; let f;
    while ((f = fre.exec(body))) { const [v, end] = braced(body, fre.lastIndex - 1); fields[f[1].toLowerCase()] = v; fre.lastIndex = end; }
    entries[m[2]] = { type: m[1].toLowerCase(), ...fields };
  }
  return entries;
}
const ACCENT = { "'": { e: "é", a: "á", o: "ó" }, "~": { a: "ã", o: "õ", n: "ñ" }, '"': { o: "ö", u: "ü", a: "ä" } };
function bibText(s) {
  return s.replace(/\{\\(['~"])\{?(\w)\}?\}/g, (_, a, ch) => (ACCENT[a] || {})[ch] || ch)
    .replace(/\\(['~"])\{?(\w)\}?/g, (_, a, ch) => (ACCENT[a] || {})[ch] || ch)
    .replace(/\\url\{([^}]*)\}/g, "$1").replace(/--/g, "–").replace(/[{}]/g, "").trim();
}
function bibAuthors(a) {
  const names = bibText(a).split(/\s+and\s+/).map((n) => {
    if (n === "others") return "et al.";
    const [last, first] = n.split(",").map((x) => x.trim());
    return first ? `${first} ${last}` : last;
  });
  return names.join(", ").replace(/, et al\./, " et al.");
}
function bibLine(e) {
  const venue = e.booktitle || e.journal || e.publisher || "";
  const parts = [bibAuthors(e.author || ""), bibText(e.title || "")];
  if (venue) parts.push(bibText(venue));
  if (e.pages) parts.push("pp. " + bibText(e.pages));
  parts.push(e.year || "");
  if (e.doi) parts.push("doi:" + e.doi);
  if (e.note) parts.push(bibText(e.note));
  return parts.filter(Boolean).map((p) => p.replace(/\.$/, "")).join(". ") + ".";
}

// ---------- document source ----------
let src = read(texPath).replace(/\\input\{([^}]+)\}/g, (_, f) => read(path.join(ROOT, f.endsWith(".tex") ? f : f + ".tex")));
const macros = {};
src.replace(/\\(?:re)?newcommand\{\\(\w+)\}\{/g, (m, name, off) => { const [v] = braced(src, off + m.length - 1); macros[name] = v; return m; });
const titleSrc = braced(src, src.indexOf("\\title{") + 6)[0];
const authorSrc = braced(src, src.indexOf("\\author{") + 7)[0];
let body = src.slice(src.indexOf("\\begin{document}") + 16, src.lastIndexOf("\\end{document}"));
const bib = parseBib(fs.readFileSync(path.join(ROOT, "refs.bib"), "utf8"));

// ---------- pass 1: numbering of sections, tables and citations ----------
const labels = {}, citeOrder = [];
{
  let sec = 0, sub = 0, app = false, tab = 0;
  const re = /\\(section|subsection)\{|\\appendix|\\begin\{table\}|\\label\{([^}]+)\}|\\cite\{([^}]+)\}/g;
  let m, last = null;
  while ((m = re.exec(body))) {
    if (m[0] === "\\appendix") { app = true; sec = 0; continue; }
    if (m[1] === "section") { sec++; sub = 0; last = app ? String.fromCharCode(64 + sec) : String(sec); continue; }
    if (m[1] === "subsection") { sub++; last = (app ? String.fromCharCode(64 + sec) : sec) + "." + sub; continue; }
    if (m[0].startsWith("\\begin{table}")) { tab++; last = String(tab); continue; }
    if (m[2]) { labels[m[2]] = last; continue; }
    if (m[3]) m[3].split(",").map((k) => k.trim()).forEach((k) => { if (!citeOrder.includes(k)) citeOrder.push(k); });
  }
}

// ---------- inline conversion ----------
const MATH = [[/\\rightarrow/g, "→"], [/\\times/g, "×"], [/\\le\b/g, "≤"], [/\\ge\b/g, "≥"], [/\\pm/g, "±"],
  [/\\dagger/g, "†"], [/\\,/g, " "], [/\^/g, ""], [/\\Delta/g, "Δ"], [/(^|[\s\[(,])-/g, "$1−"]];
function mathText(s) { let t = s; for (const [a, b] of MATH) t = t.replace(a, b); return t.replace(/[{}]/g, ""); }
function plain(s) {
  return s.replace(/\\(['~"])\{?(\w)\}?/g, (_, a, ch) => (ACCENT[a] || {})[ch] || ch)
    .replace(/``/g, "“").replace(/''/g, "”").replace(/---/g, "—").replace(/--/g, "–")
    .replace(/~/g, " ").replace(/\\,/g, " ").replace(/\\%/g, "%").replace(/\\&/g, "&").replace(/\\_/g, "_")
    .replace(/\\#/g, "#").replace(/\\ /g, " ").replace(/\s+/g, " ");
}
// returns segments [{text, bold, italics, mono, br}]
function segs(s, fmt = {}) {
  const out = []; let i = 0, buf = "";
  const flush = () => { if (buf) out.push({ text: plain(buf), ...fmt }); buf = ""; };
  while (i < s.length) {
    if (s[i] === "$") {
      const j = s.indexOf("$", i + 1); flush(); out.push({ text: mathText(s.slice(i + 1, j)), ...fmt }); i = j + 1; continue;
    }
    if (s.startsWith("\\\\", i)) { flush(); out.push({ br: true }); i += 2; continue; }
    if (s[i] === "\\") {
      const m = /^\\([a-zA-Z]+)\*?/.exec(s.slice(i));
      if (m) {
        const name = m[1]; let j = i + m[0].length;
        const arg = () => { while (s[j] === " ") j++; const [v, e] = braced(s, j); j = e; return v; };
        if (["textbf", "emph", "textit", "texttt", "url"].includes(name)) {
          const v = arg(); flush();
          out.push(...segs(v, { ...fmt, ...(name === "textbf" ? { bold: true } : name === "texttt" || name === "url" ? { mono: true } : { italics: true }) }));
          i = j; continue;
        }
        if (name === "textcolor") { arg(); const v = arg(); flush(); out.push(...segs(v, fmt)); i = j; continue; }
        if (name === "cite") { const v = arg(); flush(); out.push({ text: "[" + v.split(",").map((k) => citeOrder.indexOf(k.trim()) + 1).join(", ") + "]", ...fmt }); i = j; continue; }
        if (name === "ref") { const v = arg(); flush(); out.push({ text: labels[v] || "?", ...fmt }); i = j; continue; }
        if (name === "label") { arg(); i = j; continue; }
        if (["noindent", "small", "centering", "maketitle", "medskip", "smallskip"].includes(name)) { i = j; continue; }
        if (macros[name] !== undefined) { flush(); out.push(...segs(macros[name], fmt)); i = j; continue; }
      }
    }
    buf += s[i]; i++;
  }
  flush();
  // merge whitespace at boundaries
  return out.filter((x) => x.br || x.text !== "");
}
const run = (x, extra = {}) => x.br ? new D.TextRun({ break: 1 }) : new D.TextRun({
  text: x.text, bold: x.bold || extra.bold, italics: x.italics, size: extra.size,
  font: x.mono ? { ascii: "Consolas", hAnsi: "Consolas", eastAsia: "Malgun Gothic" } : undefined,
});
const runs = (s, extra) => { const sg = segs(s); if (sg.length && !sg[0].br) sg[0].text = sg[0].text.replace(/^\s+/, ""); return sg.map((x) => run(x, extra)); };

// ---------- tables ----------
function tableBlock(content) {
  const [caption] = braced(content, content.indexOf("\\caption{") + 8);
  const ti = content.indexOf("\\begin{tabular}{");
  const [spec, afterSpec] = braced(content, ti + 15);
  const inner = content.slice(afterSpec, content.indexOf("\\end{tabular}"));
  const aligns = []; spec.replace(/p\{([^}]*)\}|[lrc]/g, (m, w) => { aligns.push(w ? { a: "l", w: parseFloat(w) } : { a: m }); return m; });
  const n = aligns.length, rows = [];
  let ruleNext = false;
  for (let raw of splitTop(inner, "\\\\")) {
    let r = raw.trim();
    const rule = /\\(toprule|midrule|bottomrule)/.test(r);
    r = r.replace(/\\(toprule|midrule|bottomrule)/g, "").trim();
    if (!r) { ruleNext = ruleNext || rule; continue; }
    rows.push({ cells: splitTop(r, "&").map((c) => c.trim()), ruleBefore: ruleNext || rule });
    ruleNext = false;
  }
  // column widths from content length (p{} columns use their cm width)
  const textOf = (c) => segs(c.replace(/\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}/, "$1")).map((x) => x.text || "").join("");
  const lens = aligns.map((al, k) => al.w ? al.w * 10 : Math.max(4, ...rows.map((r) => r.cells.length === n ? textOf(r.cells[k] || "").length : 0)));
  const total = lens.reduce((a, b) => a + b, 0);
  const widths = lens.map((x) => Math.floor((x / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const line = { style: D.BorderStyle.SINGLE, size: 8, color: "000000" }, none = { style: D.BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const trows = rows.map((r, ri) => {
    const cells = [];
    let k = 0;
    for (const c of r.cells) {
      const mc = /^\\multicolumn\{(\d+)\}\{[^}]*\}\{([\s\S]*)\}$/.exec(c);
      const span = mc ? parseInt(mc[1], 10) : 1;
      const text = mc ? mc[2] : c.replace(/^\\multirow\{[^}]*\}\{[^}]*\}\{([\s\S]*)\}$/, "$1");
      const w = widths.slice(k, k + span).reduce((a, b) => a + b, 0);
      const al = mc ? "l" : aligns[k].a;
      cells.push(new D.TableCell({
        columnSpan: span, width: { size: w, type: D.WidthType.DXA },
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        shading: ri === 0 ? { type: D.ShadingType.CLEAR, color: "auto", fill: "EDEDED" } : undefined,
        borders: { top: ri === 0 || r.ruleBefore ? line : none, bottom: ri === rows.length - 1 || ri === 0 ? line : none, left: none, right: none },
        children: [new D.Paragraph({ alignment: al === "r" ? D.AlignmentType.RIGHT : al === "c" ? D.AlignmentType.CENTER : D.AlignmentType.LEFT,
          spacing: { before: 0, after: 0 }, children: runs(text, { size: 17, bold: ri === 0 }) })],
      }));
      k += span;
    }
    return new D.TableRow({ children: cells, tableHeader: ri === 0 });
  });
  const number = String(Object.keys(labelsSeen).length + 1); labelsSeen[number] = true;
  return [
    new D.Paragraph({ keepNext: true, spacing: { before: 200, after: 80 }, children: [new D.TextRun({ text: `${L.table} ${number}. `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
    new D.Table({ width: { size: CONTENT_W, type: D.WidthType.DXA }, columnWidths: widths, rows: trows }),
    new D.Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const labelsSeen = {};

// ---------- pass 2: blocks ----------
const children = [];
const P = (s, opts = {}) => new D.Paragraph({ alignment: BODY_ALIGN, spacing: { after: 120, line: LINE }, ...opts, children: opts.children || runs(s) });
// title page
titleSrc.split("\\\\").forEach((t) => children.push(new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 60 }, children: [new D.TextRun({ text: plain(t).trim(), bold: true, size: 32 })] })));
children.push(new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { before: 200, after: 300 }, children: [new D.TextRun({ text: plain(authorSrc), italics: true })] }));

let i = 0, buf = "", sec = 0, sub = 0, app = false, runIn = null;
const flush = () => {
  const t = buf.trim(); buf = "";
  if (!t && !runIn) return;
  const kids = runIn ? [new D.TextRun({ text: plain(runIn) + " ", bold: true }), ...runs(t)] : runs(t);
  runIn = null;
  if (!kids.length) return; // e.g. a bare \label after a heading
  children.push(P("", { children: kids }));
};
const isList = (env) => env === "itemize" || env === "enumerate";
let listInstance = 0;
while (i < body.length) {
  if (body.startsWith("\\begin{", i)) {
    const [env, a] = braced(body, i + 6), endTag = `\\end{${env}}`, e = body.indexOf(endTag, a);
    const content = body.slice(a, e); i = e + endTag.length; flush();
    if (env === "abstract") {
      children.push(new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 100 }, children: [new D.TextRun({ text: L.abstract, bold: true })] }));
      children.push(P(content, { indent: { left: 400, right: 400 } }));
    } else if (env === "table") {
      children.push(...tableBlock(content));
    } else if (isList(env)) {
      listInstance++;
      content.split(/\\item\s+/).slice(1).forEach((it) => children.push(new D.Paragraph({
        alignment: BODY_ALIGN, spacing: { after: 80, line: LINE },
        numbering: { reference: env === "itemize" ? "bullets" : "numbers", level: 0, instance: env === "itemize" ? 0 : listInstance },
        children: runs(it.trim()),
      })));
    }
    continue;
  }
  const cmd = /^\\(section|subsection|paragraph)\{/.exec(body.slice(i, i + 15));
  if (cmd) {
    flush();
    const [title, a] = braced(body, i + cmd[0].length - 1); i = a;
    if (cmd[1] === "paragraph") { runIn = title; continue; }
    if (cmd[1] === "section") { sec++; sub = 0; }
    else sub++;
    const num = cmd[1] === "section" ? (app ? `${L.appendix} ${String.fromCharCode(64 + sec)}.` : `${sec}`) : `${app ? String.fromCharCode(64 + sec) : sec}.${sub}`;
    children.push(new D.Paragraph({ heading: cmd[1] === "section" ? D.HeadingLevel.HEADING_1 : D.HeadingLevel.HEADING_2, keepNext: true,
      children: [new D.TextRun({ text: `${num} ` }), ...segs(title).map((x) => run(x))] }));
    continue;
  }
  if (body.startsWith("\\appendix", i)) { flush(); app = true; sec = 0; i += 9; continue; }
  if (body.startsWith("\\bibliographystyle{", i)) { i = braced(body, i + 18)[1]; continue; }
  if (body.startsWith("\\bibliography{", i)) {
    flush(); i = braced(body, i + 13)[1];
    children.push(new D.Paragraph({ heading: D.HeadingLevel.HEADING_1, children: [new D.TextRun(L.references)] }));
    citeOrder.forEach((k, n) => children.push(new D.Paragraph({ spacing: { after: 80 }, indent: { left: 500, hanging: 500 },
      children: [new D.TextRun({ text: `[${n + 1}]\t`, size: 19 }), new D.TextRun({ text: bib[k] ? bibLine(bib[k]) : k, size: 19 })] })));
    continue;
  }
  if (body.startsWith("\\maketitle", i)) { i += 10; continue; }
  if (body[i] === "\n" && /^\n[ \t]*\n/.test(body.slice(i, i + 10))) { flush(); i++; continue; }
  buf += body[i]; i++;
}
flush();

const doc = new D.Document({
  creator: "InsPLAD-QA", title: plain(titleSrc.replace(/\\\\/g, " ")),
  styles: {
    default: { document: { run: { font: FONT, size: 21 }, paragraph: { spacing: { after: 120 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: D.LevelFormat.DECIMAL, text: "%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1300, bottom: 1300, left: MARGIN, right: MARGIN } } },
    footers: { default: new D.Footer({ children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER, children: [new D.TextRun({ children: [D.PageNumber.CURRENT], size: 18 })] })] }) },
    children,
  }],
});
D.Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length, "bytes;", Object.keys(labelsSeen).length, "tables;", citeOrder.length, "citations"); });
