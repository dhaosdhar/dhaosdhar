"""Vue Tkinter de dhaos (``dhaos gui``) : onglets = conversations simultanées,
fil de conversation, appels d'outils et confirmations en ligne, sessions,
bases de savoir, journal et réglages. Tkinter est fourni avec Python
(paquet ``python3-tk`` sur Debian/Ubuntu) : aucune dépendance externe.
"""
from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk
from typing import Any, Callable

from .. import __version__
from ..config import Settings, write_config_keys
from ..policy import Journal
from .controller import Conversation, Event, GuiController

PALETTE = {
    "bg": "#0f1115", "surface": "#171a21", "surface2": "#1e222b", "text": "#e6e8ee", "muted": "#9aa3b2",
    "border": "#272c37", "accent": "#7c8cff", "accent_text": "#0f1115", "user": "#262c4a", "code": "#0c0e12",
    "ok": "#4ade80", "warn": "#fbbf24", "danger": "#f87171",
}
BACKENDS = ("ollama", "claude")
POLL_MS = 60
HEALTH_MS = 30_000


# ============================================================== Markdown → Text
_INLINE_RE = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`)")


def insert_inline(widget: tk.Text, line: str, base_tag: str) -> None:
    for part in _INLINE_RE.split(line):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            widget.insert("end", part[2:-2], (base_tag, "bold"))
        elif part.startswith("`") and part.endswith("`"):
            widget.insert("end", part[1:-1], (base_tag, "inline"))
        else:
            widget.insert("end", part, (base_tag,))


def insert_markdown(widget: tk.Text, text: str, base_tag: str = "assistant") -> None:
    """Rendu Markdown minimal : blocs de code, titres, puces, gras, code en ligne."""
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            buf: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            widget.insert("end", "\n".join(buf) + "\n", (base_tag, "code"))
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            widget.insert("end", m.group(2) + "\n", (base_tag, f"h{len(m.group(1))}"))
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            bullet = re.sub(r"^\s*([-*+])\s+", "• ", line)
            insert_inline(widget, bullet, base_tag)
            widget.insert("end", "\n", (base_tag,))
        elif line.strip() and set(line.strip()) <= {"|", "-", ":", " "}:
            pass  # séparateur de tableau Markdown
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            insert_inline(widget, "  ·  ".join(cells), base_tag)
            widget.insert("end", "\n", (base_tag,))
        else:
            insert_inline(widget, line, base_tag)
            widget.insert("end", "\n", (base_tag,))
        i += 1


# ============================================================ onglet de conversation
class ConversationTab(ttk.Frame):
    def __init__(self, master: Any, app: "DhaosApp", conv: Conversation) -> None:
        super().__init__(master)
        self.app = app
        self.conv = conv
        self._tool_marks: dict[str, str] = {}
        self._assist_open = False
        self._raw = ""
        self._t0 = 0.0
        self._timer: str | None = None

        header = ttk.Frame(self)
        header.pack(fill="x", padx=12, pady=(8, 0))
        ttk.Label(header, text=f"{conv.backend_name} · {conv.model or app.controller.default_model(conv.backend_name)} · session {conv.session.id}", style="Muted.TLabel").pack(side="left")
        ttk.Button(header, text="Fermer l'onglet", style="Ghost.TButton", command=lambda: app.close_tab(self)).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        self.text = tk.Text(
            body, wrap="word", state="disabled", bg=PALETTE["surface"], fg=PALETTE["text"],
            insertbackground=PALETTE["text"], relief="flat", padx=18, pady=14, spacing1=2, spacing3=4,
            font=app.font_base, highlightthickness=0, cursor="arrow",
        )
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._configure_tags()

        self.activity = ttk.Label(self, text="", style="Muted.TLabel")
        self.activity.pack(fill="x", padx=16)

        composer = ttk.Frame(self)
        composer.pack(fill="x", padx=12, pady=(4, 12))
        self.entry = tk.Text(
            composer, height=3, wrap="word", bg=PALETTE["surface2"], fg=PALETTE["text"], insertbackground=PALETTE["text"],
            relief="flat", padx=12, pady=10, font=app.font_base, highlightthickness=1,
            highlightbackground=PALETTE["border"], highlightcolor=PALETTE["accent"],
        )
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<KP_Enter>", self._on_return)
        self.send_btn = ttk.Button(composer, text="Envoyer ➤", style="Accent.TButton", command=self.send)
        self.send_btn.pack(side="right", padx=(8, 0), fill="y")
        self.status = ttk.Label(self, text="", style="Muted.TLabel")
        self.status.pack(fill="x", padx=16, pady=(0, 6))

        for m in conv.session.messages:  # fil de conversation d'une session reprise
            if m.role == "user":
                self.append_user(m.content)
            elif m.role == "assistant":
                if m.content.strip():
                    self.begin_assistant()
                    self._raw = m.content
                    self.end_assistant()
                for c in m.tool_calls:
                    self.add_tool_call(c.id, c.name, c.arguments)
            elif m.role == "tool":
                self.set_tool_result(m.tool_call_id or "", m.name or "", bool(m.is_error), m.content[:800])
        self.entry.focus_set()

    # ------------------------------------------------------------- style
    def _configure_tags(self) -> None:
        t = self.text
        t.tag_configure("user", background=PALETTE["user"], lmargin1=160, lmargin2=160, rmargin=12, justify="right", spacing1=8, spacing3=8)
        t.tag_configure("assistant", lmargin1=0, lmargin2=0, rmargin=40)
        t.tag_configure("bold", font=self.app.font_bold)
        t.tag_configure("inline", font=self.app.font_mono, background=PALETTE["code"])
        t.tag_configure("code", font=self.app.font_mono, background=PALETTE["code"], lmargin1=12, lmargin2=12, rmargin=12, spacing1=4, spacing3=4)
        t.tag_configure("h1", font=self.app.font_h1, spacing1=10)
        t.tag_configure("h2", font=self.app.font_h2, spacing1=8)
        t.tag_configure("h3", font=self.app.font_bold, spacing1=6)
        t.tag_configure("tool", foreground=PALETTE["muted"], font=self.app.font_mono, lmargin1=24, lmargin2=24, spacing1=4)
        t.tag_configure("tool_ok", foreground=PALETTE["ok"], font=self.app.font_mono)
        t.tag_configure("tool_err", foreground=PALETTE["danger"], font=self.app.font_mono)
        t.tag_configure("tool_result", foreground=PALETTE["muted"], font=self.app.font_small_mono, lmargin1=48, lmargin2=48, spacing3=6)
        t.tag_configure("error", foreground=PALETTE["danger"], lmargin1=24)
        t.tag_configure("meta", foreground=PALETTE["muted"], font=self.app.font_small, lmargin1=8, spacing3=10)
        t.tag_configure("thinking", foreground=PALETTE["muted"], font=self.app.font_small, lmargin1=24)

    def _write(self, fn: Callable[[], None]) -> None:
        self.text.configure(state="normal")
        try:
            fn()
        finally:
            self.text.configure(state="disabled")
        self.text.see("end")

    # ------------------------------------------------------------ rendu
    def append_user(self, content: str) -> None:
        self._write(lambda: self.text.insert("end", content.strip() + "\n", ("user",)))
        self._write(lambda: self.text.insert("end", "\n"))

    def begin_assistant(self) -> None:
        if self._assist_open:
            return
        self._assist_open = True
        self._raw = ""

        def go() -> None:
            self.text.mark_set("assist_start", "end-1c")
            self.text.mark_gravity("assist_start", "left")

        self._write(go)

    def append_assistant(self, chunk: str) -> None:
        if not self._assist_open:
            self.begin_assistant()
        self._raw += chunk
        self._write(lambda: self.text.insert("end", chunk, ("assistant",)))

    def end_assistant(self) -> None:
        if not self._assist_open:
            return
        raw = self._raw

        def go() -> None:
            self.text.delete("assist_start", "end-1c")
            insert_markdown(self.text, raw.strip(), "assistant")
            self.text.insert("end", "\n")

        self._write(go)
        self._assist_open = False
        self._raw = ""

    def add_tool_call(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.end_assistant()
        args = ", ".join(f"{k}={self._short(v)}" for k, v in (arguments or {}).items())

        def go() -> None:
            self.text.insert("end", f"⚙ {name}({args}) ", ("tool",))
            mark = f"tool_{call_id}"
            self.text.mark_set(mark, "end-1c")
            self.text.mark_gravity(mark, "left")
            self._tool_marks[call_id] = mark
            self.text.insert("end", "en cours…\n", ("tool",))

        self._write(go)

    def set_tool_result(self, call_id: str, name: str, is_error: bool, preview: str) -> None:
        mark = self._tool_marks.pop(call_id, None)

        def go() -> None:
            if mark and mark in self.text.mark_names():
                self.text.delete(mark, f"{mark} lineend")
                self.text.insert(mark, "erreur" if is_error else "ok", ("tool_err" if is_error else "tool_ok",))
                self.text.mark_unset(mark)
                self.text.insert(f"{mark} lineend" if mark in self.text.mark_names() else "end", "")
            else:
                self.text.insert("end", f"⚙ {name} → {'erreur' if is_error else 'ok'}\n", ("tool",))
            snippet = preview.strip()
            if snippet:
                if len(snippet) > 600:
                    snippet = snippet[:600] + " […]"
                self.text.insert("end", snippet + "\n", ("tool_result",))

        self._write(go)

    def add_confirm(self, request_id: str, prompt: str) -> None:
        self.end_assistant()
        frame = tk.Frame(self.text, bg=PALETTE["surface2"], highlightthickness=1, highlightbackground=PALETTE["warn"], padx=12, pady=8)
        title = tk.Label(frame, text="Confirmation demandée", bg=PALETTE["surface2"], fg=PALETTE["warn"], font=self.app.font_bold)
        title.pack(anchor="w")
        tk.Label(frame, text=prompt, bg=PALETTE["surface2"], fg=PALETTE["text"], font=self.app.font_mono, wraplength=640, justify="left").pack(anchor="w", pady=(2, 8))
        buttons = tk.Frame(frame, bg=PALETTE["surface2"])
        buttons.pack(anchor="w")

        def answer(ok: bool) -> None:
            self.app.controller.answer_confirm(request_id, ok)
            for b in (yes, no):
                b.configure(state="disabled")
            title.configure(text="Autorisé" if ok else "Refusé", fg=PALETTE["ok"] if ok else PALETTE["danger"])

        yes = ttk.Button(buttons, text="Oui, autoriser", style="Accent.TButton", command=lambda: answer(True))
        no = ttk.Button(buttons, text="Non, refuser", command=lambda: answer(False))
        yes.pack(side="left", padx=(0, 8))
        no.pack(side="left")

        def go() -> None:
            self.text.window_create("end", window=frame, padx=24, pady=6)
            self.text.insert("end", "\n")

        self._write(go)

    def add_error(self, detail: str) -> None:
        self.end_assistant()
        self._write(lambda: self.text.insert("end", f"Erreur : {detail}\n", ("error",)))

    def add_meta(self, text: str) -> None:
        self._write(lambda: self.text.insert("end", text + "\n", ("meta",)))

    @staticmethod
    def _short(value: Any) -> str:
        s = repr(value) if not isinstance(value, str) else value
        return (s[:70] + "…") if len(s) > 70 else s

    # ------------------------------------------------------------ activité
    def set_activity(self, label: str | None) -> None:
        if self._timer:
            self.after_cancel(self._timer)
            self._timer = None
        if not label:
            self.activity.configure(text="")
            return
        import time as _time

        self._t0 = _time.monotonic()

        def tick() -> None:
            elapsed = int(_time.monotonic() - self._t0)
            self.activity.configure(text=f"⏳ {label}  {elapsed} s")
            self._timer = self.after(1000, tick)

        tick()

    # ------------------------------------------------------------- envoi
    def _on_return(self, event: Any) -> str | None:
        if event.state & 0x1:  # Maj+Entrée : nouvelle ligne
            return None
        self.send()
        return "break"

    def send(self) -> None:
        text = self.entry.get("1.0", "end").strip()
        if not text or self.conv.busy:
            return
        if not self.app.controller.send(self.conv.id, text):
            return
        self.entry.delete("1.0", "end")
        self.append_user(text)
        self.send_btn.configure(state="disabled")
        self.status.configure(text="")
        self.app.mark_busy(self, True)

    # ---------------------------------------------------------- événements
    def handle(self, ev: Event) -> None:
        kind, data = ev.kind, ev.data
        if kind == "status":
            self.set_activity(str(data.get("label", "")))
        elif kind == "text":
            self.set_activity(None)
            self.append_assistant(str(data.get("text", "")))
        elif kind == "thinking":
            self.set_activity("le modèle réfléchit…")
        elif kind == "tool_call":
            self.add_tool_call(str(data.get("id")), str(data.get("name")), data.get("arguments") or {})
        elif kind == "tool_result":
            self.set_tool_result(str(data.get("id")), str(data.get("name")), bool(data.get("is_error")), str(data.get("preview", "")))
        elif kind == "confirm":
            self.set_activity("en attente de votre confirmation…")
            self.add_confirm(str(data.get("id")), str(data.get("prompt", "")))
        elif kind == "done":
            self.set_activity(None)
            self.end_assistant()
            usage = data.get("usage") or {}
            self.status.configure(
                text=f"{usage.get('input_tokens', 0)} → {usage.get('output_tokens', 0)} jetons · "
                f"{data.get('iterations', 0)} itération(s) · {data.get('tool_calls', 0)} outil(s) · {data.get('stop_reason', '')}"
            )
            if data.get("error"):
                self.add_error(str(data["error"]))
            self.send_btn.configure(state="normal")
            self.app.mark_busy(self, False)
            self.app.set_tab_title(self, str(data.get("title") or self.conv.title))
            self.app.refresh_sessions()
        elif kind == "error":
            self.set_activity(None)
            self.add_error(str(data.get("detail", "erreur")))
            self.send_btn.configure(state="normal")
            self.app.mark_busy(self, False)


# ====================================================================== fenêtres
class KnowledgeWindow(tk.Toplevel):
    def __init__(self, app: "DhaosApp") -> None:
        super().__init__(app)
        self.app = app
        self.title("dhaos — Bases de savoir")
        self.geometry("980x640")
        self.configure(bg=PALETTE["bg"])
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=12, pady=12)
        ttk.Label(left, text="Bases", style="Title.TLabel").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("docs", "chunks", "emb"), show="tree headings", height=14)
        self.tree.heading("#0", text="Nom")
        self.tree.heading("docs", text="Docs")
        self.tree.heading("chunks", text="Chunks")
        self.tree.heading("emb", text="Embedder")
        self.tree.column("#0", width=180)
        self.tree.column("docs", width=60, anchor="e")
        self.tree.column("chunks", width=70, anchor="e")
        self.tree.column("emb", width=150)
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.show_selected())
        btns = ttk.Frame(left)
        btns.pack(fill="x")
        for label, cmd in (
            ("Nouvelle base", self.create_base), ("Renommer", self.rename_base), ("Description", self.describe_base),
            ("Réindexer", self.reindex_base), ("Supprimer", self.delete_base),
        ):
            ttk.Button(btns, text=label, command=cmd).pack(fill="x", pady=2)

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        self.head = ttk.Label(right, text="Sélectionnez une base.", style="Title.TLabel")
        self.head.pack(anchor="w")
        add = ttk.Frame(right)
        add.pack(fill="x", pady=6)
        ttk.Label(add, text="Ajouter :", style="Muted.TLabel").pack(side="left")
        for label, cmd in (("Fichiers…", self.add_files), ("Dossier…", self.add_folder), ("URL…", self.add_url), ("Note…", self.add_note)):
            ttk.Button(add, text=label, command=cmd).pack(side="left", padx=4)
        search = ttk.Frame(right)
        search.pack(fill="x", pady=6)
        self.query = ttk.Entry(search)
        self.query.pack(side="left", fill="x", expand=True)
        self.query.bind("<Return>", lambda e: self.search())
        self.mode = ttk.Combobox(search, values=["hybrid", "vector", "keyword"], state="readonly", width=9)
        self.mode.set("hybrid")
        self.mode.pack(side="left", padx=4)
        ttk.Button(search, text="Chercher", style="Accent.TButton", command=self.search).pack(side="left")
        self.status = ttk.Label(right, text="", style="Muted.TLabel")
        self.status.pack(anchor="w")
        self.out = tk.Text(right, wrap="word", state="disabled", bg=PALETTE["surface"], fg=PALETTE["text"], relief="flat", padx=12, pady=10, font=app.font_base, highlightthickness=0)
        self.out.pack(fill="both", expand=True)
        self.out.tag_configure("src", foreground=PALETTE["muted"], font=app.font_small_mono)
        self.out.tag_configure("doc", foreground=PALETTE["muted"], font=app.font_small_mono)
        self.refresh()

    # ------------------------------------------------------------ helpers
    @property
    def kb(self) -> Any:
        return self.app.controller.kb

    def selected(self) -> str | None:
        sel = self.tree.selection()
        return self.tree.item(sel[0], "text") if sel else None

    def _out(self, text: str, tag: str | None = None) -> None:
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("end", text, (tag,) if tag else ())
        self.out.configure(state="disabled")

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None], label: str) -> None:
        """Opération longue dans un fil ; ``done`` sur le fil Tk."""
        self.status.configure(text=label)

        def runner() -> None:
            try:
                result = work()
            except Exception as e:  # noqa: BLE001
                result = e
            self.app.later(lambda: self._finish(result, done))

        threading.Thread(target=runner, daemon=True).start()

    def _finish(self, result: Any, done: Callable[[Any], None]) -> None:
        self.status.configure(text="")
        if isinstance(result, Exception):
            messagebox.showerror("dhaos", str(result), parent=self)
            return
        done(result)

    def refresh(self) -> None:
        current = self.selected()
        self.tree.delete(*self.tree.get_children())
        try:
            bases = self.kb.list_bases()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("dhaos", str(e), parent=self)
            return
        for b in bases:
            iid = self.tree.insert("", "end", text=b.name, values=(b.n_docs, b.n_chunks, b.embedder))
            if b.name == current:
                self.tree.selection_set(iid)
        self.show_selected()

    def show_selected(self) -> None:
        name = self.selected()
        if not name:
            self.head.configure(text="Sélectionnez une base.")
            return
        info = self.kb.get_base(name)
        if info is None:
            self.head.configure(text="Sélectionnez une base.")
            return
        docs = self.kb.list_documents(name)
        self.head.configure(text=f"{name} — {info.description or 'sans description'}")
        lines = [f"{len(docs)} document(s) :"] + [f"  {d.source}  ({d.n_chunks} chunks)" for d in docs]
        self._out("\n".join(lines), "doc")

    # ------------------------------------------------------------ actions
    def create_base(self) -> None:
        name = simpledialog.askstring("Nouvelle base", "Nom (ex. developpeur) :", parent=self)
        if not name:
            return
        desc = simpledialog.askstring("Nouvelle base", "Description (facultatif) :", parent=self) or ""
        try:
            self.kb.create_base(name, desc)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("dhaos", str(e), parent=self)
        self.refresh()

    def rename_base(self) -> None:
        name = self.selected()
        if not name:
            return
        new = simpledialog.askstring("Renommer", "Nouveau nom :", initialvalue=name, parent=self)
        if new and new != name:
            try:
                self.kb.rename_base(name, new)
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("dhaos", str(e), parent=self)
            self.refresh()

    def describe_base(self) -> None:
        name = self.selected()
        if not name:
            return
        info = self.kb.get_base(name)
        if info is None:
            return
        desc = simpledialog.askstring("Description", "Description montrée au modèle :", initialvalue=info.description, parent=self)
        if desc is not None:
            self.kb.set_description(name, desc)
            self.refresh()

    def delete_base(self) -> None:
        name = self.selected()
        if name and messagebox.askyesno("Supprimer", f"Supprimer définitivement la base « {name} » et ses documents ?", parent=self):
            self.kb.delete_base(name)
            self.refresh()

    def reindex_base(self) -> None:
        name = self.selected()
        if name:
            self._run(lambda: self.kb.reindex(name), lambda n: (self.status.configure(text=f"réindexé : {n} chunk(s)"), self.refresh()), "réindexation…")

    def _ingest(self, sources: list[str]) -> None:
        name = self.selected()
        if not name or not sources:
            return
        self._run(
            lambda: self.kb.add(name, sources, recursive=True),
            lambda r: (self._out(r.summary() + ("\n" + "\n".join(r.errors) if r.errors else "")), self.refresh()),
            "ingestion en cours… (les embeddings peuvent prendre un moment)",
        )

    def add_files(self) -> None:
        self._ingest(list(filedialog.askopenfilenames(parent=self, title="Fichiers à ingérer")))

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Dossier à ingérer (récursif)")
        if folder:
            self._ingest([folder])

    def add_url(self) -> None:
        url = simpledialog.askstring("URL", "Adresse http(s) à ingérer :", parent=self)
        if url:
            self._ingest([url.strip()])

    def add_note(self) -> None:
        name = self.selected()
        if not name:
            return
        title = simpledialog.askstring("Note", "Titre :", parent=self)
        text = simpledialog.askstring("Note", "Texte à mémoriser :", parent=self)
        if text:
            self.kb.add_text(name, text, title=title or None)
            self.refresh()

    def search(self) -> None:
        name = self.selected()
        query = self.query.get().strip()
        if not query:
            return
        bases = [name] if name else None

        def show(hits: Any) -> None:
            self.out.configure(state="normal")
            self.out.delete("1.0", "end")
            if not hits:
                self.out.insert("end", "aucun résultat")
            for i, h in enumerate(hits, start=1):
                self.out.insert("end", f"[{i}] {h.base} · {h.source} · score {h.score:.3f}\n", ("src",))
                self.out.insert("end", h.text.strip()[:1200] + "\n\n")
            self.out.configure(state="disabled")

        self._run(lambda: self.kb.search(query, bases=bases, top_k=8, mode=self.mode.get()), show, "recherche…")


class JournalWindow(tk.Toplevel):
    def __init__(self, app: "DhaosApp") -> None:
        super().__init__(app)
        self.title("dhaos — Journal des actions")
        self.geometry("980x520")
        self.configure(bg=PALETTE["bg"])
        tree = ttk.Treeview(self, columns=("kind", "detail", "result"), show="headings")
        for col, label, width in (("kind", "Action", 110), ("detail", "Détail", 560), ("result", "Résultat", 220)):
            tree.heading(col, text=label)
            tree.column(col, width=width, anchor="w")
        tree.pack(fill="both", expand=True, padx=12, pady=12)
        for e in reversed(Journal(app.settings.journal_path).tail(300)):
            detail = e.get("command") or e.get("path") or ""
            if e.get("kind") == "run_command":
                result = "délai dépassé" if e.get("timeout") else ("interrompu" if e.get("interrupted") else f"exit {e.get('exit')}")
                result += f" · {e.get('duration')}s" if e.get("duration") is not None else ""
            else:
                result = f"{e.get('bytes', '')} o" + (" · sauvegarde" if e.get("backup") else "")
            when = str(e.get("iso", ""))[5:16].replace("T", " ")
            tree.insert("", "end", values=(f"{when}  {e.get('kind', '')}" + ("  ✓" if e.get("confirmed") else ""), detail, result))


class SettingsWindow(tk.Toplevel):
    FIELDS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
        ("backends.default", "Cerveau par défaut", BACKENDS),
        ("backends.ollama.model", "Modèle Ollama (dhaos)", None),
        ("backends.ollama.base_model", "Modèle de base de dhaos", None),
        ("backends.ollama.host", "Hôte Ollama", None),
        ("backends.ollama.num_ctx", "Contexte Ollama (num_ctx)", None),
        ("backends.claude.model", "Modèle Claude", None),
        ("tools.write_policy", "Écriture (write_policy)", ("project", "ask", "all", "deny")),
        ("tools.shell_policy", "Shell (shell_policy)", ("ask", "auto", "deny")),
        ("kb.embedder", "Embedder des bases", ("auto", "ollama", "sentence-transformers", "hash")),
        ("web.provider", "Recherche web", ("duckduckgo", "searxng", "brave")),
        ("agent.language", "Langue des réponses", None),
    )

    def __init__(self, app: "DhaosApp") -> None:
        super().__init__(app)
        self.app = app
        self.title("dhaos — Réglages")
        self.configure(bg=PALETTE["bg"])
        self.vars: dict[str, tk.StringVar] = {}
        form = ttk.Frame(self)
        form.pack(padx=18, pady=16)
        for row, (key, label, choices) in enumerate(self.FIELDS):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 12))
            var = tk.StringVar(value=str(self._get(key)))
            self.vars[key] = var
            if choices:
                ttk.Combobox(form, textvariable=var, values=list(choices), state="readonly", width=28).grid(row=row, column=1, sticky="ew")
            else:
                ttk.Entry(form, textvariable=var, width=30).grid(row=row, column=1, sticky="ew")
        self.auto_kb = tk.BooleanVar(value=bool(app.settings.agent.auto_kb_search))
        ttk.Checkbutton(form, text="Consulter les bases de savoir automatiquement", variable=self.auto_kb).grid(row=len(self.FIELDS), column=0, columnspan=2, sticky="w", pady=6)
        ttk.Label(form, text=f"Fichier : {app.settings.source_path}", style="Muted.TLabel").grid(row=len(self.FIELDS) + 1, column=0, columnspan=2, sticky="w")
        ttk.Button(form, text="Enregistrer", style="Accent.TButton", command=self.save).grid(row=len(self.FIELDS) + 2, column=0, columnspan=2, pady=(12, 0), sticky="ew")

    def _get(self, key: str) -> Any:
        node: Any = self.app.settings
        for part in key.split("."):
            node = getattr(node, part)
        return "" if node is None else node

    def save(self) -> None:
        values: dict[str, Any] = {}
        for key, var in self.vars.items():
            raw = var.get().strip()
            current = self._get(key)
            if raw == str(current):
                continue
            if key == "backends.ollama.num_ctx":
                values[key] = int(raw) if raw else None
            else:
                values[key] = raw
        if self.auto_kb.get() != bool(self.app.settings.agent.auto_kb_search):
            values["agent.auto_kb_search"] = bool(self.auto_kb.get())
        if not values:
            self.destroy()
            return
        try:
            path = write_config_keys(self.app.settings, values)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("dhaos", f"réglage invalide : {e}", parent=self)
            return
        messagebox.showinfo("dhaos", f"Enregistré dans {path}\nLes nouvelles conversations utilisent ces réglages.", parent=self)
        self.app.refresh_backend_controls()
        self.destroy()


# ================================================================ application
class DhaosApp(tk.Tk):
    def __init__(self, settings: Settings, controller: GuiController | None = None) -> None:
        super().__init__()
        self.settings = settings
        self.controller = controller or GuiController(settings)
        self.tabs: dict[str, ConversationTab] = {}
        # Les fils secondaires ne touchent jamais Tk : ils déposent ici des
        # appels que le fil principal exécute dans _poll.
        self._ui_calls: "queue.Queue[Callable[[], Any]]" = queue.Queue()
        self._health_timer: str | None = None
        self.title("dhaos")
        self.geometry("1280x820")
        self.minsize(900, 600)
        self.configure(bg=PALETTE["bg"])
        self._fonts()
        self._style()
        self._menu()
        self._layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind_all("<Control-n>", lambda e: self.new_tab())
        self.bind_all("<Control-w>", lambda e: self.close_tab(self.current_tab()))
        self.new_tab()
        self.after(POLL_MS, self._poll)
        self.after(200, self.refresh_health)

    # ------------------------------------------------------------- apparence
    def _fonts(self) -> None:
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=11)
        self.font_base = base
        self.font_bold = tkfont.Font(font=base)
        self.font_bold.configure(weight="bold")
        self.font_small = tkfont.Font(font=base)
        self.font_small.configure(size=9)
        self.font_h1 = tkfont.Font(font=base)
        self.font_h1.configure(size=16, weight="bold")
        self.font_h2 = tkfont.Font(font=base)
        self.font_h2.configure(size=13, weight="bold")
        mono = tkfont.nametofont("TkFixedFont")
        self.font_mono = tkfont.Font(font=mono)
        self.font_mono.configure(size=10)
        self.font_small_mono = tkfont.Font(font=mono)
        self.font_small_mono.configure(size=9)

    def _style(self) -> None:
        p = PALETTE
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=p["bg"], foreground=p["text"], bordercolor=p["border"], troughcolor=p["surface2"], fieldbackground=p["surface2"], font=self.font_base)
        style.configure("TFrame", background=p["bg"])
        style.configure("TLabel", background=p["bg"], foreground=p["text"])
        style.configure("Muted.TLabel", foreground=p["muted"], font=self.font_small)
        style.configure("Title.TLabel", font=self.font_h2)
        style.configure("TButton", background=p["surface2"], foreground=p["text"], borderwidth=0, padding=(12, 6))
        style.map("TButton", background=[("active", p["border"])])
        style.configure("Accent.TButton", background=p["accent"], foreground=p["accent_text"], font=self.font_bold)
        style.map("Accent.TButton", background=[("active", "#95a3ff"), ("disabled", p["surface2"])], foreground=[("disabled", p["muted"])])
        style.configure("Ghost.TButton", background=p["bg"], foreground=p["muted"])
        style.configure("TNotebook", background=p["bg"], borderwidth=0, tabmargins=(8, 6, 0, 0))
        style.configure("TNotebook.Tab", background=p["surface2"], foreground=p["muted"], padding=(14, 7), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", p["surface"])], foreground=[("selected", p["text"])])
        style.configure("TEntry", fieldbackground=p["surface2"], foreground=p["text"], insertcolor=p["text"], padding=6)
        style.configure("TCombobox", fieldbackground=p["surface2"], background=p["surface2"], foreground=p["text"], arrowcolor=p["muted"], padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", p["surface2"])], foreground=[("readonly", p["text"])])
        style.configure("TCheckbutton", background=p["bg"], foreground=p["text"])
        style.configure("Treeview", background=p["surface"], fieldbackground=p["surface"], foreground=p["text"], rowheight=26, borderwidth=0)
        style.configure("Treeview.Heading", background=p["surface2"], foreground=p["muted"], relief="flat")
        style.map("Treeview", background=[("selected", p["user"])])
        style.configure("Vertical.TScrollbar", background=p["surface2"], troughcolor=p["bg"], arrowcolor=p["muted"], borderwidth=0)
        self.option_add("*TCombobox*Listbox.background", p["surface2"])
        self.option_add("*TCombobox*Listbox.foreground", p["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", p["user"])

    def _menu(self) -> None:
        bar = tk.Menu(self, bg=PALETTE["surface"], fg=PALETTE["text"], activebackground=PALETTE["user"], activeforeground=PALETTE["text"], bd=0)
        fichier = tk.Menu(bar, tearoff=0, bg=PALETTE["surface"], fg=PALETTE["text"], activebackground=PALETTE["user"])
        fichier.add_command(label="Nouvelle conversation", accelerator="Ctrl+N", command=self.new_tab)
        fichier.add_command(label="Fermer l'onglet", accelerator="Ctrl+W", command=lambda: self.close_tab(self.current_tab()))
        fichier.add_separator()
        fichier.add_command(label="Quitter", command=self.on_close)
        outils = tk.Menu(bar, tearoff=0, bg=PALETTE["surface"], fg=PALETTE["text"], activebackground=PALETTE["user"])
        outils.add_command(label="Bases de savoir…", command=lambda: KnowledgeWindow(self))
        outils.add_command(label="Journal des actions…", command=lambda: JournalWindow(self))
        outils.add_command(label="Réglages…", command=lambda: SettingsWindow(self))
        outils.add_separator()
        outils.add_command(label="Créer le modèle dhaos dans Ollama…", command=self.create_model)
        aide = tk.Menu(bar, tearoff=0, bg=PALETTE["surface"], fg=PALETTE["text"], activebackground=PALETTE["user"])
        aide.add_command(label="À propos", command=lambda: messagebox.showinfo("dhaos", f"dhaos {__version__}\nAssistant de codage agentique avec bases de savoir locales.\nCerveau : Ollama (local) ou Claude (API).", parent=self))
        bar.add_cascade(label="Fichier", menu=fichier)
        bar.add_cascade(label="Outils", menu=outils)
        bar.add_cascade(label="Aide", menu=aide)
        self.configure(menu=bar)

    def _layout(self) -> None:
        side = ttk.Frame(self, width=270)
        side.pack(side="left", fill="y", padx=(12, 0), pady=12)
        side.pack_propagate(False)
        brand = ttk.Frame(side)
        brand.pack(fill="x", pady=(0, 10))
        ttk.Label(brand, text="dhaos", style="Title.TLabel").pack(side="left")
        self.health_label = ttk.Label(brand, text="●", foreground=PALETTE["muted"])
        self.health_label.pack(side="right")
        ttk.Button(side, text="＋ Nouvelle conversation", style="Accent.TButton", command=self.new_tab).pack(fill="x")

        controls = ttk.Frame(side)
        controls.pack(fill="x", pady=(14, 6))
        ttk.Label(controls, text="Cerveau", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.backend_var = tk.StringVar(value=self.settings.backends.default)
        backend_box = ttk.Combobox(controls, textvariable=self.backend_var, values=list(BACKENDS), state="readonly", width=12)
        backend_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        backend_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_backend_controls())
        ttk.Label(controls, text="Modèle", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.model_var = tk.StringVar(value="")
        self.model_box = ttk.Combobox(controls, textvariable=self.model_var, width=18)
        self.model_box.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.tools_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Outils (disque, shell, web, bases)", variable=self.tools_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        controls.columnconfigure(1, weight=1)
        self.health_text = ttk.Label(side, text="", style="Muted.TLabel", wraplength=250, justify="left")
        self.health_text.pack(fill="x")

        ttk.Label(side, text="SESSIONS", style="Muted.TLabel").pack(anchor="w", pady=(14, 4))
        lst = ttk.Frame(side)
        lst.pack(fill="both", expand=True)
        self.sessions = tk.Listbox(lst, bg=PALETTE["surface"], fg=PALETTE["text"], selectbackground=PALETTE["user"], selectforeground=PALETTE["text"], relief="flat", highlightthickness=0, activestyle="none", font=self.font_small)
        sb = ttk.Scrollbar(lst, orient="vertical", command=self.sessions.yview)
        self.sessions.configure(yscrollcommand=sb.set)
        self.sessions.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.sessions.bind("<Double-Button-1>", lambda e: self.open_selected_session())
        row = ttk.Frame(side)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Ouvrir", command=self.open_selected_session).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Supprimer", command=self.delete_selected_session).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(side, text=f"v{__version__} · projet : {self.settings.resolve_project_root()}", style="Muted.TLabel", wraplength=250).pack(anchor="w", pady=(8, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self._session_ids: list[str] = []
        self.refresh_sessions()
        self.refresh_backend_controls()

    # -------------------------------------------------------------- onglets
    def current_tab(self) -> ConversationTab | None:
        try:
            widget = self.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            return None
        return widget if isinstance(widget, ConversationTab) else None

    def _add_tab(self, conv: Conversation, title: str | None = None) -> ConversationTab:
        tab = ConversationTab(self.notebook, self, conv)
        self.tabs[conv.id] = tab
        self.notebook.add(tab, text=self._tab_text(title or conv.title))
        self.notebook.select(tab)
        return tab

    @staticmethod
    def _tab_text(title: str, busy: bool = False) -> str:
        short = title if len(title) <= 24 else title[:23] + "…"
        return ("⏳ " if busy else "") + short

    def new_tab(self) -> None:
        model = self.model_var.get().strip() or None
        try:
            conv = self.controller.new_conversation(backend=self.backend_var.get(), model=model, tools=bool(self.tools_var.get()))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("dhaos", str(e), parent=self)
            return
        self._add_tab(conv, "Nouvelle conversation")

    def close_tab(self, tab: ConversationTab | None) -> None:
        if tab is None:
            return
        if tab.conv.busy and not messagebox.askyesno("dhaos", "Cette conversation travaille encore. Fermer l'onglet ?", parent=self):
            return
        self.controller.close_conversation(tab.conv.id)
        self.tabs.pop(tab.conv.id, None)
        self.notebook.forget(tab)
        tab.destroy()
        self.refresh_sessions()
        if not self.tabs:
            self.new_tab()

    def set_tab_title(self, tab: ConversationTab, title: str) -> None:
        self.notebook.tab(tab, text=self._tab_text(title, tab.conv.busy))

    def mark_busy(self, tab: ConversationTab, busy: bool) -> None:
        self.notebook.tab(tab, text=self._tab_text(tab.conv.title, busy))

    # ------------------------------------------------------------- sessions
    def refresh_sessions(self) -> None:
        self.sessions.delete(0, "end")
        self._session_ids = []
        for info in self.controller.list_sessions():
            self.sessions.insert("end", f"{info.title or info.id}   ·  {info.backend} · {info.n_messages} msg")
            self._session_ids.append(info.id)

    def _selected_session_id(self) -> str | None:
        sel = self.sessions.curselection()
        return self._session_ids[sel[0]] if sel else None

    def open_selected_session(self) -> None:
        sid = self._selected_session_id()
        if not sid:
            return
        for tab in self.tabs.values():
            if tab.conv.session.id == sid:
                self.notebook.select(tab)
                return
        try:
            conv = self.controller.open_session(sid, tools=bool(self.tools_var.get()))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("dhaos", str(e), parent=self)
            return
        self._add_tab(conv)

    def delete_selected_session(self) -> None:
        sid = self._selected_session_id()
        if sid and messagebox.askyesno("dhaos", "Supprimer cette session ?", parent=self):
            for tab in list(self.tabs.values()):
                if tab.conv.session.id == sid:
                    self.close_tab(tab)
            self.controller.delete_session(sid)
            self.refresh_sessions()

    # ------------------------------------------------------------- backend
    def refresh_backend_controls(self) -> None:
        name = self.backend_var.get() or self.settings.backends.default
        self.model_var.set("")

        def work() -> None:
            models = self.controller.models(name)
            self.later(lambda: (self.model_box.configure(values=models), self.model_box.set(""), self.model_box.configure(state="normal")))

        threading.Thread(target=work, daemon=True).start()
        self.refresh_health()

    def refresh_health(self) -> None:
        name = self.backend_var.get() or self.settings.backends.default
        default = self.controller.default_model(name)

        def work() -> None:
            h = self.controller.health(name)
            ok = bool(h.get("ok"))
            detail = str(h.get("detail") or ("prêt" if ok else "indisponible"))
            self.later(lambda: self.health_label.configure(foreground=PALETTE["ok"] if ok else PALETTE["danger"]))
            self.later(lambda: self.health_text.configure(text=f"modèle par défaut : {default}\n{detail}"))

        threading.Thread(target=work, daemon=True).start()
        if self._health_timer:
            self.after_cancel(self._health_timer)
        self._health_timer = self.after(HEALTH_MS, self.refresh_health)

    def create_model(self) -> None:
        from ..agent.prompts import MODEL_IDENTITY
        from ..backends.ollama import OllamaBackend

        base = simpledialog.askstring("Modèle dhaos", "Modèle de base Ollama (doit être présent) :", initialvalue=self.settings.backends.ollama.base_model, parent=self)
        if not base:
            return

        def work() -> None:
            try:
                backend = OllamaBackend(self.settings)
                params = {"num_ctx": self.settings.backends.ollama.num_ctx} if self.settings.backends.ollama.num_ctx else {}
                backend.create_model("dhaos", base, system=MODEL_IDENTITY, parameters=params)
                write_config_keys(self.settings, {"backends.ollama.model": "dhaos", "backends.ollama.base_model": base})
                msg = f"Modèle dhaos créé à partir de {base} et défini par défaut."
            except Exception as e:  # noqa: BLE001
                msg = f"Échec : {e}"
            self.later(lambda: (messagebox.showinfo("dhaos", msg, parent=self), self.refresh_backend_controls()))

        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------------------- boucle
    def later(self, fn: Callable[[], Any]) -> None:
        """Exécuter ``fn`` sur le fil Tk (appelable depuis n'importe quel fil)."""
        self._ui_calls.put(fn)

    def _poll(self) -> None:
        while True:
            try:
                fn = self._ui_calls.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:  # noqa: BLE001 — une erreur d'affichage ne doit pas tuer la boucle
                print(f"dhaos gui : {type(e).__name__}: {e}")
        for ev in self.controller.drain():
            tab = self.tabs.get(ev.conv_id)
            if tab is not None:
                try:
                    tab.handle(ev)
                except Exception as e:  # noqa: BLE001
                    print(f"dhaos gui : {type(e).__name__}: {e}")
        self.after(POLL_MS, self._poll)

    def on_close(self) -> None:
        if any(t.conv.busy for t in self.tabs.values()) and not messagebox.askyesno("dhaos", "Des conversations travaillent encore. Quitter quand même ?", parent=self):
            return
        self.controller.shutdown()
        self.destroy()


def run_app(settings: Settings) -> None:
    """Point d'entrée de ``dhaos gui``."""
    app = DhaosApp(settings)
    app.mainloop()
