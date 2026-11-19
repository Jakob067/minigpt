"""Desktop interface for generating text from local miniGPT checkpoints."""

import tkinter as tk
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread
from tkinter import filedialog, font, ttk

import torch

from minigpt.generate import generate
from minigpt.tokenizer import CharTokenizer
from minigpt.train import load_model
from minigpt.utils import select_device

PAPER = "#f7f4ed"
WHITE = "#fffdf8"
INK = "#272d28"
MUTED = "#545d54"
LINE = "#d6d8cc"
ACCENT = "#315b48"
ERROR = "#a12626"

CHECKPOINTS = {
    "Tiny · bundled": Path("models/tiny.pt"),
    "Tiny · training run": Path("runs/tiny/best.pt"),
    "Small · training run": Path("runs/small/best.pt"),
}
SAMPLING_FIELDS = (
    ("max_new_tokens", "New characters", int, "200"),
    ("temperature", "Temperature", float, "0.8"),
    ("top_k", "Top-k (optional)", int, ""),
    ("top_p", "Top-p", float, "1.0"),
    ("seed", "Seed", int, "1337"),
)


def _run_generation(
    checkpoint: Path,
    prompt: str,
    sampling: dict[str, int | float | None],
    results: SimpleQueue[tuple[str, str] | Exception],
    closing: Event,
) -> None:
    try:
        device = select_device()
        model, state = load_model(checkpoint, device)
        tokenizer = CharTokenizer(state["vocab"])
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        result = generate(model, tokens, stop_requested=closing.is_set, **sampling)
        continuation = tokenizer.decode(result[0, tokens.size(1) :].tolist())
        results.put((continuation, device.type.upper()))
    except Exception as error:
        results.put(error)


class GeneratorWindow:
    """Collect generation settings and show the result in a single desktop window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.checkpoints = {name: path for name, path in CHECKPOINTS.items() if path.is_file()}
        self.fields: dict[str, tk.StringVar] = {}
        self.results: SimpleQueue[tuple[str, str] | Exception] = SimpleQueue()
        self.closing = Event()
        self.busy = False
        root.title("miniGPT")
        root.geometry("980x680")
        root.minsize(710, 590)
        root.configure(background=PAPER)
        self._configure_styles()
        self._build_window()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.bind("<Control-Return>", self._shortcut)
        if root.tk.call("tk", "windowingsystem") == "aqua":
            root.bind("<Command-Return>", self._shortcut)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        ui_font = font.nametofont("TkDefaultFont").actual("family")
        style.configure(".", background=PAPER, foreground=INK, font=(ui_font, 12))
        style.configure("Title.TLabel", font=("Georgia", 36, "bold"), foreground=ACCENT)
        style.configure("Subtitle.TLabel", foreground=MUTED, font=(ui_font, 13))
        style.configure("Heading.TLabel", font=(ui_font, 15, "bold"))
        style.configure("Status.TLabel", foreground=MUTED)
        style.configure("Error.TLabel", foreground=ERROR)
        style.configure("TButton", padding=(10, 7), bordercolor=LINE, relief="flat")
        style.map("TButton", background=[("active", LINE)])
        style.configure("Generate.TButton", background=ACCENT, foreground=WHITE, padding=(12, 10))
        style.map(
            "Generate.TButton",
            background=[("disabled", LINE), ("active", "#234735")],
            foreground=[("disabled", MUTED)],
        )
        for name in ("TEntry", "TCombobox"):
            style.configure(name, fieldbackground=WHITE, bordercolor=LINE, padding=6)
            style.map(name, bordercolor=[("focus", ACCENT)])
        style.map("TCombobox", fieldbackground=[("readonly", WHITE)])
        style.configure("TSeparator", background=LINE)

    def _build_window(self) -> None:
        page = ttk.Frame(self.root, padding=(26, 16))
        page.pack(fill="both", expand=True)
        page.columnconfigure(2, weight=1)
        page.rowconfigure(2, weight=1)

        header = ttk.Frame(page)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Label(header, text="miniGPT", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Text generator", style="Subtitle.TLabel").pack(
            side="left", padx=16, pady=(12, 0)
        )
        ttk.Separator(page).grid(row=1, column=0, columnspan=3, sticky="ew")

        controls = ttk.Frame(page, width=285)
        controls.grid(row=2, column=0, sticky="nsew", pady=16)
        controls.columnconfigure((0, 1), weight=1, uniform="settings")
        controls.rowconfigure(3, weight=1)
        ttk.Label(controls, text="Checkpoint", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        picker = ttk.Frame(controls)
        picker.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.checkpoint = ttk.Combobox(picker, width=20, state="readonly")
        self.checkpoint.pack(side="left", fill="x", expand=True)
        self.checkpoint["values"] = tuple(self.checkpoints)
        if self.checkpoints:
            self.checkpoint.current(0)
        ttk.Button(picker, text="Browse…", command=self._browse).pack(side="left", padx=(6, 0))

        ttk.Label(controls, text="Prompt", style="Heading.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        self.prompt = self._text_area(controls, row=3, columnspan=2, height=4)
        self.prompt.insert("1.0", "ROMEO:")
        ttk.Label(controls, text="Sampling", style="Heading.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(13, 5)
        )
        for index, (key, label, _, default) in enumerate(SAMPLING_FIELDS):
            row, column = divmod(index, 2)
            field = ttk.Frame(controls, padding=(0, 0, 6 if column == 0 else 0, 5))
            field.grid(row=5 + row, column=column, sticky="ew")
            ttk.Label(field, text=label).pack(anchor="w", pady=(0, 2))
            self.fields[key] = tk.StringVar(self.root, value=default)
            ttk.Entry(field, textvariable=self.fields[key], width=10).pack(fill="x")
        self.submit = ttk.Button(
            controls, text="Generate", style="Generate.TButton", command=self._start
        )
        self.submit.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        ttk.Separator(page, orient="vertical").grid(row=2, column=1, sticky="ns", padx=20, pady=16)
        result = ttk.Frame(page)
        result.grid(row=2, column=2, sticky="nsew", pady=16)
        result.columnconfigure(0, weight=1)
        result.rowconfigure(1, weight=1)
        ttk.Label(result, text="Generated text", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        self.copy = ttk.Button(result, text="Copy text", command=self._copy_text, state="disabled")
        self.copy.grid(row=0, column=1, sticky="e", pady=(0, 5))
        self.output = self._text_area(result, row=1, columnspan=2, height=4)
        self.output.tag_configure("prompt", foreground=ACCENT, background="#e7eee5")
        self.output.insert("1.0", "Your generated text will appear here.")
        self.output.configure(state="disabled")

        ttk.Separator(page).grid(row=3, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(page, text="Ready", style="Status.TLabel", wraplength=640)
        self.status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _text_area(self, parent: ttk.Frame, row: int, columnspan: int, height: int) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=columnspan, sticky="nsew")
        text = tk.Text(
            frame,
            width=1,
            height=height,
            wrap="word",
            font="TkFixedFont",
            foreground=INK,
            background=WHITE,
            insertbackground=INK,
            selectbackground="#cfe0d6",
            selectforeground=INK,
            highlightbackground=LINE,
            highlightcolor=ACCENT,
            highlightthickness=1,
            borderwidth=0,
            padx=12,
            pady=10,
            spacing3=3,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return text

    def _browse(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Open a trusted miniGPT checkpoint",
            filetypes=[("PyTorch checkpoints", "*.pt")],
        )
        if filename:
            path = Path(filename)
            label = str(path)
            self.checkpoints[label] = path
            self.checkpoint["values"] = tuple(self.checkpoints)
            self.checkpoint.set(label)

    def _sampling_values(self) -> dict[str, int | float | None]:
        sampling: dict[str, int | float | None] = {}
        for key, label, parse, _ in SAMPLING_FIELDS:
            value = self.fields[key].get().strip()
            try:
                sampling[key] = None if key == "top_k" and not value else parse(value)
            except ValueError as error:
                number = "a whole number" if parse is int else "a number"
                raise ValueError(f"{label} needs {number}.") from error
        return sampling

    def _shortcut(self, _event: tk.Event) -> str:
        self._start()
        return "break"

    def _start(self) -> None:
        if self.busy:
            return
        checkpoint = self.checkpoints.get(self.checkpoint.get())
        prompt = self.prompt.get("1.0", "end-1c")
        try:
            if checkpoint is None:
                raise ValueError("Choose a checkpoint to generate text.")
            if not prompt:
                self.prompt.focus_set()
                raise ValueError("Enter a prompt to generate text.")
            sampling = self._sampling_values()
        except ValueError as error:
            self.status.configure(text=str(error), style="Error.TLabel")
            return

        self.busy = True
        self.submit.state(["disabled"])
        self.copy.state(["disabled"])
        self.status.configure(text="Generating text…", style="Status.TLabel")
        Thread(
            target=_run_generation,
            args=(checkpoint, prompt, sampling, self.results, self.closing),
        ).start()
        self.root.after(50, self._poll_result, prompt)

    def _close(self) -> None:
        self.closing.set()
        self.root.destroy()

    def _poll_result(self, prompt: str) -> None:
        try:
            result = self.results.get_nowait()
        except Empty:
            self.root.after(50, self._poll_result, prompt)
            return

        self.busy = False
        self.submit.state(["!disabled"])
        if isinstance(result, Exception):
            self.status.configure(text=str(result).partition("\n")[0], style="Error.TLabel")
        else:
            continuation, device = result
            self.output.configure(state="normal")
            self.output.delete("1.0", "end")
            self.output.insert("end", prompt, "prompt")
            self.output.insert("end", continuation)
            self.output.configure(state="disabled")
            self.output.see("1.0")
            self.status.configure(text=f"Generated {len(continuation)} characters on {device}.")
        if self.output.tag_ranges("prompt"):
            self.copy.state(["!disabled"])

    def _copy_text(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output.get("1.0", "end-1c"))
        self.status.configure(text="Copied to clipboard.", style="Status.TLabel")


def main() -> None:
    """Launch the miniGPT desktop window."""
    root = tk.Tk()
    GeneratorWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
