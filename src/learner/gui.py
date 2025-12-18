import tkinter as tk
from pathlib import Path
from tkinter import font, ttk

from PIL import Image, ImageTk


class LabelingWindow(tk.Tk):
    def __init__(self, class_names: tuple[str, str], labeled: int, unlabeled: int) -> None:
        super().__init__()

        self.title("Image Labeling Tool")
        self.configure(bg="#f5f5f5")
        self.protocol("WM_DELETE_WINDOW", self._on_quit)

        self.font = font.Font(family="Liberation Sans", size=10, weight="bold")
        self.button_font = font.Font(family="Liberation Sans", size=10, weight="normal")

        self.labeled = labeled
        self.unlabeled = unlabeled
        self.all = labeled + unlabeled

        progressbar_frame = tk.Frame(self, bg="#f5f5f5")
        progressbar_frame.pack(padx=20, pady=25, fill=tk.X)

        total_samples = self.labeled + self.unlabeled
        progress_text = f"{self.labeled} / {total_samples} samples labeled"
        self.progress_bar_label = tk.Label(
            progressbar_frame, text=progress_text, font=self.font, fg="#333333", bg="#f5f5f5"
        )
        self.progress_bar_label.pack(pady=(0, 10))

        progress_container = tk.Frame(progressbar_frame, bg="#e0e0e0", relief=tk.FLAT)
        progress_container.pack(pady=(0, 5))

        style = ttk.Style()
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor="#e0e0e0",
            background="#4caf50",
            bordercolor="#e0e0e0",
            lightcolor="#66bb6a",
            darkcolor="#388e3c",
            thickness=20,
        )

        self.progress_bar = ttk.Progressbar(
            progress_container,
            orient=tk.HORIZONTAL,
            length=512,
            mode="determinate",
            value=(self.labeled / total_samples * 100) if total_samples > 0 else 0,
            style="Custom.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(padx=2, pady=2)

        percentage = (self.labeled / total_samples * 100) if total_samples > 0 else 0
        self.percentage_label = tk.Label(
            progressbar_frame, text=f"{percentage:.1f}%", font=self.font, fg="#666666", bg="#f5f5f5"
        )
        self.percentage_label.pack()

        self.current_image: ImageTk.PhotoImage | None = None

        self.image_container = tk.Frame(self, bg="#ffffff", relief=tk.SOLID, borderwidth=2)
        self.image_container.pack(padx=20, pady=5)

        self.image_frame = tk.Frame(self.image_container, width=512, height=512, bg="#ffffff")
        self.image_frame.pack(padx=5, pady=5)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack(expand=True)

        minority_class, majority_class = class_names
        self.selected_class = tk.StringVar(value="")

        radio_frame = tk.Frame(self, bg="#f5f5f5")
        radio_frame.pack(pady=15)

        self.radio_button_0 = tk.Radiobutton(
            radio_frame,
            text=f"{minority_class} (0)",
            variable=self.selected_class,
            value="0",
            font=self.font,
            bg="#f5f5f5",
            activebackground="#d3d3d3",
            selectcolor="#e3f2fd",
            padx=20,
            pady=10,
        )
        self.radio_button_0.pack(side=tk.LEFT, padx=15)

        self.radio_button_1 = tk.Radiobutton(
            radio_frame,
            text=f"{majority_class} (1)",
            variable=self.selected_class,
            value="1",
            font=self.font,
            bg="#f5f5f5",
            activebackground="#d3d3d3",
            selectcolor="#e3f2fd",
            padx=20,
            pady=10,
        )
        self.radio_button_1.pack(side=tk.LEFT, padx=15)

        button_frame = tk.Frame(self, bg="#f5f5f5")
        button_frame.pack(pady=15, padx=20, fill=tk.X)

        self.confirmed = tk.IntVar(value=0)
        self.quitted = False

        self.training_label = tk.Label(
            button_frame,
            text="Model is training. Wait for next batch.",
            font=self.font,
            fg="#ff9800",
            bg="#f5f5f5",
        )
        self.training_label.pack(side=tk.LEFT, padx=5)
        self.training_label.pack_forget()

        self.quit_button = tk.Button(
            button_frame,
            text="Quit",
            command=self._on_quit,
            font=self.button_font,
            bg="#ef5350",
            fg="white",
            activebackground="#e53935",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
        )
        self.quit_button.pack(side=tk.RIGHT, padx=5)

        self.confirm_button = tk.Button(
            button_frame,
            text="Confirm",
            command=self._on_confirm,
            font=self.button_font,
            bg="#66bb6a",
            fg="white",
            activebackground="#4caf50",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
        )
        self.confirm_button.pack(side=tk.RIGHT, padx=5)

    def set_sample(self, image_path: Path) -> None:
        img = Image.open(image_path)
        img = img.resize((512, 512))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.current_image = photo
        self.confirmed.set(0)
        self.selected_class.set("")

    def update_progress_bar(self, labeled: int, unlabeled: int) -> None:
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.all = labeled + unlabeled

        self.progress_bar["value"] = (self.labeled / self.all) * 100
        self.progress_bar_label["text"] = f"{self.labeled}/{self.all} samples labeled"
        self.percentage_label["text"] = f"{(self.labeled / self.all * 100):.1f}%"

    def show_training_status(self) -> None:
        self.training_label.pack(side=tk.LEFT, padx=5)
        self.update_idletasks()

    def hide_training_status(self) -> None:
        self.training_label.pack_forget()
        self.update_idletasks()

    def wait_for_label(self) -> str:
        self.wait_variable(self.confirmed)
        return self.selected_class.get()

    def quit(self) -> None:
        self.destroy()

    def _on_confirm(self) -> None:
        if self.selected_class.get():
            self.confirmed.set(1)

    def _on_quit(self) -> None:
        self.quitted = True
        self.selected_class.set("q")
        self.confirmed.set(1)
