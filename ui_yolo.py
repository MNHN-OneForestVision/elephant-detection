import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import cv2
import threading


class MediaViewer:
    def __init__(self, parent):
        self.parent = parent
        self.viewer_window = None
        self.current_image = None
        self.video_cap = None
        self.video_thread = None
        self.video_playing = False
        self.video_paused = False
        self.video_after_id = None
        self.video_label = None

    def show_image(self, image_path):
        if not PIL_AVAILABLE:
            messagebox.showerror("Erreur",
                                 "PIL/Pillow n'est pas installé. Veuillez installer Pillow pour visualiser les images.")
            return

        try:
            self.stop_video()

            if self.viewer_window is None or not self.viewer_window.winfo_exists():
                self.create_viewer_window()

            image = Image.open(image_path)
            max_width, max_height = 800, 600
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_image)
            self.viewer_window.title(f"Visualiseur - {os.path.basename(image_path)}")
            self.video_controls.pack_forget()
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.pack_forget()
            self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image: {str(e)}")

    def show_video(self, video_path):
        try:
            if self.viewer_window is None or not self.viewer_window.winfo_exists():
                self.create_viewer_window()
            self.stop_video()
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                messagebox.showerror("Erreur", "Impossible d'ouvrir la vidéo")
                return

            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.viewer_window.title(f"Visualiseur - {os.path.basename(video_path)}")
            self.image_label.pack_forget()
            if self.video_label and self.video_label.winfo_exists():
                self.video_label.destroy()

            self.video_label = tk.Label(self.viewer_window)
            self.video_controls.pack(fill=tk.X, padx=10, pady=5)
            self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            self.video_progress['value'] = 0
            self.video_playing = True
            self.video_paused = False
            self.play_video()

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher la vidéo: {str(e)}")

    def create_viewer_window(self):
        self.viewer_window = tk.Toplevel(self.parent)
        self.viewer_window.title("Visualiseur de Médias")
        self.viewer_window.geometry("900x700")
        self.viewer_window.protocol("WM_DELETE_WINDOW", self.close_viewer)

        self.image_label = tk.Label(self.viewer_window)
        self.video_controls = ttk.Frame(self.viewer_window)

        self.play_button = ttk.Button(self.video_controls, text="▶ Play", command=self.toggle_video)
        self.play_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.video_controls, text="⏹ Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)

        self.video_progress = ttk.Progressbar(self.video_controls, length=300)
        self.video_progress.pack(side=tk.LEFT, padx=10)

    def play_video(self):
        if not self.video_playing or self.video_cap is None:
            return
        if (not self.viewer_window or not self.viewer_window.winfo_exists() or
                not self.video_label or not self.video_label.winfo_exists()):
            self.stop_video()
            return

        try:
            if not self.video_paused:
                ret, frame = self.video_cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    max_width, max_height = 800, 600

                    if width > max_width or height > max_height:
                        ratio = min(max_width / width, max_height / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        frame = cv2.resize(frame, (new_width, new_height))

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image)
                    if self.video_label and self.video_label.winfo_exists():
                        self.video_label.configure(image=photo)
                        self.video_label.image = photo
                    current_frame = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if self.total_frames > 0:
                        progress = (current_frame / self.total_frames) * 100
                        self.video_progress['value'] = progress
                    delay = int(1000 / self.fps) if self.fps > 0 else 33
                    if self.viewer_window and self.viewer_window.winfo_exists():
                        self.video_after_id = self.viewer_window.after(delay, self.play_video)
                else:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self.viewer_window and self.viewer_window.winfo_exists():
                        self.video_after_id = self.viewer_window.after(100, self.play_video)
            else:
                if self.viewer_window and self.viewer_window.winfo_exists():
                    self.video_after_id = self.viewer_window.after(100, self.play_video)
        except tk.TclError:
            self.stop_video()

    def toggle_video(self):
        if self.video_playing:
            self.video_paused = not self.video_paused
            if hasattr(self, 'play_button'):
                if self.video_paused:
                    self.play_button.config(text="▶ Play")
                else:
                    self.play_button.config(text="⏸ Pause")

    def stop_video(self):
        self.video_playing = False
        self.video_paused = False
        if self.video_after_id and self.viewer_window and self.viewer_window.winfo_exists():
            try:
                self.viewer_window.after_cancel(self.video_after_id)
            except tk.TclError:
                pass
            self.video_after_id = None
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if hasattr(self, 'video_progress'):
            try:
                self.video_progress['value'] = 0
            except tk.TclError:
                pass
        if hasattr(self, 'play_button'):
            try:
                self.play_button.config(text="▶ Play")
            except tk.TclError:
                pass

    def close_viewer(self):
        self.stop_video()
        if self.video_label:
            try:
                self.video_label.destroy()
            except tk.TclError:
                pass
            self.video_label = None

        if self.viewer_window:
            try:
                self.viewer_window.destroy()
            except tk.TclError:
                pass
            self.viewer_window = None


class YOLOInterface:
    def __init__(self, root):
        self.processing = None
        self.root = root
        self.root.title("YOLO - Interface de Détection et Segmentation")
        self.root.geometry("1000x700")

        self.model_path = tk.StringVar()
        self.confidence = tk.DoubleVar(value=0.80)
        self.input_files = []
        self.output_folder = tk.StringVar()
        self.name = tk.StringVar()
        self.tracking_enabled = tk.BooleanVar(value=False)
        self.model = None
        self.input_type = None

        self.media_viewer = MediaViewer(self.root)
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        model_frame = ttk.LabelFrame(main_frame, text="Configuration du Modèle", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Chemin du modèle (.pt):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Parcourir", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(model_frame, text="Confiance:").grid(row=1, column=0, sticky=tk.W, pady=5)
        confidence_frame = ttk.Frame(model_frame)
        confidence_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Scale(confidence_frame, from_=0.0, to=1.0, variable=self.confidence,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=(0, 10))
        confidence_entry = ttk.Entry(confidence_frame, textvariable=self.confidence, width=8)
        confidence_entry.pack(side=tk.LEFT)
        confidence_entry.bind('<Return>', self.update_confidence_from_entry)
        confidence_entry.bind('<FocusOut>', self.update_confidence_from_entry)

        input_frame = ttk.LabelFrame(main_frame, text="Fichiers d'Entrée", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(buttons_frame, text="Sélectionner Images",
                   command=self.select_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Sélectionner Vidéos",
                   command=self.select_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Visualiser",
                   command=self.preview_selected_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Effacer",
                   command=self.clear_files).pack(side=tk.LEFT, padx=5)

        files_frame = ttk.Frame(input_frame)
        files_frame.pack(fill=tk.BOTH, expand=True)

        self.files_listbox = tk.Listbox(files_frame, height=6)
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL)
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.files_listbox.yview)

        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.files_listbox.bind('<Double-Button-1>', self.on_file_double_click)

        options_frame = ttk.LabelFrame(main_frame, text="Options de Traitement", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(options_frame, text="Activer le tracking (vidéos uniquement)",
                        variable=self.tracking_enabled).pack(anchor=tk.W, pady=2)

        self.show_results = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Afficher les résultats",
                        variable=self.show_results).pack(anchor=tk.W, pady=2)

        self.save_results = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Sauvegarder les résultats",
                        variable=self.save_results, command=self.toggle_output_frame).pack(anchor=tk.W, pady=2)

        self.output_container = ttk.Frame(options_frame)
        self.output_container.pack(fill=tk.X)

        self.output_frame = ttk.LabelFrame(self.output_container, text="Configuration de Sortie", padding=10)

        ttk.Label(self.output_frame, text="Dossier de sortie:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.output_frame, textvariable=self.output_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.output_frame, text="Parcourir", command=self.browse_output).grid(row=0, column=2, padx=5,
                                                                                         pady=5)
        ttk.Label(self.output_frame, text="Nom du sous-dossier:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.output_frame, textvariable=self.name, width=30).grid(row=1, column=1, sticky=tk.W,
                                                                            padx=5, pady=5)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="Lancer Traitement",
                   command=self.start_processing).pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.processing = False

    def on_file_double_click(self, event):
        self.preview_selected_file()

    def preview_selected_file(self):
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner un fichier à prévisualiser")
            return

        selected_index = selection[0]
        if selected_index < len(self.input_files):
            file_path = self.input_files[selected_index]
            self.preview_file(file_path)

    def preview_file(self, file_path):
        if not os.path.exists(file_path):
            messagebox.showerror("Erreur", "Le fichier n'existe pas")
            return

        file_ext = Path(file_path).suffix.lower()

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

        if file_ext in image_extensions:
            self.media_viewer.show_image(file_path)
            self.log_message(f"Prévisualisation de l'image: {os.path.basename(file_path)}")
        elif file_ext in video_extensions:
            self.media_viewer.show_video(file_path)
            self.log_message(f"Prévisualisation de la vidéo: {os.path.basename(file_path)}")
        else:
            messagebox.showwarning("Format non supporté",
                                   f"Le format {file_ext} n'est pas supporté pour la prévisualisation")

    def toggle_output_frame(self):
        if self.save_results.get():
            self.output_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.output_frame.pack_forget()

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Sélectionner le modèle YOLO",
            filetypes=[("Modèles PyTorch", "*.pt"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            if not filename.lower().endswith('.pt'):
                messagebox.showerror("Erreur de Format",
                                     "Le fichier sélectionné n'est pas un modèle valide (.pt)")
                return

            self.model_path.set(filename)
            self.load_model_automatically()

    def update_confidence_from_entry(self, event=None):
        try:
            value = float(self.confidence.get())
            if 0.0 <= value <= 1.0:
                pass
            else:
                messagebox.showwarning("Valeur Invalide",
                                       "La confiance doit être entre 0.0 et 1.0")
                self.confidence.set(0.80)
        except ValueError:
            messagebox.showwarning("Valeur Invalide",
                                   "Veuillez entrer une valeur numérique valide")
            self.confidence.set(0.80)
    def load_model_automatically(self):
        if not self.model_path.get():
            return

        try:
            self.log_message("Chargement du modèle...")
            self.model = YOLO(self.model_path.get())
            self.log_message(f"Modèle chargé: {os.path.basename(self.model_path.get())}")
        except Exception as e:
            self.log_message(f"Erreur lors du chargement: {str(e)}")
            messagebox.showerror("Erreur", f"Impossible de charger le modèle: {str(e)}")

    def select_videos(self):
        if self.input_type == "image":
            messagebox.showwarning("Type de fichier",
                                   "Vous avez déjà sélectionné des images. Veuillez effacer la liste avant de sélectionner des vidéos.")
            return

        files = filedialog.askopenfilenames(
            title="Sélectionner des vidéos",
            filetypes=[
                ("Vidéos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.MP4 *.AVI"),
                ("Tous les fichiers", "*.*")
            ]
        )

        if files:
            self.input_type = "video"
            self.add_files(files)

    def count_video_files(self):
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.MP4', '.AVI'}
        count = 0
        for file_path in self.input_files:
            if Path(file_path).suffix.lower() in video_extensions:
                count += 1
        return count

    def browse_output(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if folder:
            self.output_folder.set(folder)

    def select_images(self):
        if self.input_type == "video":
            messagebox.showwarning("Type de fichier",
                                   "Vous avez déjà sélectionné des vidéos. Veuillez effacer la liste avant de sélectionner des images.")
            return

        files = filedialog.askopenfilenames(
            title="Sélectionner des images",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Tous les fichiers", "*.*")
            ]
        )
        if files:
            self.input_type = "image"
            self.add_files(files)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner un dossier")
        if folder:
            files = self.get_all_media_files(folder)
            self.add_files(files)

    def get_all_media_files(self, folder):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                      '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if Path(filename).suffix.lower() in extensions:
                    files.append(os.path.join(root, filename))
        return files

    def add_files(self, files):
        for file in files:
            if file not in self.input_files:
                self.input_files.append(file)
                self.files_listbox.insert(tk.END, os.path.basename(file))

    def clear_files(self):
        self.input_files.clear()
        self.files_listbox.delete(0, tk.END)
        self.input_type = None

    def get_processing_options(self):
        return {
            'model_path': self.model_path.get(),
            'confidence': self.confidence.get(),
            'tracking_enabled': self.tracking_enabled.get(),
            'save_results': self.save_results.get(),
            'show_results': self.show_results.get(),
            'output_folder': self.output_folder.get(),
            'name': self.name.get(),
            'input_files': self.input_files.copy(),
            'total_videos': self.count_video_files()
        }

    def update_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()

    def get_log_callback(self):
        return self.log_message

    def validate_inputs(self):
        if not self.model_path.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un modèle")
            return False

        if not self.model_path.get().lower().endswith('.pt'):
            messagebox.showerror("Erreur de Format",
                                 "Le modèle doit être un fichier .pt")
            return False

        if not self.input_files:
            messagebox.showerror("Erreur", "Veuillez sélectionner des fichiers à traiter")
            return False

        if self.save_results.get() and not self.output_folder.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de sortie pour sauvegarder les résultats")
            return False

        return True

    def start_processing(self):
        if not self.validate_inputs():
            return

        if self.processing:
            messagebox.showwarning("Avertissement", "Un traitement est déjà en cours")
            return

        if not hasattr(self, 'model') or self.model is None:
            self.load_model_automatically()

        self.processing = True
        self.progress_var.set(0)

        options = self.get_processing_options()
        self.log_message("Options de traitement configurées - Prêt pour le backend")
        self.log_message(f"Fichiers à traiter: {len(options['input_files'])}")
        self.log_message(f"Modèle: {os.path.basename(options['model_path'])}")
        self.log_message(f"Confiance: {options['confidence']}")
        self.log_message(f"Affichage: {'Oui' if options['show_results'] else 'Non'}")
        self.log_message(f"Sauvegarde: {'Oui' if options['save_results'] else 'Non'}")
        if options['total_videos'] > 0:
            self.log_message(f"Tracking vidéo: {'Oui' if options['tracking_enabled'] else 'Non'}")

        self.parsing(options)

    def parsing(self, options):
        if options['total_videos'] == 1:
            options['input_files'] = self.input_files[0]
        if options['total_videos'] > 1:
            return self.analyze_multiple_video(options)
        self.analyse(options)
        return None

    def analyze_multiple_video(self, options):
        print(f'hello')

    def analyse(self, options):
        self.log_message('Debut de l\'analyse')
        print(options)
        if options['save_results'] is True:
            self.model(options['input_files'], show=options['show_results'], conf=options['confidence'],
                       save=options['save_results'], name=options['name'], project=options['output_folder'])
            self.log_message(
                f'Sauvegarde de l\'analyse à : {options["output_folder"]}/{options["name"] if options["name"] else "predict"}')
        else:
            self.model(options['input_files'], show=options['show_results'], conf=options['confidence'])
        self.log_message('Fin de l\'analyse')
        self.processing = False


def main():
    root = tk.Tk()
    app = YOLOInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()