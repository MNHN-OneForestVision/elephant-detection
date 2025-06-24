import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class YOLOInterface:
    def __init__(self, root):
        self.processing = None
        self.root = root
        self.root.title("YOLO - Interface de Détection et Segmentation")
        self.root.geometry("1000x700")

        # Variables
        self.model_path = tk.StringVar()
        self.confidence = tk.DoubleVar(value=0.80)
        self.input_files = []
        self.output_folder = tk.StringVar()
        self.name = tk.StringVar()
        self.tracking_enabled = tk.BooleanVar(value=False)
        self.model = None
        self.input_type = None

        self.setup_ui()

    def setup_ui(self):
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configuration du modèle
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

        # Frame pour les fichiers d'entrée
        input_frame = ttk.LabelFrame(main_frame, text="Fichiers d'Entrée", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(buttons_frame, text="Sélectionner Images",
                   command=self.select_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Sélectionner Vidéos",
                   command=self.select_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Effacer",
                   command=self.clear_files).pack(side=tk.LEFT, padx=5)

        # Liste des fichiers sélectionnés
        self.files_listbox = tk.Listbox(input_frame, height=6)
        scrollbar = ttk.Scrollbar(input_frame, orient=tk.VERTICAL)
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.files_listbox.yview)

        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Options de traitement
        options_frame = ttk.LabelFrame(main_frame, text="Options de Traitement", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(options_frame, text="Activer le tracking (vidéos uniquement)",
                        variable=self.tracking_enabled).pack(anchor=tk.W, pady=2)

        # Variables pour les nouvelles options
        self.show_results = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Afficher les résultats",
                        variable=self.show_results).pack(anchor=tk.W, pady=2)

        self.save_results = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Sauvegarder les résultats",
                        variable=self.save_results, command=self.toggle_output_frame).pack(anchor=tk.W, pady=2)

        self.output_container = ttk.Frame(options_frame)
        self.output_container.pack(fill=tk.X)

        # Configuration de sortie (LabelFrame dans le conteneur, non affiché au départ)
        self.output_frame = ttk.LabelFrame(self.output_container, text="Configuration de Sortie", padding=10)

        ttk.Label(self.output_frame, text="Dossier de sortie:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.output_frame, textvariable=self.output_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.output_frame, text="Parcourir", command=self.browse_output).grid(row=0, column=2, padx=5,
                                                                                         pady=5)
        ttk.Label(self.output_frame, text="Nom du sous-dossier:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.output_frame, textvariable=self.name, width=30).grid(row=1, column=1, sticky=tk.W,
                                                                                      padx=5, pady=5)
        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="Lancer Traitement",
                   command=self.start_processing).pack(side=tk.LEFT, padx=5)

        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Variables de contrôle
        self.processing = False

    def toggle_output_frame(self):
        if self.save_results.get():
            self.output_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.output_frame.pack_forget()

    def log_message(self, message):
        """Ajoute un message dans la zone de log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def browse_model(self):
        """Sélectionne le fichier de modèle et le charge automatiquement"""
        filename = filedialog.askopenfilename(
            title="Sélectionner le modèle YOLO",
            filetypes=[("Modèles PyTorch", "*.pt"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            # Vérifier l'extension
            if not filename.lower().endswith('.pt'):
                messagebox.showerror("Erreur de Format",
                                     "Le fichier sélectionné n'est pas un modèle valide (.pt)")
                return

            self.model_path.set(filename)
            self.load_model_automatically()

    def update_confidence_from_entry(self, event=None):
        """Met à jour la confiance depuis la zone de saisie"""
        try:
            value = float(self.confidence.get())
            if 0.0 <= value <= 1.0:
                # La valeur est déjà mise à jour par la variable liée
                pass
            else:
                messagebox.showwarning("Valeur Invalide",
                                       "La confiance doit être entre 0.0 et 1.0")
                self.confidence.set(0.25)  # Valeur par défaut
        except ValueError:
            messagebox.showwarning("Valeur Invalide",
                                   "Veuillez entrer une valeur numérique valide")
            self.confidence.set(0.25)  # Valeur par défaut

    def load_model_automatically(self):
        """Charge automatiquement le modèle sélectionné"""
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
        """Sélectionne des vidéos avec vérification multi-vidéos"""

        if self.input_type == "image":
            messagebox.showwarning("Type de fichier",
                                   "Vous avez déjà sélectionné des vidéos. Veuillez effacer la liste avant de sélectionner des images.")
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
        """Compte le nombre de vidéos dans la liste actuelle"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.MP4', '.AVI'}
        count = 0
        for file_path in self.input_files:
            if Path(file_path).suffix.lower() in video_extensions:
                count += 1
        return count


    def browse_output(self):
        """Sélectionne le dossier de sortie"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if folder:
            self.output_folder.set(folder)

    def select_images(self):
        """Sélectionne des images"""
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
        """Sélectionne un dossier et ajoute tous les fichiers image/vidéo"""
        folder = filedialog.askdirectory(title="Sélectionner un dossier")
        if folder:
            files = self.get_all_media_files(folder)
            self.add_files(files)

    def get_all_media_files(self, folder):
        """Récupère tous les fichiers média d'un dossier"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                      '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if Path(filename).suffix.lower() in extensions:
                    files.append(os.path.join(root, filename))
        return files

    def add_files(self, files):
        """Ajoute des fichiers à la liste"""
        for file in files:
            if file not in self.input_files:
                self.input_files.append(file)
                self.files_listbox.insert(tk.END, os.path.basename(file))

    def clear_files(self):
        """Efface la liste des fichiers"""
        self.input_files.clear()
        self.files_listbox.delete(0, tk.END)

        # Supprimer les méthodes liées au chargement manuel et dossier

    def get_processing_options(self):
        """Retourne les options de traitement pour le backend"""
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
        """Met à jour la barre de progression (interface pour backend)"""
        self.progress_var.set(value)
        self.root.update_idletasks()

    def get_log_callback(self):
        """Retourne la fonction de log pour le backend"""
        return self.log_message

    def validate_inputs(self):
        """Valide les entrées avant le traitement"""
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
        """Démarre le traitement en arrière-plan"""
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
        if options['save_results']:
            self.log_message(f'folder_name: {options["output_folder"]}\nname: {options["name"]}')
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

    def analyse(self,options):
        self.log_message('Debut de l\'analyse')
        print(options)
        if options['save_results'] is True:
            self.model(options['input_files'], show=options['show_results'], conf=options['confidence'], save=options['save_results'], name=options['name'], project=options['output_folder'])
            self.log_message(f'Sauvegarde de l\'analyse à : {options["output_folder"]}/{options["name"] if options["name"] else "predict"}')
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