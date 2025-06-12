import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font

# Couleurs modernes
BG_COLOR = "#f5f7fa"
PRIMARY_COLOR = "#4b6cb7"
SECONDARY_COLOR = "#ffffff"
TEXT_COLOR = "#2d3748"
ACCENT_COLOR = "#48bb78"
ERROR_COLOR = "#e53e3e"
FONT_FAMILY = "Segoe UI"

# Construction du modèle
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(18,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Simuler un modèle et scaler
model = build_model()
scaler = StandardScaler()
scaler.mean_ = np.zeros(18)
scaler.scale_ = np.ones(18)

class ModernObesityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Prédiction d'Obésité")
        self.root.geometry("800x700")
        self.root.configure(bg=BG_COLOR)
        
        # Configuration du style
        self.setup_styles()
        
        # Cadre principal
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # En-tête
        self.setup_header()
        
        # Formulaire
        self.setup_form()
        
        # Résultats
        self.setup_results()
        
        # Bouton de prédiction
        self.setup_button()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configuration des styles
        style.configure("Main.TFrame", background=BG_COLOR)
        style.configure("Header.TLabel", 
                      background=BG_COLOR, 
                      foreground=PRIMARY_COLOR,
                      font=(FONT_FAMILY, 18, 'bold'))
        style.configure("Section.TFrame", 
                      background=SECONDARY_COLOR,
                      relief=tk.RAISED,
                      borderwidth=1)
        style.configure("Field.TLabel", 
                      background=SECONDARY_COLOR,
                      font=(FONT_FAMILY, 10))
        style.configure("Accent.TButton", 
                      background=PRIMARY_COLOR,
                      foreground=SECONDARY_COLOR,
                      font=(FONT_FAMILY, 12, 'bold'),
                      padding=10)
        style.map("Accent.TButton",
                 background=[('active', '#3a56a0')])
        style.configure("Result.TLabel", 
                      font=(FONT_FAMILY, 14))
        style.configure("Error.TEntry", 
                      fieldbackground="#fee2e2")
    
    def setup_header(self):
        header_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(
            header_frame,
            text="SYSTÈME DE PRÉDICTION D'OBÉSITÉ",
            style="Header.TLabel"
        )
        title.pack()
        
        subtitle = ttk.Label(
            header_frame,
            text="Veuillez remplir le formulaire ci-dessous",
            style="Field.TLabel"
        )
        subtitle.pack()
    
    def setup_form(self):
        form_frame = ttk.Frame(self.main_frame, style="Section.TFrame")
        form_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Configuration des champs
        self.fields = [
            ("Genre", "Gender", ["Femme (0)", "Homme (1)"]),
            ("Âge", "Age", None),
            ("Antécédents familiaux", "family_history_with_overweight", ["Non (0)", "Oui (1)"]),
            ("Aliments caloriques", "FAVC", ["Non (0)", "Oui (1)"]),
            ("Légumes (1-3)", "FCVC", None),
            ("Repas/jour (1-4)", "NCP", None),
            ("Grignotage", "CAEC", ["Non (0)", "Oui (1)"]),
            ("Fumeur", "SMOKE", ["Non (0)", "Oui (1)"]),
            ("Eau/jour (1-3)", "CH2O", None),
            ("Surveillance calories", "SCC", ["Non (0)", "Oui (1)"]),
            ("Activité physique (0-3)", "FAF", None),
            ("Temps écrans (0-2)", "TUE", None),
            ("Consommation alcool", "CALC", ["Non (0)", "Oui (1)"]),
            ("Transport - Voiture", "Automobile", ["Non (0)", "Oui (1)"]),
            ("Transport - Vélo", "Bike", ["Non (0)", "Oui (1)"]),
            ("Transport - Moto", "Motorbike", ["Non (0)", "Oui (1)"]),
            ("Transport - Commun", "Public_Transportation", ["Non (0)", "Oui (1)"]),
            ("Transport - Marche", "Walking", ["Non (0)", "Oui (1)"])
        ]
        
        # Création des champs dans une grille
        self.entries = {}
        for i, (label_text, key, options) in enumerate(self.fields):
            row = i // 2
            col = i % 2
            
            field_frame = ttk.Frame(form_frame, style="Section.TFrame")
            field_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
            
            label = ttk.Label(field_frame, text=label_text, style="Field.TLabel")
            label.pack(anchor=tk.W)
            
            if options:
                entry = ttk.Combobox(field_frame, values=options, state="readonly")
                entry.set(options[0])
            else:
                entry = ttk.Entry(field_frame)
            
            entry.pack(fill=tk.X, pady=5)
            self.entries[key] = entry
            
            # Configurer le poids des colonnes pour le redimensionnement
            form_frame.columnconfigure(col, weight=1)
        
        # Configurer le poids des lignes
        for r in range((len(self.fields) + 1) // 2):
            form_frame.rowconfigure(r, weight=1)
    
    def setup_results(self):
        self.result_frame = ttk.Frame(self.main_frame, style="Section.TFrame")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.result_icon = ttk.Label(
            self.result_frame,
            text="❓",
            font=(FONT_FAMILY, 48),
            background=SECONDARY_COLOR
        )
        self.result_icon.pack(pady=10)
        
        self.result_text = ttk.Label(
            self.result_frame,
            text="Veuillez saisir vos informations",
            style="Result.TLabel"
        )
        self.result_text.pack()
        
        self.confidence_text = ttk.Label(
            self.result_frame,
            text="",
            style="Field.TLabel"
        )
        self.confidence_text.pack()
    
    def setup_button(self):
        btn_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        btn_frame.pack(fill=tk.X)
        
        predict_btn = ttk.Button(
            btn_frame,
            text="PRÉDIRE",
            style="Accent.TButton",
            command=self.predict
        )
        predict_btn.pack(pady=10, ipadx=20)
    
    def predict(self):
        try:
            data = {}
            for key in self.entries:
                entry = self.entries[key]
                
                if isinstance(entry, ttk.Combobox):
                    # Extraire la valeur numérique entre parenthèses
                    value_str = entry.get().split("(")[-1].rstrip(")")
                    value = float(value_str)
                else:
                    value = float(entry.get())
                
                data[key] = value
            
            # Validation des transports (un seul doit être sélectionné)
            transport_keys = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
            transport_values = [data[key] for key in transport_keys]
            if sum(transport_values) != 1:
                raise ValueError("Veuillez sélectionner un seul moyen de transport")
            
            # Création du DataFrame
            columns_order = [k for _, k, _ in self.fields]
            user_df = pd.DataFrame([data], columns=columns_order)
            
            # Prédiction
            scaled_data = scaler.transform(user_df)
            prediction = model.predict(scaled_data)
            result = "OBÈSE" if prediction[0][0] > 0.5 else "NON OBÈSE"
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
            
            # Mise à jour de l'interface
            self.update_results(result, confidence)
            
        except ValueError as e:
            self.show_error(str(e))
    
    def update_results(self, result, confidence):
        if result == "OBÈSE":
            self.result_icon.config(text="⚠️", foreground=ERROR_COLOR)
            self.result_text.config(text="Résultat: OBÈSE", foreground=ERROR_COLOR)
            advice = "Consultez un professionnel de santé pour évaluer votre situation."
        else:
            self.result_icon.config(text="✓", foreground=ACCENT_COLOR)
            self.result_text.config(text="Résultat: NON OBÈSE", foreground=ACCENT_COLOR)
            advice = "Continuez vos bonnes habitudes alimentaires et physiques."
        
        self.confidence_text.config(text=f"Confiance: {confidence:.1%}\n{advice}")
    
    def show_error(self, message):
        self.result_icon.config(text="❌", foreground=ERROR_COLOR)
        self.result_text.config(text="Erreur de saisie", foreground=ERROR_COLOR)
        self.confidence_text.config(text=message)
        messagebox.showerror("Erreur", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernObesityApp(root)
    root.mainloop()