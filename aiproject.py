import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
import warnings
import sys # Import sys module to access command-line arguments
import os # Import os for checking file existence
import joblib # Import joblib for saving/loading models

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Define file paths for models ---
MODEL_FILE = 'model.joblib'
SCALER_FILE = 'scaler.joblib'

# --- Core Analysis Logic (Modified for GUI) ---

def run_analysis(file_path, logger_callback):
    """
    Runs the full data analysis and modeling pipeline.
    Replaces print() with logger_callback() and returns figures.
    """
    logger_callback("Analysis script started...")
    
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        logger_callback("Dataset loaded successfully.")
    except FileNotFoundError:
        logger_callback(f"Error: File not found at '{file_path}'.")
        return None, None
    except Exception as e:
        logger_callback(f"An error occurred while loading the file: {e}")
        return None, None
    
    original_df = df.copy()
    logger_callback(f"Original DataFrame shape: {df.shape}")

    # --- 3. Exploring the Dataset ---
    to_remove = ['world_revenue', 'opening_revenue']
    df.drop(to_remove, axis=1, inplace=True, errors='ignore')
    logger_callback(f"Dropped columns: {to_remove}")

    # --- 4. Handling Missing Values ---
    logger_callback("Handling missing values...")
    try:
        df.drop('budget', axis=1, inplace=True, errors='ignore')
        logger_callback("Dropped 'budget' column.")

        # Fill missing values efficiently
        for col in ['MPAA', 'genres']:
            if col in df.columns:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        
        df.dropna(inplace=True)
        logger_callback(f"Dropped remaining NA rows. New shape: {df.shape}")
    except Exception as e:
        logger_callback(f"An error occurred during missing value handling: {e}")
        return None, None

    # 4.1 Cleaning Numeric Columns (Optimized)
    logger_callback("Cleaning numeric columns...")
    try:
        numeric_cols = ['domestic_revenue', 'opening_theaters', 'release_days']
        
        # Vectorized string cleanup
        for col in numeric_cols:
            if df[col].dtype == 'object':
                # Remove '$' and ',' in one go
                df[col] = df[col].str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=numeric_cols, inplace=True)
        logger_callback("Numeric columns cleaned and converted.")
    except Exception as e:
        logger_callback(f"Error during numeric conversion: {e}")
        return None, None

    figures = [] # To store all generated plots

    # --- 5. Visualizing MPAA Rating Distribution ---
    logger_callback("Generating MPAA Rating Distribution plot...")
    fig1 = Figure(figsize=(10, 5), dpi=100)
    ax1 = fig1.add_subplot(111)
    sb.countplot(x=df['MPAA'], ax=ax1)
    ax1.set_title('MPAA Rating Distribution')
    figures.append(fig1)

    # --- 6. Visualizing Distributions of Key Numeric Features (Original) ---
    logger_callback("Generating original distributions of key numeric features...")
    features = ['domestic_revenue', 'opening_theaters', 'release_days']
    fig2 = Figure(figsize=(15, 5), dpi=100)
    for i, col in enumerate(features):
        ax = fig2.add_subplot(1, 3, i + 1)
        sb.distplot(df[col], ax=ax)
        ax.set_title(f'Original Distribution: {col}')
    fig2.tight_layout()
    figures.append(fig2)

    # --- 7. Detecting Outliers Using Boxplots ---
    logger_callback("Generating boxplots to check for outliers...")
    fig3 = Figure(figsize=(15, 5), dpi=100)
    for i, col in enumerate(features):
        ax = fig3.add_subplot(1, 3, i + 1)
        sb.boxplot(y=df[col], ax=ax)
        ax.set_title(f'Boxplot: {col}')
    fig3.tight_layout()
    figures.append(fig3)

    # --- 8. Applying Log Transformation (Vectorized) ---
    logger_callback("Applying vectorized log10 transformation...")
    try:
        # Optimized: Apply log10 to all feature columns at once using NumPy
        df[features] = np.log10(df[features] + 1e-6)
    except Exception as e:
        logger_callback(f"Error during log transformation: {e}")
        return None, None

    # 8.1 Checking Distributions After Log Transformation
    logger_callback("Generating distributions after log transformation...")
    fig4 = Figure(figsize=(15, 5), dpi=100)
    for i, col in enumerate(features):
        ax = fig4.add_subplot(1, 3, i + 1)
        sb.distplot(df[col], ax=ax)
        ax.set_title(f'Log-Transformed Distribution: {col}')
    fig4.tight_layout()
    figures.append(fig4)

    # --- 9. Converting Movie Genres into Numeric Features ---
    logger_callback("Converting 'genres' column using CountVectorizer...")
    try:
        vectorizer = CountVectorizer()
        vectorizer.fit(df['genres'])
        genre_features = vectorizer.transform(df['genres']).toarray()
        genre_names = vectorizer.get_feature_names_out()

        # Create a DataFrame for genres and concatenate (faster than loop)
        genre_df = pd.DataFrame(genre_features, columns=genre_names, index=df.index)
        df = pd.concat([df, genre_df], axis=1)
        df.drop('genres', axis=1, inplace=True)
        
        logger_callback("'genres' column vectorized and merged.")
    except Exception as e:
        logger_callback(f"An error occurred during genre vectorization: {e}")
        return None, None

    # 9.1 Removing Rare Genre Columns (Optimized)
    logger_callback("Removing rare genre columns...")
    
    # Identify genre columns present in the dataframe
    present_genre_cols = [col for col in genre_names if col in df.columns]
    
    if present_genre_cols:
        # Calculate mean of all genre columns at once
        genre_means = df[present_genre_cols].mean()
        # Filter columns where mean is <= 0.05 (i.e., > 95% zeros)
        cols_to_drop = genre_means[genre_means <= 0.05].index
        
        df.drop(columns=cols_to_drop, inplace=True)
        logger_callback(f"Removed {len(cols_to_drop)} rare genre columns.")

    # --- 10. Encoding Categorical Columns into Numbers ---
    logger_callback("Label encoding 'distributor' and 'MPAA' columns...")
    try:
        le = LabelEncoder()
        for col in ['distributor', 'MPAA']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])
    except Exception as e:
        logger_callback(f"Error during label encoding: {e}")
        return None, None

    # --- 11. Visualizing Strong Correlations ---
    logger_callback("Generating feature correlation heatmap (correlation > 0.8)...")
    fig5 = Figure(figsize=(10, 10), dpi=100)
    ax5 = fig5.add_subplot(111)
    numeric_df = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    sb.heatmap(correlation_matrix > 0.8, annot=True, cbar=False, cmap='viridis', ax=ax5)
    ax5.set_title('Feature Correlation Heatmap (> 0.8)')
    figures.append(fig5)

    # --- 12. Preparing Data for Model Training ---
    logger_callback("Preparing data for training and validation...")
    target_col = 'domestic_revenue'
    if target_col not in df.columns:
        logger_callback(f"Error: Target column '{target_col}' not in DataFrame. Exiting.")
        return None, None

    target = df[target_col].values
    features_df = df.drop(['title', target_col], axis=1, errors='ignore')
    features_df = features_df.select_dtypes(include=np.number) 

    X_train, X_val, Y_train, Y_val = train_test_split(features_df, target, test_size=0.1, random_state=22)
    logger_callback(f"Training features shape: {X_train.shape}")
    logger_callback(f"Validation features shape: {X_val.shape}")

    # 12.1 Normalizing Features
    logger_callback("Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    if os.path.exists(SCALER_FILE):
        logger_callback(f"Loading existing scaler from {SCALER_FILE}...")
        scaler = joblib.load(SCALER_FILE)
        X_train = scaler.transform(X_train)
    else:
        logger_callback(f"Fitting new scaler and saving to {SCALER_FILE}...")
        X_train = scaler.fit_transform(X_train)
        joblib.dump(scaler, SCALER_FILE)
    
    X_val = scaler.transform(X_val)
    logger_callback("Feature normalization complete.")

    # --- 13. Training the XGBoost Regression Model (with Hyperparameter Tuning) ---
    if os.path.exists(MODEL_FILE):
        logger_callback(f"Loading existing model from {MODEL_FILE}...")
        model = joblib.load(MODEL_FILE)
    else:
        logger_callback(f"Starting Hyperparameter Tuning with RandomizedSearchCV (this improves accuracy)...")
        logger_callback(f"This may take a moment...")
        
        # Define parameter grid for optimization
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        
        xgb_model = XGBRegressor(random_state=22, n_jobs=-1)
        
        # Use RandomizedSearchCV for efficiency over GridSearchCV
        search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=10, # Try 10 different random combinations
            cv=3,      # 3-fold cross-validation
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, Y_train)
        
        model = search.best_estimator_
        logger_callback(f"Best parameters found: {search.best_params_}")
        logger_callback(f"Model training complete. Saving best model to {MODEL_FILE}...")
        joblib.dump(model, MODEL_FILE)

    # --- 14. Evaluating Model Performance ---
    logger_callback("Evaluating model performance...")
    train_preds = model.predict(X_train)
    logger_callback(f'Training Error (MAE): {mae(Y_train, train_preds)}')
    val_preds = model.predict(X_val)
    logger_callback(f'Validation Error (MAE): {mae(Y_val, val_preds)}')
    logger_callback("\n--- Analysis Script Finished ---")
    
    return figures, original_df

# --- GUI Application Class ---

class AnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Box Office Revenue Analyzer")
        self.geometry("800x600")

        self.file_path = ""
        self.plot_widgets = []
        self.original_df = None # To store the loaded data for Q&A

        self.setup_gui()

    def setup_gui(self):
        # --- Main Tab Control ---
        self.notebook = ttk.Notebook(self)
        
        # --- Tab 1: Analysis Control ---
        self.analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_tab, text='Run Analysis')
        
        control_frame = ttk.Frame(self.analysis_tab)
        control_frame.pack(fill='x', pady=5)

        self.select_btn = ttk.Button(control_frame, text="Select boxoffice.csv", command=self.select_file)
        self.select_btn.pack(side='left', padx=5)

        self.file_label = ttk.Label(control_frame, text="No file selected.")
        self.file_label.pack(side='left', padx=5)

        self.run_btn = ttk.Button(control_frame, text="Run Analysis", command=self.start_analysis_thread, state='disabled')
        self.run_btn.pack(side='right', padx=5)

        self.console = scrolledtext.ScrolledText(self.analysis_tab, wrap=tk.WORD, height=10, bg="#f0f0f0", fg="black")
        self.console.pack(expand=True, fill='both', pady=5)
        self.console.insert(tk.END, "Welcome! Please select the boxoffice.csv file and click 'Run Analysis'.\n")
        self.console.config(state='disabled')

        # --- Tab 2: Plots ---
        self.plot_tab = ttk.Frame(self.notebook, padding="10")
        
        # We will add a canvas for scrolling
        self.plot_canvas = tk.Canvas(self.plot_tab)
        self.plot_scrollbar = ttk.Scrollbar(self.plot_tab, orient="vertical", command=self.plot_canvas.yview)
        self.plot_scrollable_frame = ttk.Frame(self.plot_canvas)

        self.plot_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(
                scrollregion=self.plot_canvas.bbox("all")
            )
        )

        self.plot_canvas.create_window((0, 0), window=self.plot_scrollable_frame, anchor="nw")
        self.plot_canvas.configure(yscrollcommand=self.plot_scrollbar.set)

        self.plot_canvas.pack(side="left", fill="both", expand=True)
        self.plot_scrollbar.pack(side="right", fill="y")
        
        self.notebook.add(self.plot_tab, text='Plots', state='disabled')

        # --- Tab 3: Data Q&A ---
        self.qa_tab = ttk.Frame(self.notebook, padding="10")
        
        # Add a console for answers
        self.qa_console = scrolledtext.ScrolledText(self.qa_tab, wrap=tk.WORD, height=10, bg="#f0f0f0", fg="black")
        self.qa_console.pack(expand=True, fill='both', pady=5)
        self.qa_console.insert(tk.END, "Run analysis on the 'Run Analysis' tab to enable questions.\n")
        self.qa_console.config(state='disabled')
        
        # Add a frame for question buttons
        question_frame = ttk.Frame(self.qa_tab)
        question_frame.pack(fill='x', pady=5)

        # Define questions and their corresponding handler functions
        questions = [
            ("Show data summary (df.describe())", self.answer_q1),
            ("What are the top 10 distributors by movie count?", self.answer_q2),
            ("What is the average domestic revenue?", self.answer_q3),
            ("What is the movie count by MPAA rating?", self.answer_q4),
            ("What are the top 10 most common genres?", self.answer_q5),
            ("Which movie had the highest domestic revenue?", self.answer_q6),
            ("What is the average number of opening theaters?", self.answer_q7),
            ("How many movies are in the dataset?", self.answer_q8)
        ]

        # Create buttons for each question
        for text, command in questions:
            btn = ttk.Button(question_frame, text=text, command=command)
            btn.pack(fill='x', pady=2)

        self.notebook.add(self.qa_tab, text='Data Q&A', state='disabled')

        self.notebook.pack(expand=True, fill='both')

    def select_file(self):
        # Set a default path to make it easier for the user
        default_path = r'C:\Users\mahar\OneDrive\Documents\AI-Proj\boxoffice.csv'
        if os.path.exists(default_path):
            initial_dir = os.path.dirname(default_path)
        else:
            initial_dir = os.path.expanduser('~') # Home directory as fallback

        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select boxoffice.csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(self.file_path))
            self.run_btn.config(state='normal')

    def log_to_console(self, message):
        """Thread-safe way to add messages to the console."""
        self.console.config(state='normal')
        self.console.insert(tk.END, str(message) + "\n")
        self.console.see(tk.END)
        self.console.config(state='disabled')

    def start_analysis_thread(self):
        if not self.file_path:
            self.log_to_console("Error: No file selected.")
            return
        
        self.run_btn.config(state='disabled')
        self.select_btn.config(state='disabled')
        self.notebook.tab(self.plot_tab, state='disabled')
        self.notebook.tab(self.qa_tab, state='disabled') # Disable Q&A tab
        
        # Clear previous plots
        for widget in self.plot_scrollable_frame.winfo_children():
            widget.destroy()
        self.plot_widgets = []
        
        # Clear Q&A console
        self.log_to_qa_console("Running analysis... Please wait.", clear=True)
        
        self.log_to_console("Starting analysis thread...")
        
        analysis_thread = threading.Thread(
            target=self.run_analysis_in_thread,
            daemon=True
        )
        analysis_thread.start()

    def run_analysis_in_thread(self):
        """Runs the analysis, then schedules GUI updates."""
        try:
            figures, original_df = run_analysis(self.file_path, self.log_to_console)
            
            if figures and original_df is not None:
                self.original_df = original_df # Store the dataframe
                # Schedule plot display in main thread
                self.after(0, self.display_plots, figures)
                # Schedule Q&A tab enabling in main thread
                self.after(0, self.enable_qa_tab)
            else:
                self.log_to_console("Analysis failed. See log for details.")
                self.log_to_qa_console("Analysis failed. Cannot load Q&A.", clear=True)

        except Exception as e:
            self.log_to_console(f"--- FATAL THREAD ERROR ---")
            self.log_to_console(f"{e}")
        finally:
            # Schedule button re-enabling in main thread
            self.after(0, lambda: [
                self.run_btn.config(state='normal'),
                self.select_btn.config(state='normal')
            ])
            
    def display_plots(self, figures):
        """Embeds matplotlib figures in the Plots tab. Runs in main thread."""
        self.log_to_console("Displaying plots in 'Plots' tab...")
        for fig in figures:
            canvas = FigureCanvasTkAgg(fig, master=self.plot_scrollable_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(pady=10, fill='x', expand=True)
            self.plot_widgets.append(widget) # Keep reference
        
        self.notebook.tab(self.plot_tab, state='normal')
        self.notebook.select(self.plot_tab)

    # --- Q&A Tab Functions ---

    def log_to_qa_console(self, message, clear=False):
        """Thread-safe way to add messages to the Q&A console."""
        self.qa_console.config(state='normal')
        if clear:
            self.qa_console.delete('1.0', tk.END)
        self.qa_console.insert(tk.END, str(message) + "\n\n")
        self.qa_console.see(tk.END)
        self.qa_console.config(state='disabled')

    def enable_qa_tab(self):
        """Enables the Q&A tab after analysis is complete."""
        self.notebook.tab(self.qa_tab, state='normal')
        self.log_to_qa_console("Dataset is loaded. You can now ask questions.", clear=True)

    def answer_q1(self):
        """Q1: Show data summary (df.describe())"""
        if self.original_df is not None:
            try:
                answer = self.original_df.describe().to_string()
                self.log_to_qa_console("--- Data Summary (df.describe()) ---\n" + answer)
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q1: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")

    def answer_q2(self):
        """Q2: What are the top 10 distributors by movie count?"""
        if self.original_df is not None:
            try:
                answer = self.original_df['distributor'].value_counts().head(10).to_string()
                self.log_to_qa_console("--- Top 10 Distributors by Movie Count ---\n" + answer)
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q2: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")

    def answer_q3(self):
        """Q3: What is the average domestic revenue?"""
        if self.original_df is not None:
            try:
                # Need to clean the original revenue column just for this calculation
                cleaned_revenue = pd.to_numeric(
                    self.original_df['domestic_revenue'].astype(str).str.replace('$', '').str.replace(',', ''), 
                    errors='coerce'
                )
                avg_revenue = cleaned_revenue.mean()
                self.log_to_qa_console(f"--- Average Domestic Revenue ---\n${avg_revenue:,.2f}")
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q3: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")

    def answer_q4(self):
        """Q4: What is the movie count by MPAA rating?"""
        if self.original_df is not None:
            try:
                answer = self.original_df['MPAA'].value_counts().to_string()
                self.log_to_qa_console("--- Movie Count by MPAA Rating ---\n" + answer)
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q4: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")
    
    def answer_q5(self):
        """Q5: What are the top 10 most common genres?"""
        if self.original_df is not None:
            try:
                answer = self.original_df['genres'].value_counts().head(10).to_string()
                self.log_to_qa_console("--- Top 10 Most Common Genres ---\n" + answer)
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q5: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")

    def answer_q6(self):
        """Q6: Which movie had the highest domestic revenue?"""
        if self.original_df is not None:
            try:
                # Create a temporary copy for cleaning
                temp_df = self.original_df.copy()
                temp_df['cleaned_revenue'] = pd.to_numeric(
                    temp_df['domestic_revenue'].astype(str).str.replace('$', '').str.replace(',', ''), 
                    errors='coerce'
                )
                temp_df = temp_df.dropna(subset=['cleaned_revenue'])
                
                # Find the movie with the max revenue
                idx_max = temp_df['cleaned_revenue'].idxmax()
                movie = temp_df.loc[idx_max]
                
                answer = f"Movie: {movie['title']}\nRevenue: ${movie['cleaned_revenue']:,.2f}"
                self.log_to_qa_console("--- Movie with Highest Domestic Revenue ---\n" + answer)
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q6: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")

    def answer_q7(self):
        """Q7: What is the average number of opening theaters?"""
        if self.original_df is not None:
            try:
                # Need to clean the original theaters column just for this calculation
                cleaned_theaters = pd.to_numeric(
                    self.original_df['opening_theaters'].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
                avg_theaters = cleaned_theaters.mean()
                self.log_to_qa_console(f"--- Average Opening Theaters ---\n{avg_theaters:,.0f} theaters")
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q7: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")

    def answer_q8(self):
        """Q8: How many movies are in the dataset?"""
        if self.original_df is not None:
            try:
                count = len(self.original_df)
                self.log_to_qa_console(f"--- Total Movie Count ---\nThere are {count} movies in the loaded dataset.")
            except Exception as e:
                self.log_to_qa_console(f"Error processing Q8: {e}")
        else:
            self.log_to_qa_console("Error: Data not loaded. Please run analysis again.")


if __name__ == "__main__":
    app = AnalysisApp()
    app.mainloop()