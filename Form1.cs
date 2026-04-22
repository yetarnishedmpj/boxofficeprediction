using System;
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using ScottPlot;
using ScottPlot.WinForms;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;

namespace BoxOfficeAnalyzer
{
    public partial class Form1 : Form
    {
        private TabControl tabControl;
        private TabPage analysisTab;
        private TabPage plotTab;
        private TabPage qaTab;

        private Button selectFileBtn;
        private Button runBtn;
        private Label fileLabel;
        private RichTextBox console;

        private FlowLayoutPanel plotPanel;
        private RichTextBox qaConsole;
        private FlowLayoutPanel qaButtonPanel;

        private string filePath = "";
        private DataFrame originalDf;
        private MLContext mlContext = new MLContext(seed: 22);
        private ITransformer model;
        private DataViewSchema modelSchema;

        public Form1()
        {
            InitializeComponent();
            SetupGui();
        }

        private void SetupGui()
        {
            this.Text = "Box Office Revenue Analyzer (C#)";
            this.Size = new Size(1000, 800);

            tabControl = new TabControl { Dock = DockStyle.Fill };
            this.Controls.Add(tabControl);

            // --- Analysis Tab ---
            analysisTab = new TabPage("Run Analysis");
            tabControl.TabPages.Add(analysisTab);

            Panel controlPanel = new Panel { Dock = DockStyle.Top, Height = 50 };
            analysisTab.Controls.Add(controlPanel);

            selectFileBtn = new Button { Text = "Select boxoffice.csv", Location = new Point(10, 10), Width = 150 };
            selectFileBtn.Click += SelectFileBtn_Click;
            controlPanel.Controls.Add(selectFileBtn);

            fileLabel = new Label { Text = "No file selected.", Location = new Point(170, 15), AutoSize = true };
            controlPanel.Controls.Add(fileLabel);

            runBtn = new Button { Text = "Run Analysis", Location = new Point(850, 10), Width = 100, Enabled = false };
            runBtn.Click += RunBtn_Click;
            controlPanel.Controls.Add(runBtn);

            console = new RichTextBox { Dock = DockStyle.Fill, ReadOnly = true, BackColor = Color.FromArgb(240, 240, 240) };
            analysisTab.Controls.Add(console);
            console.BringToFront();

            // --- Plot Tab ---
            plotTab = new TabPage("Plots") { Enabled = false };
            tabControl.TabPages.Add(plotTab);

            plotPanel = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoScroll = true };
            plotTab.Controls.Add(plotPanel);

            // --- QA Tab ---
            qaTab = new TabPage("Data Q&A") { Enabled = false };
            tabControl.TabPages.Add(qaTab);

            qaConsole = new RichTextBox { Dock = DockStyle.Top, Height = 300, ReadOnly = true, BackColor = Color.FromArgb(240, 240, 240) };
            qaTab.Controls.Add(qaConsole);

            qaButtonPanel = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoScroll = true };
            qaTab.Controls.Add(qaButtonPanel);

            AddQAButtons();
        }

        private void AddQAButtons()
        {
            string[] questions = {
                "Show data summary",
                "Top 10 distributors",
                "Average domestic revenue",
                "Movie count by MPAA rating",
                "Top 10 most common genres",
                "Highest revenue movie",
                "Average opening theaters",
                "Total movie count"
            };

            foreach (var q in questions)
            {
                Button btn = new Button { Text = q, Width = 300, Height = 40 };
                btn.Click += (s, e) => HandleQA(q);
                qaButtonPanel.Controls.Add(btn);
            }
        }

        private void SelectFileBtn_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*";
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    filePath = openFileDialog.FileName;
                    fileLabel.Text = Path.GetFileName(filePath);
                    runBtn.Enabled = true;
                }
            }
        }

        private async void RunBtn_Click(object sender, EventArgs e)
        {
            runBtn.Enabled = false;
            selectFileBtn.Enabled = false;
            plotTab.Enabled = false;
            qaTab.Enabled = false;

            console.Clear();
            plotPanel.Controls.Clear();
            Log("Starting analysis...");

            await Task.Run(() => RunAnalysis());

            runBtn.Enabled = true;
            selectFileBtn.Enabled = true;
        }

        private void Log(string message)
        {
            if (console.InvokeRequired)
            {
                console.Invoke(new Action(() => Log(message)));
                return;
            }
            console.AppendText(message + Environment.NewLine);
            console.SelectionStart = console.Text.Length;
            console.ScrollToCaret();
        }

        private void RunAnalysis()
        {
            try
            {
                Log("Loading dataset...");
                // Load with CsvHelper to handle encoding and potentially messy CSVs
                var config = new CsvConfiguration(CultureInfo.InvariantCulture)
                {
                    HasHeaderRecord = true,
                    BadDataFound = null,
                };

                originalDf = DataFrame.LoadCsv(filePath);
                Log($"Dataset loaded. Shape: {originalDf.Rows.Count} rows, {originalDf.Columns.Count} columns.");

                var df = originalDf.Clone();

                // Preprocessing
                Log("Handling missing values and cleaning columns...");
                
                // Drop columns
                string[] toRemove = { "world_revenue", "opening_revenue", "budget" };
                foreach (var col in toRemove)
                {
                    if (df.Columns.Any(c => c.Name == col))
                        df.Columns.Remove(col);
                }

                // Clean numeric columns
                string[] numericCols = { "domestic_revenue", "opening_theaters", "release_days" };
                foreach (var colName in numericCols)
                {
                    var col = df.Columns[colName];
                    if (col.DataType == typeof(string))
                    {
                        var stringCol = (StringDataFrameColumn)col;
                        var newCol = new SingleDataFrameColumn(colName, col.Length);
                        for (long i = 0; i < col.Length; i++)
                        {
                            string val = stringCol[i];
                            if (val != null)
                            {
                                string cleaned = val.Replace("$", "").Replace(",", "");
                                if (float.TryParse(cleaned, out float result))
                                    newCol[i] = result;
                                else
                                    newCol[i] = null;
                            }
                        }
                        df.Columns.Remove(colName);
                        df.Columns.Add(newCol);
                    }
                }

                // Drop NAs
                df = df.Filter(df.Rows.Select(r => !numericCols.Any(c => df.Columns[c][r.Index] == null)));
                Log($"Cleaned data shape: {df.Rows.Count} rows.");

                // Visualize (ScottPlot)
                this.Invoke(new Action(() => {
                    GeneratePlots(df);
                    plotTab.Enabled = true;
                    qaTab.Enabled = true;
                }));

                // Log transformation
                Log("Applying log transformation...");
                foreach (var colName in numericCols)
                {
                    var col = (SingleDataFrameColumn)df.Columns[colName];
                    for (long i = 0; i < col.Length; i++)
                    {
                        if (col[i].HasValue)
                            col[i] = (float)Math.Log10(col[i].Value + 1e-6);
                    }
                }

                // ML.NET Training
                Log("Training ML model...");
                TrainModel(df);

                Log("Analysis finished.");
            }
            catch (Exception ex)
            {
                Log("Error: " + ex.Message);
            }
        }

        private void GeneratePlots(DataFrame df)
        {
            // Plot 1: Domestic Revenue Distribution
            var revenueCol = (SingleDataFrameColumn)df.Columns["domestic_revenue"];
            double[] revenues = revenueCol.Cast<float?>().Where(v => v.HasValue).Select(v => (double)v.Value).ToArray();

            FormsPlot plot1 = new FormsPlot { Width = 450, Height = 300 };
            plot1.Plot.Add.Histogram(ScottPlot.Statistics.Histogram.CreateWithBinCount(revenues, 20));
            plot1.Plot.Title("Domestic Revenue Distribution");
            plotPanel.Controls.Add(plot1);

            // Plot 2: Opening Theaters vs Revenue
            var theatersCol = (SingleDataFrameColumn)df.Columns["opening_theaters"];
            double[] theaters = theatersCol.Cast<float?>().Where(v => v.HasValue).Select(v => (double)v.Value).ToArray();
            
            FormsPlot plot2 = new FormsPlot { Width = 450, Height = 300 };
            plot2.Plot.Add.Scatter(theaters, revenues);
            plot2.Plot.Title("Theaters vs Revenue");
            plotPanel.Controls.Add(plot2);
        }

        private void TrainModel(DataFrame df)
        {
            // Simple ML.NET training
            // Convert DataFrame to IDataView
            var dataView = mlContext.Data.LoadFromEnumerable(ToMovieData(df));
            
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "DomesticRevenue")
                .Append(mlContext.Transforms.Concatenate("Features", "OpeningTheaters", "ReleaseDays"))
                .Append(mlContext.Regression.Trainers.FastTree());

            model = pipeline.Fit(dataView);
            
            // Evaluate
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions);
            Log($"Training MAE: {metrics.MeanAbsoluteError}");
        }

        private IEnumerable<MovieData> ToMovieData(DataFrame df)
        {
            for (long i = 0; i < df.Rows.Count; i++)
            {
                yield return new MovieData
                {
                    DomesticRevenue = (float)df.Columns["domestic_revenue"][i],
                    OpeningTheaters = (float)df.Columns["opening_theaters"][i],
                    ReleaseDays = (float)df.Columns["release_days"][i]
                };
            }
        }

        private void HandleQA(string question)
        {
            qaConsole.AppendText($"Q: {question}\n");
            string answer = "";
            try
            {
                switch (question)
                {
                    case "Show data summary":
                        answer = originalDf.ToString();
                        break;
                    case "Total movie count":
                        answer = $"There are {originalDf.Rows.Count} movies.";
                        break;
                    case "Average domestic revenue":
                        var col = originalDf.Columns["domestic_revenue"];
                        // Need cleaning for original
                        double sum = 0; int count = 0;
                        for (long i = 0; i < col.Length; i++) {
                            string s = col[i]?.ToString().Replace("$", "").Replace(",", "");
                            if (double.TryParse(s, out double d)) { sum += d; count++; }
                        }
                        answer = $"Average: ${(sum / count):N2}";
                        break;
                    default:
                        answer = "Feature coming soon in C# version!";
                        break;
                }
            }
            catch (Exception ex) { answer = "Error: " + ex.Message; }
            
            qaConsole.AppendText($"A: {answer}\n\n");
            qaConsole.ScrollToCaret();
        }
    }

    public class MovieData
    {
        public float DomesticRevenue { get; set; }
        public float OpeningTheaters { get; set; }
        public float ReleaseDays { get; set; }
    }
}
