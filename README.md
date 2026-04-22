# Box Office Revenue Analyzer (C#)

A powerful desktop application built with C# WinForms to analyze and predict box office revenues using machine learning. This project is a C# port of the original Python-based analyzer.

## Features

- **Data Analysis**: Load and process movie datasets (`boxoffice.csv`).
- **Data Visualization**: Interactive plots using **ScottPlot** showing revenue distributions and correlations.
- **Machine Learning**: Predicts domestic revenue using **ML.NET** (Regression/FastTree).
- **Data Q&A**: Quick access to statistical summaries, top distributors, and revenue trends.
- **Modern UI**: Clean Tabbed interface for Analysis, Visualization, and Q&A.

## Tech Stack

- **Framework**: .NET 8.0 WinForms
- **Data Processing**: [Microsoft.Data.Analysis](https://www.nuget.org/packages/Microsoft.Data.Analysis)
- **Machine Learning**: [ML.NET](https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet)
- **Plotting**: [ScottPlot](https://scottplot.net/)
- **CSV Parsing**: [CsvHelper](https://joshclose.github.io/CsvHelper/)

## Getting Started

### Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Visual Studio 2022 (optional, but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yetarnishedmpj/boxofficeprediction.git
   cd boxofficeprediction
   ```

2. Restore dependencies:
   ```bash
   dotnet restore
   ```

3. Run the application:
   ```bash
   dotnet run
   ```

## How to Use

1. Launch the application.
2. Click **"Select boxoffice.csv"** to load your dataset.
3. Click **"Run Analysis"** to start the preprocessing and ML training pipeline.
4. Navigate to the **"Plots"** tab to view data visualizations.
5. Use the **"Data Q&A"** tab to ask specific questions about the dataset.

## License

This project is open-source and available under the MIT License.
