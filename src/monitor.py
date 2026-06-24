import pandas as pd
import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

def generate_drift_report():
    console.clear()
    console.print(Panel.fit("[bold cyan]🚀 Enterprise MLOps Drift Watchdog[/bold cyan]", border_style="cyan"))
    
    # Adding a cinematic progress bar for data ingestion
    for _ in track(range(100), description="[green]Ingesting live production data streams..."):
        time.sleep(0.01)
        
    # 1. Load the data
    baseline_data = pd.read_csv("data/baseline_training.csv")
    prod_data = pd.read_csv("data/production_simulation.csv")
    
    # 2. The financial features we want to monitor for economic crashes
    features_to_monitor = ['income', 'credit_score', 'age']
    
    # Create a stunning Rich Table
    table = Table(title="📈 Live Statistical Drift Analysis", show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan", width=15)
    table.add_column("Baseline Avg", justify="right", style="green")
    table.add_column("Live Avg", justify="right", style="yellow")
    table.add_column("Shift (%)", justify="right")
    table.add_column("Status", justify="center")
    
    drift_detected = False
    
    # 3. Mathematically compare the averages of the old data vs the new data
    for feature in features_to_monitor:
        base_mean = baseline_data[feature].mean()
        prod_mean = prod_data[feature].mean()
        
        # Calculate percentage shift
        shift_pct = abs(prod_mean - base_mean) / base_mean * 100
        
        status = "[bold green]✅ Stable[/bold green]"
        shift_text = f"[green]{shift_pct:.1f}%[/green]"
        
        if shift_pct > 10.0:  # If the average changes by more than 10%, flag it!
            status = "[bold red]⚠️ DRIFTED[/bold red]"
            shift_text = f"[bold red]{shift_pct:.1f}%[/bold red]"
            drift_detected = True
            
        # Add the row to our beautiful table
        table.add_row(
            feature.upper(),
            f"{base_mean:.2f}",
            f"{prod_mean:.2f}",
            shift_text,
            status
        )
        
    console.print(table)
    console.print("\n")
        
    if drift_detected:
        console.print(Panel("[bold red]🚨 MAJOR ECONOMIC DRIFT DETECTED![/bold red]\n[white]The live data environment has shifted dramatically.\nACTION REQUIRED: Retrain the Random Forest model on the latest data immediately.[/white]", border_style="red"))
    else:
        console.print(Panel("[bold green]✅ ALL SYSTEMS NORMAL[/bold green]\n[white]Live production data matches the training distribution. The model is safe.[/white]", border_style="green"))

if __name__ == "__main__":
    generate_drift_report()