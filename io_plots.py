"""Import/Exports Visualization Module

This module provides several functionality for visualization of
maritime trade flows.
"""

__version__ = "1.0.0"

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os.path


def save_fig(
    img_name: str, img_dir: str = '.', 
    exts = ('.pdf', '.png'), force: bool = False) -> None:
    """Save current matplotlib figure to multiple file formats.
    
    This function saves the current matplotlib figure to one or more file formats
    in the specified directory. It includes protection against accidentally
    overwriting existing files unless explicitly forced.
    
    Args:
        img_dir (str): Directory path where the image files will be saved.
            Should include trailing slash if not part of img_name.
        img_name (str): Base name for the image file without extension.
            The function will append each extension from exts to create
            the full filename.
        exts (tuple of str, optional): File extensions to save the figure in.
            Each extension should include the leading dot (e.g., '.pdf', '.png').
            Defaults to ('.pdf', '.png').
        force (bool, optional): If True, overwrites existing files without
            prompting. If False, checks for existing files and skips saving
            if the first format already exists. Defaults to False.
            
    Returns:
        None
        
    Note:
        - Only checks for existence of the first file format in exts tuple
        - If the first format exists and force=False, no files are saved
        - Uses matplotlib's current active figure for saving
        - Directory must exist before calling this function
    """
    full_name = img_dir + img_name 
    if os.path.isfile(full_name + exts[0]):
        if not force:
            print(f"image {full_name + exts[0]} exists, use force=True to replace")
            return
    for ext in exts:
        plt.savefig(full_name + ext)

def plot_sankey(day_exports_all_trunc_OO_df, title="Flux Sankey exportateurs → importateurs (Coal)"):
    """Create a Sankey diagram for maritime trade flow visualization.
    
    This function generates an interactive Sankey diagram showing trade flows
    between exporting countries (sources) and importing countries (targets).
    The diagram visualizes the volume of trade flows, with flow thickness
    proportional to trade volume.
    
    Args:
        day_exports_all_trunc_OO_df (pandas.DataFrame): Trade flow matrix where:
            - Rows represent exporting countries (sources)
            - Columns represent importing countries (targets)  
            - Values represent trade volumes (e.g., tonnes of coal)
            - Zero values are filtered out from the visualization
        title (str, optional): Title for the Sankey diagram. 
            Defaults to "Flux Sankey exportateurs → importateurs (Coal)".
            
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure object containing
            the Sankey diagram. Can be further customized or saved.
            
    Note:
        - Only positive trade flow values are included in the diagram
        - The function automatically displays the figure using fig.show()
        - Node labels are derived from DataFrame index (exporters) and 
          columns (importers)
        - Link thickness is proportional to trade volume values
        
    Example:
        >>> import pandas as pd
        >>> # Assume we have a trade matrix
        >>> trade_matrix = pd.DataFrame({
        ...     'China': [100, 50, 0],
        ...     'India': [75, 0, 25],
        ...     'Japan': [30, 20, 10]
        ... }, index=['Australia', 'Indonesia', 'Russia'])
        >>> fig = plot_sankey(trade_matrix, "Coal Trade Flows 2023")
        >>> # Figure is displayed and returned for further use
    """    
    # diagramme sankey pour la matrice reduite
    
    # Étape 1 : construire les listes source, target, value
    sources_2 = []
    targets_2 = []
    values_2 = []
    #colors_2 =[]
    source_labels_2 = list(day_exports_all_trunc_OO_df.index)
    target_labels_2 = list(day_exports_all_trunc_OO_df.columns)
    all_labels_2 = source_labels_2 + target_labels_2
    # dictionnaire pour retrouver les indices pour le diagramme sankey
    label_indices_2 = {label: i for i, label in enumerate(all_labels_2)}
    for source in source_labels_2:
        for target in target_labels_2:
            value = day_exports_all_trunc_OO_df.loc[source, target]
            if value > 0:
                sources_2.append(label_indices_2[source])
                targets_2.append(label_indices_2[target])
                values_2.append(value)
                #colors_2.append(source_colors_2[source])
    # Étape 2 : créer le diagramme
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            #line=dict(color="black", width=0.5),
            label=all_labels_2
        ),
        link=dict(
            source=sources_2,
            target=targets_2,
            value=values_2,
            #color=colors_2
        )
    )])
    fig.update_layout(title_text=title, font_size=10)
    fig.show()

    return fig
