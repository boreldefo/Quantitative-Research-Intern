"""Import/Exports Loeading and formating Module

This module provides functionality for loading
maritime trade flows.
"""

from glob import glob
import pandas 
import numpy as np
from tqdm.notebook import tqdm
from collections import namedtuple
import warnings
import os
import re

__version__ = "1.0"

FSCC = namedtuple('FSC', ['file', 'name', 'shape', 'columns', 'content'])

class FileContainer:
    """Container class for managing loaded file data.
    
    This class provides a centralized way to store and access file data
    loaded from Excel files containing maritime traffic information.
    
    Attributes:
        content (list): Class-level list storing FSCC namedtuples with file data.
    """
    content = []
    
    @classmethod
    def get(cls, name=None):
        """Retrieve file data by name or get all file names.
        
        Args:
            name (str, optional): Name of the file to retrieve. If None, returns
                all available file names.
                
        Returns:
            list or FSCC or None: If name is None, returns list of all file names.
                If name is provided, returns the corresponding FSCC object or None
                if not found.
        """
        if name is None:
            return [f.name for f in cls.content]
        else:
            for f in cls.content:
                if f.name == name:
                    return f
            return None
            
    @classmethod
    def pprint(cls):
        """Pretty print all loaded files with their column information.
        
        Returns:
            str: Formatted string showing file names and their columns.
        """
        txt_lst = []
        for f in files_content_lst:
            txt_lst.append(f">>> {f.name}:\n {f.columns.values}")
        return "\n".join(txt_lst)
        
    @classmethod
    def iterate(cls):
        """Iterator over all loaded files.
        
        Yields:
            FSCC: Each loaded file's data structure.
        """
        for f in files_content_lst:
            yield(f)
            
    @classmethod
    def update(cls, content):
        """Update the content with new file data.
        
        Args:
            content (list): List of FSCC namedtuples to store as class content.
        """
        cls.content = content


def load_one_xls_dir(dir_name):
    """Load all Excel files from a specified directory.
    
    This function reads all .xlsx files from a given directory and creates
    FSCC namedtuples containing file metadata and pandas DataFrame content.
    
    Args:
        dir_name (str): Path to the directory containing Excel files.
        
    Returns:
        list: List of FSCC namedtuples, each containing:
            - file: Full file path
            - name: Base filename without extension
            - shape: DataFrame shape tuple (rows, columns)
            - columns: DataFrame column names
            - content: pandas DataFrame with the file data
            
    Note:
        Suppresses openpyxl UserWarnings about stylesheets to reduce noise
        during batch processing.
    """
    files_content_lst = []
    print(f"loading from {dir_name}...")
    for fname in tqdm(glob(dir_name + '/*.xlsx')):
        # Replace 'your_file.xlsx' with the path to your Excel file
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=re.escape('openpyxl.styles.stylesheet'))
            df = pandas.read_excel(fname, engine='openpyxl')
        files_content_lst.append(
            FSCC(fname, os.path.basename(fname).split('.')[0], df.shape, df.columns, df)
        )    
    print(f"...{len(files_content_lst)} files loaded.")
    return files_content_lst


class DataConfigurator:
    """Main class for configuring and processing maritime traffic data.
    
    This class handles loading, organizing, and preprocessing maritime trade data
    from multiple Excel files, specifically focusing on coal import/export patterns
    between different countries.
    
    Attributes:
        dirs (dict): Dictionary mapping data type names to directory paths.
        all_content (list): List of all loaded file data.
        countries_export (dict): Mapping of export countries to their filename patterns.
        countries_import (dict): Mapping of import countries to their filename patterns.
        data_export_to (dict): Processed export data by destination country.
        data_import_in (dict): Processed import data by source country.
        export_to_countries (list): List of countries that export to others.
        import_in_countries (list): List of countries that import from others.
    """

    def __init__(self, dirs={
        'export_coal'    : './data_in/shipfix/export_coal',
        'import_coal'    : './data_in/shipfix/import_coal',
    }):
        """Initialize DataConfigurator with directory paths.
        
        Args:
            dirs (dict): Dictionary mapping data categories to their directory paths.
                Default includes coal export and import directories.
        """
        self.dirs = dirs


    def load_all(self):
        """Load all Excel files from configured directories.
        
        Iterates through all configured directories and loads Excel files,
        storing the combined results in self.all_content.
        
        Note:
            This method must be called before other processing methods.
        """
        files_content_lst = []

        for name, dir_name in self.dirs.items():
            files_content_lst = files_content_lst + load_one_xls_dir(dir_name)
        
        self.all_content =  files_content_lst

    def define_import_export(
        self, 
        countries_export={ # _to
            'Indonesia': 'all_voyage_orders_by_cargo_type_time_series_stacked_Indonesia_to',
            'Australia': 'all_voyage_orders_by_cargo_type_time_series_stacked_Australia_to',
            'SAFR'     : 'all_voyage_orders_by_cargo_type_time_series_stacked_SAFR_to',
            'US'       : 'all_voyage_orders_by_cargo_type_time_series_stacked_US_to',
            'Russia'   : 'all_voyage_orders_by_cargo_type_time_series_stacked_Russia_to',
        },
        countries_import={ # _in
            'India'    : 'all_voyage_orders_by_cargo_type_time_series_stacked_to_India',
            'China'    : 'all_voyage_orders_by_cargo_type_time_series_stacked_to_China',
            'Vietnam'  : 'all_voyage_orders_by_cargo_type_time_series_stacked_to_Vietnam',
            'Japan'    : 'all_voyage_orders_by_cargo_type_time_series_stacked_to_Japan',
            'South Korea' : 'all_voyage_orders_by_cargo_type_time_series_stacked_to_SK',
        }
    ):
        """Define country mappings and organize import/export data.
        
        This method establishes the relationship between countries and their
        corresponding data files, then organizes the data for further processing.
        
        Args:
            countries_export (dict): Mapping of exporting countries to their
                corresponding filename patterns. Keys are country names, values
                are filename identifiers for files containing export data.
            countries_import (dict): Mapping of importing countries to their
                corresponding filename patterns. Keys are country names, values
                are filename identifiers for files containing import data.
                
        Note:
            Requires load_all() to be called first. Updates the FileContainer
            with loaded content and creates data structures for export and import
            analysis.
        """
        self.countries_export = countries_export
        self.countries_import = countries_import
        # I will use the FileContainer
        FileContainer.update(self.all_content)

        self.data_export_to = {c: FileContainer.get(fname).content for c, fname in self.countries_export.items()}
        self.data_import_in = {c: FileContainer.get(fname).content for c, fname in self.countries_import.items()}

        # reminder:
        self.export_to_countries = list(self.countries_export.keys()) #+ ['Others']
        self.import_in_countries = list(self.countries_import.keys()) #+ ['Others']

        print("Done:")
        print(f"... export to  {self.export_to_countries}")
        print(f"... import in {self.import_in_countries}")


    def export_on(self, day='2021-03-10'):
        """Extract and process export/import data for a specific date.
        
        This method is marked as deprecated and extracts trade flow data
        for a single day, creating balanced import/export matrices.
        
        Args:
            day (str): Date in 'YYYY-MM-DD' format for data extraction.
                Defaults to '2021-03-10'.
                
        Returns:
            tuple: Contains three elements:
                - pandas.DataFrame: Balanced export matrix with 'Others' category
                - dict: Raw daily export data by destination country
                - dict: Raw daily import data by source country
                
        Note:
            This method is deprecated. The 'Others' category represents
            aggregated data for countries not explicitly tracked.
        """
        day_exports_to = {c: df[df['Date'] == day].iloc[0, 1:] for c, df in self.data_export_to.items()}
        day_imports_in = {c: df[df['Date'] == day].iloc[0, 1:] for c, df in self.data_import_in.items()}

        # known export_to
        day_imports_in_df = pandas.DataFrame(day_imports_in)
        # known import to
        day_exports_to_df = pandas.DataFrame(day_exports_to).T

        # all the rest (column)
        other_exports = list(set(day_exports_to_df.columns) - set(self.import_in_countries))
        
        others_ser = day_exports_to_df[other_exports].sum(axis=1).rename('Others')
        
        day_exports_to_trunc_df = pandas.concat([
            day_exports_to_df[day_exports_to_df.columns.intersection(self.import_in_countries)],
            others_ser
        ], axis=1)

        # all the rest (row)
        other_imports = list(set(day_imports_in_df.columns) - set(self.export_to_countries))
        
        others_ser = (
            day_imports_in_df[other_imports].sum(axis=0) - day_exports_to_trunc_df.sum(axis=0)
        ).rename('Others')
        
        day_exports_all_trunc_df = pandas.concat([
                day_exports_to_trunc_df,
                pandas.DataFrame(others_ser).T
            ], axis=0).fillna(0) 
        
        return day_exports_all_trunc_df, day_exports_to, day_imports_in   
     

    def reduce_exports(self, country, df):
        """Transform export data for a single country into time-indexed format.
        
        This method processes export data from one country to create a time-series
        DataFrame with MultiIndex columns showing destination countries. Countries
        not in the main import list are aggregated into an 'Others' category.
        
        Args:
            country (str): Name of the exporting country.
            df (pandas.DataFrame): Raw export data with 'Date' column and
                destination countries as remaining columns.
                
        Returns:
            pandas.DataFrame: Time-indexed DataFrame with MultiIndex columns:
                - Level 0: Source country (repeated for all destinations)
                - Level 1: Destination countries (including 'Others')
                Index is datetime-converted from the original Date column.
        """
        df_2 = df.copy()
        import_list=list(self.import_in_countries+['Date'])
        # import_list.remove('Others') <-- in my new version I add Others at the end
        new_others = [col for col in df.columns if col not in import_list]
        others_ser = df_2[new_others].sum(axis=1).rename('Others')
        
        all_except_date_df = pandas.concat([
            df_2[list(set(df.columns).difference(set(new_others + ['Date'])))],
            others_ser], axis=1)
        date = df_2.Date
        
        df_temp = pandas.DataFrame(
            all_except_date_df.values, 
            index=pandas.to_datetime(date),
            columns=pandas.MultiIndex.from_product([[country], all_except_date_df.columns],
                                   names=['country_from', 'country_to'])
        )
        return df_temp

    
    def concatenate_all(self):
        """Combine all export data into a single time-series DataFrame.
        
        This method processes each exporting country's data using reduce_exports()
        and concatenates them into a unified DataFrame with consistent time indexing.
        
        Returns:
            pandas.DataFrame: Combined time-series DataFrame with MultiIndex columns:
                - Level 0 ('country_from'): Source countries
                - Level 1 ('country_to'): Destination countries
                Index is datetime representing the time series.
                
        Note:
            Uses inner join to ensure all countries have data for the same time periods.
        """
        export_temp_dict = {
            key: self.reduce_exports(key, self.data_export_to[key]) 
            for key in self.data_export_to.keys()
        }
    
        df_from_to = pandas.concat(export_temp_dict.values(), axis=1, join='inner')

        return df_from_to

    def concatenate_and_truncate_all(self):
        """Create complete export-import matrix including 'Others' category.
        
        This method extends concatenate_all() by adding an 'Others' exporting
        category that represents the difference between total imports reported
        by importing countries and exports reported by tracked exporting countries.
        
        Returns:
            pandas.DataFrame: Complete trade flow matrix with MultiIndex columns:
                - Level 0 ('country_from'): All source countries including 'Others'
                - Level 1 ('country_to'): All destination countries
                
        Note:
            The 'Others' category helps balance the trade flow accounting by
            capturing exports from countries not explicitly tracked in the dataset.
        """
        df_from_to = self.concatenate_all()
    
        result_lst = []
        for country in self.data_import_in.keys():
            result_lst.append(_other_exp_temp(self.data_import_in, df_from_to, country))
        result_df = pandas.concat(result_lst, axis=1, join='inner')
        df_other_exp_temp=pandas.DataFrame(
            result_df.values, 
            index=pandas.to_datetime(result_df.index),
            columns=pandas.MultiIndex.from_product([['Others'], result_df.columns],
                                   names=['country_from', 'country_to'])
        )
        df_from_to = pandas.concat([df_from_to, df_other_exp_temp], axis=1, join='inner')
        
        return df_from_to
        

def _other_exp_temp(data_import, df_from_to, country):
    """Calculate 'Others' export values for a specific importing country.
    
    This helper function computes the difference between total imports reported
    by a country and the sum of exports to that country from tracked exporters.
    This difference represents exports from untracked countries ('Others').
    
    Args:
        data_import (dict): Dictionary containing import data DataFrames by country.
        df_from_to (pandas.DataFrame): Existing export flow data with MultiIndex columns.
        country (str): Name of the importing country to calculate 'Others' for.
        
    Returns:
        pandas.DataFrame: Single-column DataFrame with datetime index showing
            calculated 'Others' export values to the specified country.
            
    Note:
        Handles date alignment and sorting to ensure consistent time series
        operations between import totals and export sums.
    """
    df_sum_indexed = data_import[country].set_index('Date').sum(axis=1)
    df_sum_indexed.index = pandas.to_datetime(df_sum_indexed.index)  # <- c'est ça qu'il manquait 
    
    country_total = df_from_to.xs(country, level='country_to', axis=1).sum(axis=1)
    country_total.index = pandas.to_datetime(country_total.index)
    country_total = country_total.sort_index()
    
    df_sum_indexed = df_sum_indexed.sort_index()

    common_index = country_total.index.intersection(df_sum_indexed.index)
    result = df_sum_indexed[common_index] - country_total[common_index]
    result.name = country

    result_df = result.reset_index()  # Transforme la Series en DataFrame
    result_df=result_df.set_index('Date')
    return result_df