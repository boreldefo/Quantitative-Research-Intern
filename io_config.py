"""Dynamic Configuration Loader

A simple module for loading YAML configuration files into nested namedtuple structures.
The namedtuple structure dynamically reflects the configuration file structure.
"""

__version__ = "1.0.0"

import yaml
import pickle
import os.path
from collections import namedtuple
from typing import Any, Dict, Union

def wrap_with_config(function, **outer_kwargs):
    """Wraps a function with pre-configured keyword arguments.
    
    Creates a new function that combines the outer keyword arguments with
    any arguments passed at call time. Inner keyword arguments take precedence
    over outer ones in case of conflicts.
    
    Args:
        function (callable): The function to wrap with configuration.
        **outer_kwargs: Keyword arguments to pre-configure the function with.
            These will be passed to the wrapped function on every call.
    
    Returns:
        callable: A wrapped function that accepts both positional and keyword
            arguments. The wrapped function combines outer_kwargs with any
            arguments provided at call time.
    
    Example:
        >>> def greet(name, greeting="Hello", punctuation="!"):
        ...     return f"{greeting} {name}{punctuation}"
        >>> say_hi = wrap_with_config(greet, greeting="Hi")
        >>> say_hi("Alice")
        'Hi Alice!'
        >>> say_hi("Bob", punctuation="?")
        'Hi Bob?'
    """
    def wrapped(*args, **inner_kwargs):
        if len(inner_kwargs)==0:
            inner_kwargs={}
        all_kwargs = outer_kwargs | inner_kwargs
        if len(args)==0:
            return function(**all_kwargs)
        return function(*args, **all_kwargs)
    return wrapped


def activate_save_fig(config):
    """Activates a pre-configured figure saving function.
    
    Creates a save_fig function that is pre-configured with the image directory
    from the provided configuration object. This allows for consistent figure
    saving across the application without repeatedly specifying the directory.
    
    Args:
        config: Configuration object containing the image directory setting.
            Must have a `config.image_dir` attribute that specifies where
            images should be saved.
    
    Returns:
        callable: A pre-configured save_fig function from the io_plots module
            that will save figures to the specified image directory. The returned
            function can be called with additional arguments as needed.
    
    Raises:
        ImportError: If the io_plots module cannot be imported.
        AttributeError: If the config object doesn't have the expected structure
            (config.config.image_dir).
    
    Example:
        >>> save_fig = activate_save_fig(my_config)
        >>> save_fig("this_is_a_plot")  # Saves to config.config.image_dir/<this_is_a_plot>.[png|pdf]
    """
    import io_plots
    
    return wrap_with_config(io_plots.save_fig, img_dir=config.config.image_dir)

    
def dict_to_namedtuple(name: str, data: Dict[str, Any], eval_key: str ='eval_') -> Any:
    """Convert a nested dictionary to nested namedtuples recursively.
    
    This function creates namedtuple classes dynamically based on the structure
    of the input dictionary. Nested dictionaries become nested namedtuples,
    except when the key ends with '_dict' - in that case, the dictionary
    is preserved as a regular Python dictionary. Additionally, keys starting
    with the eval_key prefix will have their values evaluated as Python
    expressions.
    
    Args:
        name (str): Name for the namedtuple class to be created.
        data (dict): Dictionary to convert, can contain nested dictionaries.
        eval_key (str, optional): Prefix that triggers Python evaluation of
            string values. Keys starting with this prefix will have their
            values passed to eval() and the prefix removed from the final
            attribute name. Defaults to 'eval_'.
        
    Returns:
        namedtuple: An instance of a dynamically created namedtuple class
            with the same structure as the input dictionary.
            
    Note:
        - Keys ending with '_dict' preserve their values as dictionaries
        - Keys starting with eval_key have their values evaluated as Python code
        - The eval_key prefix is stripped from the final attribute name
        
    Warning:
        Using eval() can be dangerous with untrusted input. Only use this
        feature with trusted configuration files.
            
    Example:
        >>> config = {
        ...     'database': {'host': 'localhost', 'port': 5432},
        ...     'countries_dict': {'US': 'United States', 'FR': 'France'},
        ...     'eval_max_connections': '100 * 2',
        ...     'eval_timeout_list': '[30, 60, 120]'
        ... }
        >>> result = dict_to_namedtuple('Config', config)
        >>> print(result.database.host)  # 'localhost' (namedtuple)
        >>> print(result.countries_dict['US'])  # 'United States' (dict)
        >>> print(result.max_connections)  # 200 (evaluated)
        >>> print(result.timeout_list)  # [30, 60, 120] (evaluated list)
    """
    # Convert nested dictionaries to namedtuples recursively
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Create nested namedtuple with capitalized name
            nested_name = f"{name}_{key.capitalize()}"
            converted_data[key] = dict_to_namedtuple(nested_name, value)
        elif len(key)>len(eval_key) and key[:len(eval_key)]==eval_key:
            converted_data[key[len(eval_key):]] = eval(value)
        else:
            converted_data[key] = value
    
    # Create namedtuple class dynamically
    ConfigTuple = namedtuple(name, converted_data.keys())
    return ConfigTuple(**converted_data)


def load_config(file_path: str, root_name: str = "Config") -> Any:
    """Load YAML configuration file into nested namedtuple structure.
    
    This function reads a YAML file and converts it into a tree of namedtuple
    objects that mirrors the original structure of the configuration file.
    
    Args:
        file_path (str): Path to the YAML configuration file.
        root_name (str, optional): Name for the root namedtuple class.
            Defaults to "Config".
            
    Returns:
        namedtuple: Root namedtuple object containing the entire configuration
            structure as nested namedtuples.
            
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        
    Example:
        >>> config = load_config('config.yaml')
        >>> print(config.INPUT.PORTS.input_positions_file)
        >>> print(config.OUTPUT.ports.intermediate.comb1_file)
        >>> print(config.CONFIG.countries_export_dict['Indonesia'])  # Dict access
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
    
    return dict_to_namedtuple(root_name, config_data)


# Example usage and testing
if __name__ == "__main__":
    # Create example YAML file with proper indentation structure
    yaml_content = """
inputs:
  ports:
    input_positions_file: './data_in/input_positions_ports.csv'
    input_distances_file: './data_in/out.geojson'
  flows:
    export_coal: './data_in/shipfix/export_coal'
    import_coal: './data_in/shipfix/import_coal'
    eval_countries_export: "{
    'Indonesia': 'all_voyage_orders_by_cargo_type_time_series_stacked_Indonesia_to',
    'Australia': 'all_voyage_orders_by_cargo_type_time_series_stacked_Australia_to',
    'SAFR': 'all_voyage_orders_by_cargo_type_time_series_stacked_SAFR_to',
    'US': 'all_voyage_orders_by_cargo_type_time_series_stacked_US_to',
    'Russia': 'all_voyage_orders_by_cargo_type_time_series_stacked_Russia_to'}"
    eval_countries_import: "{
    'India': 'all_voyage_orders_by_cargo_type_time_series_stacked_to_India',
    'China': 'all_voyage_orders_by_cargo_type_time_series_stacked_to_China',
    'Vietnam': 'all_voyage_orders_by_cargo_type_time_series_stacked_to_Vietnam',
    'Japan': 'all_voyage_orders_by_cargo_type_time_series_stacked_to_Japan',
    'South Korea': 'all_voyage_orders_by_cargo_type_time_series_stacked_to_SK'}"
    
output:
  ports:
    intermediate:
      comb1_file: './data_out/all_combinations_of_ports_positions.csv'
      comb2_file: './data_out/all_combinations_of_ports_positions_nocountry.csv'
    final:
      output_distances_countries_file: './data_out/distances_countries.csv'
  timeseries:
    save_concatenated_and_truncated_ts_in: "./data_out/all_from_to.pkl"

config:
  image_world_ports: './images/world_ports'
  image_distrib_distances: './images/distrib_distances'
""".strip()
    
    # Write example YAML file
    with open('example_config.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Load and test the configuration
    config = load_config('example_config.yaml')
    
    # Test access patterns
    print("configuration:")
    print(config.config._asdict())

    print("inputs ralted to ports:")
    print(config.inputs.ports._asdict())


def memoize(side, fname, default_value, mode='pickle'):
    assert mode=='pickle', "the only implemented mode is [pickle]"
    assert side in ['read', 'r', 'write', 'w'], "the side has to be read or write"
    if side in ['write', 'w']:
        with open(fname, "wb") as f:
            pickle.dump(default_value, f)
        return 
        
    if os.path.isfile(fname):
        print(f"{fname} exists")
        with open(fname, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        return default_value
    