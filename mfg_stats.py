"""Maritime Trade Flow Statistics and Regression Analysis for Mean Field Game (MFG) modelling

This module provides tools for analyzing maritime trade flow data using statistical
regression models. It focuses on modeling trade flows between countries using
phi-point indicators and geographical distances.
"""
from collections import namedtuple
import pickle
from tqdm.notebook import tqdm

import numpy as np
import pandas 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

__version__ = "1.0"


DataSet = namedtuple('DataSet', ['from_to', 'phi_point', 'shift', 'roll'])

def prepare_data(daily_exports_imports_df, drop_others='Others', shift=-30, roll=11):
    """Prepare trade flow data for statistical analysis.
    
    This function processes daily trade flow data by creating phi-point indicators
    through grouping, shifting, and rolling mean calculations. The phi-point represents
    aggregated import volumes by destination country.
    
    Args:
        daily_exports_imports_df (pandas.DataFrame): Raw trade flow data with MultiIndex
            columns where level 0 is source countries and level 1 is destination countries.
        drop_others (str, optional): Name of the category to drop from phi-point 
            calculations, typically representing aggregated "other" countries. 
            If None, no categories are dropped. Defaults to 'Others'.
        shift (int, optional): Number of periods to shift the phi-point data.
            Negative values shift backwards in time. Defaults to -30.
        roll (int, optional): Window size for rolling mean calculation.
            Defaults to 11.
            
    Returns:
        DataSet: Named tuple containing:
            - from_to: Original trade flow data
            - phi_point: Processed phi-point indicators (shifted and smoothed)
            - shift: Applied shift value
            - roll: Applied rolling window size
    """
    phi_point = daily_exports_imports_df.copy().T.groupby(level=1).sum().T
    if drop_others is not None:
        phi_point.drop(drop_others, axis=1, inplace=True)
    phi_point=phi_point.shift(shift).rolling(roll, center=True).mean()

    dataset = DataSet(from_to=daily_exports_imports_df, phi_point=phi_point, shift=shift, roll=roll) 
    return dataset

def slice_of(ds, date_from, date_to):
    """Extract a time slice from a DataSet.
    
    Creates a new DataSet containing data only for the specified date range,
    while preserving the original shift and roll parameters.
    
    Args:
        ds (DataSet): Original dataset to slice.
        date_from (str): Start date for the slice in YYYY-MM-DD format.
        date_to (str): End date for the slice in YYYY-MM-DD format.
        
    Returns:
        DataSet: New dataset containing only data from the specified period,
            with original shift and roll parameters preserved.
            
    Example:
        >>> full_dataset = prepare_data(trade_data)
        >>> training_data = slice_of(full_dataset, '2018-01-01', '2020-12-31')
        >>> print(training_data.from_to.index.min())  # 2018-01-01
    """
    phi_point_temp = ds.phi_point[date_from:date_to]
    from_to_temp   = ds.from_to[date_from:date_to]
    return DataSet(from_to=from_to_temp, phi_point=phi_point_temp, shift=ds.shift, roll=ds.roll)


def df_regression_col(i, j, keep_all=True, dataset=None, verbose=True):
    """Perform OLS regression for a specific source-destination country pair.
    
    This function runs an OLS regression to model trade flows from country i to 
    country j using phi-point indicators as explanatory variables. The dependent
    variable is the smoothed trade flow, and independent variables are lagged
    phi-point values.
    
    Args:
        i (str): Source (exporting) country name.
        j (str): Destination (importing) country name.
        keep_all (bool, optional): If True, uses all countries' phi-points as
            regressors. If False, uses only destination country j's phi-point.
            Defaults to True.
        dataset (DataSet, optional): Dataset containing trade flow and phi-point data.
            Must be provided for the regression to run.
        verbose (bool, optional): If True, prints warning messages when country
            pairs are not found. Defaults to True.
            
    Returns:
        dict: Dictionary containing regression results with keys:
            - 'country_from': Source country name
            - 'country_to': Destination country name  
            - 'const': Regression constant (if fitted)
            - 'phi_point_{country}': Coefficient for each country's phi-point
            - 'residuals': Root mean squared error of the model
            
            If the country pair is not found, returns only country names.
            
    Example:
        >>> result = df_regression_col('Australia', 'China', dataset=my_dataset)
        >>> print(f"Coefficient for China: {result.get('phi_point_China', 'N/A')}")
        >>> print(f"Model RMSE: {result.get('residuals', 'N/A')}")
        
    Note:
        The function applies rolling mean smoothing to the dependent variable
        and automatically adds a constant term to the regression.
    """
    if(j in list(dataset.from_to.loc[:, i].columns)):
        Q_i_j_phi_i=dataset.from_to.copy().rolling(dataset.roll, center=True).mean()
        Q_i_j_phi_i.dropna(inplace=True)
        
        if keep_all:
            df_regression=pandas.concat([Q_i_j_phi_i[(i, j)], dataset.phi_point], axis=1)
        else:
            df_regression=pandas.concat([Q_i_j_phi_i[(i, j)], dataset.phi_point[j]], axis=1)
            
        df_regression.dropna(inplace=True)
        # 1. define X, Y
        y = df_regression.iloc[:, 0]
        X = df_regression.iloc[:, 1:]
        # 2. Add constant
        X = sm.add_constant(X)
        # 4. OLS
        model = sm.OLS(y, X)    
        results = model.fit()

        coefs = results.params.rename(
            index={name: f"phi_point_{name}" for name in results.params.index if name!='const'})

        return ({'country_from': i, 'country_to': j} | 
                coefs.to_dict() | 
                {'residuals': np.sqrt(results.mse_model)})
    else:
        if verbose:
            print(f"... {j} is not in importing from {i}")
        return {'country_from': i, 'country_to': j}


def mark_same_country(regression_df, column_ref='country_to', column_other_pattern='phi_point_', keep_cols=['country_from', 'country_to']):
    """Mark and extract regression coefficients for same-country relationships.
    
    This function processes regression results to identify and extract coefficients
    where the destination country matches the phi-point variable country. This is
    useful for analyzing how a country's own import patterns affect specific
    trade flows.
    
    Args:
        regression_df (pandas.DataFrame): DataFrame containing regression results
            with country pairs and phi-point coefficients.
        column_ref (str, optional): Column name containing the reference country
            (typically destination country). Defaults to 'country_to'.
        column_other_pattern (str, optional): Pattern prefix for phi-point coefficient
            columns. Defaults to 'phi_point_'.
        keep_cols (list, optional): List of columns to preserve in the output.
            Defaults to ['country_from', 'country_to'].
            
    Returns:
        tuple: Two DataFrames:
            - same_B_df: Simplified DataFrame with same-country coefficients
                extracted into a single column named 'same_B_as_{column_ref}'
            - renamed_reg_df: Restructured DataFrame with MultiIndex columns
                separating coefficient values ('B') and country indicators ('dest')
        
    Note:
        The function creates boolean indicators for destination countries and
        extracts coefficients where the phi-point country matches the destination.
    """
    # 1. created a dataframe with all information
    new_cnames = {c: (c.replace(column_other_pattern, ''), 'B') 
                 for c in regression_df.columns if column_other_pattern in c}
    renamed_reg_df = regression_df.copy().rename(columns=new_cnames)
    add_cnames = [(v[0], 'dest') for _, v in new_cnames.items()]
    renamed_reg_df[add_cnames] = False
    for ind in renamed_reg_df.index:
        country_to = renamed_reg_df.loc[ind, column_ref]
        renamed_reg_df.loc[ind,[(country_to, 'dest')]] = True

    # 2. collect same 'B' than cuntry_to (ie 'dest'==True)
    same_B_df = renamed_reg_df[keep_cols].copy()
    same_B_df[f'same_B_as_{column_ref}'] = np.NaN
    for ind in same_B_df.index:
        country_to = renamed_reg_df.loc[ind, column_ref]
        same_B_df.loc[ind, f'same_B_as_{column_ref}'] = renamed_reg_df.loc[ind, [(country_to, 'B')]].squeeze()
        
    return same_B_df, renamed_reg_df
    

def panel_of_regressions(dataset, date_from='2016-06-01', date_to='2020-02-01', keep_all=True, verbose=True):
    """Run regression analysis for all country pairs in the dataset.
    
    This function performs a comprehensive regression analysis by running OLS
    regressions for every possible source-destination country combination in
    the dataset, excluding the 'Others' category.
    
    Args:
        dataset (DataSet): Dataset containing trade flow and phi-point data.
        date_from (str, optional): Start date for regression period in YYYY-MM-DD format.
            Defaults to '2016-06-01'.
        date_to (str, optional): End date for regression period in YYYY-MM-DD format.
            Defaults to '2020-02-01'.
        keep_all (bool, optional): If True, uses all countries' phi-points as
            regressors in each model. If False, uses only destination-specific
            phi-points. Defaults to True.
        verbose (bool, optional): If True, prints progress information and warnings.
            Defaults to True.
            
    Returns:
        pandas.DataFrame: DataFrame containing regression results for all country pairs.
            Each row represents one source-destination pair with columns for:
            - country_from, country_to: Country pair identifiers
            - const: Regression constant
            - phi_point_{country}: Coefficients for each phi-point variable
            - residuals: Model root mean squared error
        
    Note:
        The function automatically excludes 'Others' category from both source
        and destination countries to focus on specific country relationships.
    """
    # get lists of countries
    a, b = list(zip(*dataset.from_to.columns))
    export_countries = list(np.unique(a))
    export_countries.remove('Others')
    if verbose:
        print(f"export_countries: {export_countries}")
    import_countries  = list(np.unique(b))
    import_countries.remove('Others')
    if verbose:
        print(f"import_countries: {import_countries}")
   
    # Créer un DataFrame vide rempli de zéros (ou NaN si tu préfères)
    regression_lst = []
    
    for i in export_countries:
        for j in import_countries:
            regression_lst.append(
                df_regression_col(
                    i, j, 
                    keep_all=keep_all, 
                    dataset=slice_of(dataset, date_from, date_to),
                    verbose=verbose
                )
            )
            
    regression_df = pandas.DataFrame(regression_lst)
    return regression_df

class MFGStatContainer:
    """Maritime Flow Gravity Statistics Container.
    
    This class provides a complete workflow for analyzing maritime trade flows
    using gravity-model-inspired regression techniques. It handles data preparation,
    regression analysis, and integration with geographical distance data.
    
    The workflow follows these steps:
    1. Load and prepare trade flow data
    2. Run panel regressions for all country pairs
    3. Post-process results to identify same-country effects
    4. Integrate with geographical distance data
    
    Attributes:
        daily_exports_imports (pandas.DataFrame): Raw daily trade flow data.
        dataset (DataSet): Processed dataset with phi-point indicators.
        regression_df (pandas.DataFrame): Raw regression results for all country pairs.
        same_B_df (pandas.DataFrame): Simplified same-country coefficient results.
        renamed_df (pandas.DataFrame): Restructured regression results with MultiIndex.
        stats (dict): Summary statistics from the regression analysis.
        all_distances_df (pandas.DataFrame): Bidirectional distance data.
        regression_and_distances_df (pandas.DataFrame): Merged regression and distance data.
        
    Example:
        >>> container = MFGStatContainer('trade_data.pkl')
        >>> container.prepare_dataset(shift=-30, roll=11)
        >>> container.regress_and_postprocess('2018-01-01', '2020-12-31')
        >>> container.concat_distances('distances.csv')
        >>> print(f"Average same-country effect: {container.stats['avg_B']:.3f}")
    """
    # step 0
    daily_exports_imports = None
    # step 1
    dataset = None
    # step 2
    regression_df = None
    same_B_df = None
    renamed_df = None
    stats = None
    # step 3
    all_distances_df = None
    regression_and_distances_df = None
    
    def __init__(self, daily_exports_imports_df):
        """Initialize the MFGStatContainer with trade flow data.
        
        Args:
            daily_exports_imports_df (str or pandas.DataFrame): Either a filepath
                to a pickled DataFrame or a pandas DataFrame containing daily
                trade flow data with MultiIndex columns (source, destination).
                
        Note:
            If a filepath is provided, the file will be loaded using pickle.
            The DataFrame should have a DatetimeIndex and MultiIndex columns
            where level 0 is source countries and level 1 is destination countries.
        """
        if isinstance(daily_exports_imports_df, str):
            # this is a flename
            with open(daily_exports_imports_df, "rb") as f:
                print(f"reading from <{daily_exports_imports_df}>...")
                daily_exports_imports_df = pickle.load(f)
        # this is a dataframe     
        self.daily_exports_imports = daily_exports_imports_df

    def prepare_dataset(self, drop_others='Others', shift=-30, roll=11):
        """Prepare the dataset for regression analysis.
        
        This method processes the raw trade flow data to create phi-point indicators
        and applies temporal transformations for statistical analysis.
        
        Args:
            drop_others (str, optional): Category to exclude from phi-point calculations.
                Defaults to 'Others'.
            shift (int, optional): Number of periods to shift phi-point data.
                Negative values create lags. Defaults to -30.
            roll (int, optional): Rolling window size for smoothing. Defaults to 11.
            
        Returns:
            DataSet: Prepared dataset containing processed trade flows and phi-points.
            
        Note:
            The shift parameter is automatically negated to create proper lags.
        """
        self.dataset = prepare_data(
            self.daily_exports_imports, drop_others=drop_others, shift=-shift, roll=roll
        )
        return self.dataset

    def regress_and_postprocess(self, date_from='2018-06-01', date_to='2021-02-01', verbose=True):
        """Run regression analysis and post-process results.
        
        This method performs comprehensive regression analysis for all country pairs
        and extracts same-country effects for further analysis.
        
        Args:
            date_from (str, optional): Start date for regression period.
                Defaults to '2018-06-01'.
            date_to (str, optional): End date for regression period.
                Defaults to '2021-02-01'.
            verbose (bool, optional): Whether to print progress information.
                Defaults to True.
                
        Returns:
            pandas.DataFrame: Restructured regression results with MultiIndex columns.
            
        Note:
            This method populates several attributes including regression_df,
            same_B_df, renamed_df, and stats. The stats dictionary contains
            summary statistics about the regression results.
        """
        self.regression_df = panel_of_regressions(
            self.dataset, date_from=date_from, date_to=date_to, verbose=verbose)

        # 3. clean and count negative B
        self.same_B_df, self.renamed_df = mark_same_country(self.regression_df)
        
        self.stats = {
            'shift'      : self.dataset.shift  , 
            'roll'       : self.dataset.roll, 
            'y_start'    : date_from, 
            'y_end'      : date_to,
            'nbe_rows'   : self.same_B_df.shape[0],
            'avg_B'      : self.same_B_df.same_B_as_country_to.mean(),
            'nbe_finite' : np.isfinite(self.same_B_df.same_B_as_country_to).mean(),
            'nbe_neg'    : (self.same_B_df.same_B_as_country_to<=0).mean(),
            'regression_df': self.regression_df,
        }
        return self.renamed_df


    def concat_distances(self, distances_df):
        """Integrate geographical distance data with regression results.
        
        This method merges the regression results with geographical distance
        information to enable gravity-model analysis incorporating spatial effects.
        
        Args:
            distances_df (str or pandas.DataFrame): Either a filepath to a CSV
                containing distance data or a pandas DataFrame. The data should
                have columns 'c1' and 'c2' for country pairs and distance information.
                
        Returns:
            pandas.DataFrame: Merged dataset containing both regression results
                and geographical distances for analysis.
                
        Note:
            The distance data is automatically made bidirectional (c1->c2 and c2->c1)
            before merging with regression results.
        """        
        self.all_distances_df = get_all_distance(distances_df)

        self.regression_and_distances_df = pandas.merge(
            self.all_distances_df, self.renamed_df,
            left_on=['c1', 'c2'], right_on=['country_from', 'country_to'],
            how= 'right'
        ).drop(columns=['c1', 'c2'])
        
        return self.regression_and_distances_df


def get_all_distance(distances_df):
    """Create bidirectional distance matrix from unidirectional distance data.
    
    This function takes distance data that may be unidirectional and creates
    a complete bidirectional distance matrix by adding reverse country pair
    combinations with the same distances.
    
    Args:
        distances_df (str or pandas.DataFrame): Either a filepath to a CSV file
            or a pandas DataFrame containing distance data. Should have columns
            'c1' and 'c2' representing country pairs.
            
    Returns:
        pandas.DataFrame: Bidirectional distance DataFrame where both c1->c2
            and c2->c1 relationships are included with identical distance values.
    Note:
        Geographic distance is typically symmetric, so this function ensures
        both directions are available for merging with trade flow data.
    """
    if isinstance(distances_df, str):
        # that's a filename
        distances_df= pandas.read_csv(distances_df)
    else:
        distances_df= distances_df

    all_distances_df = pandas.concat([
        distances_df, distances_df.rename(columns={'c1': 'c2', 'c2': 'c1'}) 
    ], axis=0).reset_index(drop=True)

    return all_distances_df



CostHelper = namedtuple('CostHelper', 
                        ['c_from', 'c_to', 'Aij_df', 'Aij', 'Tij_df', 'Tij', 'c', 'r', 'vj', 'vi', 'renorm'])

def create_cost_helper(regression_df, renorm=10_000):
    """Creates a CostHelper object from regression data for optimization.
    
    Processes regression data containing country-to-country relationships and
    creates a CostHelper object with normalized coefficient matrices and
    parameter transformation functions for use in cost optimization algorithms.
    
    Args:
        regression_df (pd.DataFrame): Regression data containing columns:
            'country_from', 'country_to', 'const', and 'distKM_num'.
            Should contain bilateral relationships between countries.
        renorm (int, optional): Normalization factor to scale the coefficient
            matrices. Defaults to 10,000. Used to improve numerical stability
            during optimization.
    
    Returns:
        CostHelper: A configured CostHelper object containing:
            - c_from: Index of origin countries
            - c_to: Index of destination countries  
            - Aij_df/Aij: Normalized constant coefficient matrix (DataFrame/array)
            - Tij_df/Tij: Normalized distance coefficient matrix (DataFrame/array)
            - c, r, vj, vi: Lambda functions for parameter transformations
            - renorm: The normalization factor used
    
    Raises:
        AssertionError: If the column or index names/orders of the pivoted
            matrices Aij and Tij do not match.
        KeyError: If required columns are missing from regression_df.
    """
    nci = len(regression_df["country_from"].unique())
    ncj = len(regression_df["country_to"].unique())
    
    Aij = regression_df.pivot(
            index='country_from', columns='country_to', values='const'
        )
    Tij = regression_df.pivot(
            index='country_from', columns='country_to', values='distKM_num'
        )
    assert (Aij.columns==Tij.columns).all(), "columns (country_to  ) names/orders of A and T not equal"
    assert (Aij.index  ==Tij.index  ).all(), "index   (country_from) names/orders of A and T not equal"
    
    chelp = CostHelper(
        c_from = Aij.index,
        c_to = Aij.columns,
        Aij_df = Aij / renorm,
        Aij = Aij.fillna(0).values / renorm,
        Tij_df = Tij / renorm,
        Tij = Tij.fillna(0).values / renorm,
        c  = lambda x: x[0]**2,
        r  = lambda x: np.row_stack(nci*[x[1:ncj+1]])**2,
        vj = lambda x: np.row_stack(nci*[x[ncj+1:2*ncj+1]])**2,
        vi = lambda x: np.column_stack(ncj*[x[2*ncj+1:]])**2,
        renorm = renorm,
    )
    
    return chelp


def extract_results(x0, result, chelp):
    """Extracts and formats optimization results into a readable dictionary.
    
    Takes the raw optimization result and parameter vector, then uses the
    CostHelper's transformation functions to decode the parameters into
    their meaningful economic components.
    
    Args:
        x0 (np.ndarray): Parameter vector containing the optimized values.
            Should have the structure expected by the CostHelper's lambda
            functions (c, r, vj, vi parameters).
        result: Optimization result object containing the final loss value
            in the 'fun' attribute (typically from scipy.optimize).
        chelp (CostHelper): CostHelper object containing the parameter
            transformation functions and data matrices.
    
    Returns:
        dict: Dictionary containing extracted results with keys:
            - 'loss' (float): Final optimization loss value
            - 'c' (float): Decoded cost parameter (scalar)
            - 'r' (np.ndarray): Decoded r parameters (1D array)
            - 'vj' (np.ndarray): Decoded vj parameters (1D array) 
            - 'vi' (np.ndarray): Decoded vi parameters (1D array)
    """
    return dict(
            loss = result.fun,
            c =chelp.c(x0),
            r =chelp.r(x0)[0,:],
            vj=chelp.vj(x0)[0,:],
            vi=chelp.vi(x0)[:,0],
        )

def total_cost(x, chelp=None):
    """Computes the total cost function for optimization.
    
    Calculates the root mean square error between the theoretical model
    prediction (Aij * (r + c * Tij)) and the target values (vj - vi).
    This function serves as the objective function for optimization algorithms.
    
    The cost function implements the model:
    
      -  predicted = Aij * (r + c * Tij)
      -  target = vj - vi
      -  loss = sqrt(mean((predicted - target)^2))

    This is a normalized version of this criterion:

    .. math::
        
        \\min_{c, \\{r_j\\}, \\{v_i\\}} \\sum_{i \\ne j} \\left\\| A_{ij} \\cdot \\left( r_j + c \\cdot T_{ij} \\right) - \\left( v_j - v_i \\right) \\right\\|^2
    
    Args:
        x (np.ndarray): Parameter vector containing optimization variables.
            The vector is decoded by the CostHelper's transformation functions
            into economic parameters c, r, vj, and vi.
        chelp (CostHelper, optional): CostHelper object containing the data
            matrices (Aij, Tij) and parameter transformation functions.
            Defaults to None, but must be provided for the function to work.
    
    Returns:
        float: Root mean square error between model predictions and targets.
            Lower values indicate better fit to the theoretical model.
 
    """
    # decode input vector components
    c  = chelp.c( x)
    r  = chelp.r( x)
    vj = chelp.vj(x)
    vi = chelp.vi(x)
    
    loss = np.sqrt(np.mean((chelp.Aij * (r + c * chelp.Tij) - (vj - vi))**2))

    return loss