"""
regression.py
-------------
OLS regression utilities for testing the association between CSR features and
Cumulative Abnormal Returns (CAR).

Functions:
    run_ols_regression        - Basic OLS.
    run_ols_regression_robust - OLS with HC3 heteroskedasticity-robust standard errors.
    run_regression_sweep      - Run all combinations of CSR features and CAR metrics
                                across a set of control variables.

Usage (CLI):
    python src/analysis/regression.py \\
        --data /results/merged_with_controls.csv \\
        --y-var 3factor_post4_13 \\
        --x-var csr_ratio \\
        --output /results/regression_output.csv
"""

import argparse
import itertools
from typing import Optional

import pandas as pd
import statsmodels.api as sm


def run_ols_regression(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    controls: Optional[list] = None,
    verbose: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run an OLS regression of y_var on x_var with optional control variables.

    Args:
        df: Input DataFrame.
        x_var: Name of the primary independent variable.
        y_var: Name of the dependent variable.
        controls: List of additional control variable names.
        verbose: If True, print the regression summary.

    Returns:
        Fitted statsmodels OLS result.
    """
    cols = [x_var, y_var] + (controls or [])
    filtered = df[cols].dropna()

    X = sm.add_constant(filtered[[x_var] + (controls or [])])
    y = filtered[y_var]

    model = sm.OLS(y, X).fit()
    if verbose:
        print(model.summary())
    return model


def run_ols_regression_robust(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    controls: Optional[list] = None,
    verbose: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run an OLS regression with HC3 heteroskedasticity-robust standard errors.

    Args:
        df: Input DataFrame.
        x_var: Name of the primary independent variable.
        y_var: Name of the dependent variable.
        controls: List of additional control variable names.
        verbose: If True, print the regression summary.

    Returns:
        Fitted statsmodels OLS result with robust SEs.
    """
    cols = [x_var, y_var] + (controls or [])
    filtered = df[cols].dropna()

    X = sm.add_constant(filtered[[x_var] + (controls or [])])
    y = filtered[y_var]

    model = sm.OLS(y, X).fit(cov_type="HC3")
    if verbose:
        print(model.summary())
    return model


def run_regression_sweep(
    df: pd.DataFrame,
    y_vars: list,
    x_vars: list,
    controls: list,
    robust: bool = True,
) -> pd.DataFrame:
    """
    Run OLS regressions exhaustively over all combinations of y_vars and x_vars,
    each time including all controls that have sufficient non-missing data.

    Args:
        df: Input DataFrame.
        y_vars: List of dependent variable names (e.g. different CAR windows).
        x_vars: List of CSR feature variables to test as independent variables.
        controls: List of control variable names to always include.
        robust: If True, use HC3 robust standard errors.

    Returns:
        DataFrame of regression results with columns: y_var, x_var, coef, pvalue, r2, nobs.
    """
    rows = []
    fit_fn = run_ols_regression_robust if robust else run_ols_regression

    for y_var, x_var in itertools.product(y_vars, x_vars):
        valid_controls = [c for c in controls if df[c].notna().mean() > 0.7]
        try:
            model = fit_fn(df, x_var, y_var, controls=valid_controls, verbose=False)
            rows.append({
                "y_var": y_var,
                "x_var": x_var,
                "coef": model.params.get(x_var),
                "pvalue": model.pvalues.get(x_var),
                "r2": model.rsquared,
                "nobs": int(model.nobs),
                "n_controls": len(valid_controls),
            })
        except Exception as e:
            rows.append({
                "y_var": y_var,
                "x_var": x_var,
                "coef": None,
                "pvalue": None,
                "r2": None,
                "nobs": None,
                "n_controls": None,
                "error": str(e),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run OLS regression of CSR features on Cumulative Abnormal Returns."
    )
    parser.add_argument("--data", required=True, help="Path to merged CSV dataset.")
    parser.add_argument("--y-var", required=True, help="Dependent variable (CAR metric).")
    parser.add_argument("--x-var", required=True, help="Independent variable (CSR feature).")
    parser.add_argument(
        "--controls", nargs="*", default=[], help="Space-separated control variable names."
    )
    parser.add_argument(
        "--robust", action="store_true", help="Use HC3 heteroskedasticity-robust SEs."
    )
    parser.add_argument(
        "--output", default=None, help="(Optional) Save results table to this CSV path."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    fit_fn = run_ols_regression_robust if args.robust else run_ols_regression
    model = fit_fn(df, args.x_var, args.y_var, controls=args.controls or None)

    if args.output:
        results = pd.DataFrame({
            "param": model.params.index,
            "coef": model.params.values,
            "pvalue": model.pvalues.values,
        })
        results.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
