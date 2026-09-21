"""
Scaffold & Runner for Project 1: Ames Housing
Run with: python project1_ames/scaffold_project1.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step1_load_ames_housing import load_ames_housing
from step2_prepare_target import prepare_target
from step3_handle_feature_types import handle_feature_types
from step4_detect_outliers_cooks import detect_outliers_cooks
from step5_train_regularized_models import train_regularized_models


def main():
    print("Testing Project 1: Ames Housing...")
    try:
        df = load_ames_housing()
        print(f"[OK] load_ames_housing executed")
    except NotImplementedError:
        print("[ ] load_ames_housing not implemented yet.")
        return
    except Exception as e:
        print(f"[ ] load_ames_housing returned: {e}")
        return

    try:
        X, y = prepare_target(df)
        print(f"[OK] prepare_target executed")
    except NotImplementedError:
        print("[ ] prepare_target not implemented yet.")
        return
    except Exception as e:
        print(f"[ ] prepare_target returned: {e}")
        return

    try:
        X_proc = handle_feature_types(X)
        print(f"[OK] handle_feature_types executed")
    except NotImplementedError:
        print("[ ] handle_feature_types not implemented yet.")
        return
    except Exception as e:
        print(f"[ ] handle_feature_types returned: {e}")
        return

    try:
        outliers = detect_outliers_cooks(X, y)
        print(f"[OK] detect_outliers_cooks executed (Found {len(outliers)} outliers)")
    except NotImplementedError:
        print("[ ] detect_outliers_cooks not implemented yet.")

    try:
        models = train_regularized_models(X_proc, y)
        print(f"[OK] train_regularized_models executed (Models: {list(models.keys())})")
        for name, m in models.items():
            print(f"  - {name}: alpha={getattr(m, 'alpha_', getattr(m, 'alpha', 'N/A')):.4f}, R2={m.score(X_proc, y):.4f}")
    except NotImplementedError:
        print("[ ] train_regularized_models not implemented yet.")


if __name__ == "__main__":
    main()
