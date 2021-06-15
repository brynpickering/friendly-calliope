import pandas as pd
from styleframe import StyleFrame
import argparse

SHEET = "Sheet1"


def intercomparison_iamc_to_csv(path_to_intercomparison_indicators, outpath):
    df = pd.read_excel(path_to_intercomparison_indicators, sheet_name=SHEET, header=0)
    style_df = StyleFrame.read_excel(
        path_to_intercomparison_indicators, read_style=True, sheet_name=SHEET, header=0
    )
    final_df = pd.DataFrame(
        index=df.stack().values,
        data=df.stack().index.get_level_values(1),
        columns=["indicator_family"]
    )
    final_df["required"] = False
    final_df = final_df.rename_axis(index="indicator")

    for col in df.columns:
        idx = style_df[col].style.font_color.isin(['FFC00000'])
        final_df.loc[df.loc[idx, col].dropna().index, "required"] = True

    # good to be sure we have captured *some* indicators, validating that the colour choice is correct
    assert final_df.required.sum() > len(df.columns)

    # there are duplicated indicators, but we only care about those which are "required"
    assert final_df[final_df.required].index.duplicated().sum() == 0
    final_df.to_csv(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", help="Path to intercomparison Excel file")
    parser.add_argument("outpath", help="Path to output CSV")

    args = parser.parse_args()
    intercomparison_iamc_to_csv(args.inpath, args.outpath)
