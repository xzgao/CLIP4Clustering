# txt2excel.py
import pandas as pd

def convert_cluster_txt_to_excel(txt_file: str, excel_file: str):
    lines_all = open(txt_file, encoding="utf-8").readlines()
    if len(lines_all) <= 2:
        print("Not enough data in the txt file, skipping Excel generation")
        return

    header_line = lines_all[1].strip().split("\t")
    data_rows = [line.strip().split("\t") for line in lines_all[2:]]

    df_out = pd.DataFrame(data_rows, columns=header_line)
    df_out["epoch"] = df_out["epoch"].astype(int)
    df_out["nmi"] = df_out["nmi"].astype(float)
    df_out["acc"] = df_out["acc"].astype(float)
    df_out["ari"] = df_out["ari"].astype(float)

    df_out.to_excel(excel_file, index=False, engine="openpyxl")
    #print(f"✅ Excel updated -> {excel_file}")


if __name__ == "__main__":
    convert_cluster_txt_to_excel(
        txt_file="./cluster_result.txt",
        excel_file="./cluster_result.xlsx"
    )
