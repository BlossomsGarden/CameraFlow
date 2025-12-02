# Qwen——————
# conda activate wlh-py
# cd /data2/wlh/waymo-long-tail-5000/paper
# CUDA_VISIBLE_DEVICES=0  python infer.py
# 
# Gen Metadata CSV ——————
# conda activate wlh-py
# cd /data/wlh/ReCamMaster/MultiCamVideo-Dataset
# CUDA_VISIBLE_DEVICES=0  python gen_metadata_csv.py
#
# 注意手动改一下里面可能出现全是感叹号的情况
#
# 不必描述摄像头移动了，因为它理解不了，全都是stationary我生成个毛线

import torch
import os
import csv
from pathlib import Path
import glob


def clean_csv_encoding(csv_path):
    """
    清理CSV文件中的编码问题
    """
    try:
        # 读取原始文件内容
        with open(csv_path, 'rb') as f:
            content = f.read()
        
        # 尝试用不同编码解码
        try:
            # 首先尝试UTF-8
            decoded_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试其他常见编码
                decoded_content = content.decode('latin-1')
                print("Warning: Used latin-1 encoding to fix file")
            except:
                # 最后尝试忽略错误
                decoded_content = content.decode('utf-8', errors='ignore')
                print("Warning: Used utf-8 with errors ignored to fix file")
        
        # 清理非法字符
        cleaned_content = ''
        for char in decoded_content:
            # 只保留可打印的ASCII字符和常见Unicode字符
            if ord(char) >= 32 or char in '\t\n\r':
                cleaned_content += char
            else:
                # 替换非法字符为“删除”
                cleaned_content += ''
                print(f"Replaced illegal character: {ord(char)}")
        
        # 写回文件
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Successfully cleaned CSV file: {csv_path}")
        
    except Exception as e:
        print(f"Error cleaning CSV file: {str(e)}")


def validate_csv_format(csv_path):
    """
    验证CSV文件格式，确保所有file_name都是字符串，并报告空行位置
    """
    try:
        import pandas as pd
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # # 检查file_name列的数据类型
        # print(f"Data types in CSV:")
        # print(df.dtypes)

        # 记录空行位置
        empty_rows = []
        # 检查每一行，记录空值或非字符串值
        for idx, row in df.iterrows():
            file_name = row['file_name']
            # 检查是否为NaN或空字符串
            if pd.isna(file_name) or file_name == '':
                empty_rows.append(idx + 2)  # +2 是因为索引从0开始，且跳过标题行
                print(f"Warning: Empty file_name at line {idx + 2}")
            # 检查是否为非字符串类型
            elif not isinstance(file_name, str):
                print(f"Warning: Non-string file_name at line {idx + 2}: {type(file_name)} -> {file_name}")
        
        # 确保file_name列是字符串类型
        if df['file_name'].dtype != object:  # object通常表示字符串
            print(f"Converting file_name column from {df['file_name'].dtype} to string")
            df['file_name'] = df['file_name'].astype(str)
        
        # 检查是否有NaN或空值
        nan_count = df['file_name'].isna().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in file_name column, filling with empty string")
            df['file_name'] = df['file_name'].fillna('')
        
        # 移除可能存在的空行
        original_rows = len(df)
        df = df.dropna(how='all')  # 删除全为空的行
        removed_count = original_rows - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} empty rows")
        
        # 保存修复后的CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV format validated and fixed: {csv_path}")
        
        # 报告空行情况
        if empty_rows:
            print(f"Found {len(empty_rows)} empty rows at lines: {empty_rows}")
        else:
            print("No empty rows found in the CSV file.")
        
        return True
        
    except Exception as e:
        print(f"Error validating CSV format: {str(e)}")
        return False


def main():
    output_csv_path = "metadata-f24-train.csv"
    
    
    print("Cleaning CSV file encoding...")
    clean_csv_encoding(output_csv_path)
    print("CSV file cleaning completed!")
    
    
    print("Validating CSV format...")
    validate_csv_format(output_csv_path)
    print("CSV format validation completed!")

if __name__ == "__main__":
    main()