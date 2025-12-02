#!/usr/bin/env python3
"""
合并所有 metadata-f24* 开头的CSV文件到一个新的CSV文件
"""

import os
import csv
import glob
from pathlib import Path

def merge_csv_files():
    # 获取当前目录下所有匹配的CSV文件
    pattern = "metadata-f24*.csv"
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"未找到匹配 '{pattern}' 的文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")
    
    # 输出文件名
    output_file = "metadata-f24_aperture5-merged.csv"
    
    # 合并所有CSV文件
    total_rows = 0
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # 写入表头（只写一次）
        header_written = False
        
        for csv_file in csv_files:
            print(f"\n正在处理: {csv_file}")
            file_rows = 0
            
            try:
                with open(csv_file, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    
                    # 读取表头
                    header = next(reader, None)
                    if header is None:
                        print(f"  警告: {csv_file} 是空文件，跳过")
                        continue
                    
                    # 写入表头（只在第一次）
                    if not header_written:
                        writer.writerow(header)
                        header_written = True
                    
                    # 写入数据行
                    for row in reader:
                        if len(row) >= 2:  # 确保至少有两列
                            writer.writerow(row)
                            file_rows += 1
                            total_rows += 1
                        else:
                            print(f"  警告: 跳过格式不正确的行: {row}")
                    
                    print(f"  完成: 从 {csv_file} 读取了 {file_rows} 行")
            
            except Exception as e:
                print(f"  错误: 处理 {csv_file} 时出错: {str(e)}")
                continue
    
    print(f"\n合并完成!")
    print(f"输出文件: {output_file}")
    print(f"总行数: {total_rows} (不包括表头)")
    print(f"合并了 {len(csv_files)} 个文件")

if __name__ == "__main__":
    merge_csv_files()

