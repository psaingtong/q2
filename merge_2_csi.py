import pandas as pd
import os

# รายชื่อพิกัด 4 มุม
#"182_114","1732_114","182_864","1732_864"
#points = ["162_864","162_134","1762_864","1762_134"]
points = ["182_864", "1732_114"]
#points = ["192_104","394_504","994_674","1476_154","1749_809", "162_864", "1762_134", "162_134","1762_864"]
# สัดส่วนที่ต้องการ (25%, 10%, 10%, 10%)
weights = {
    "182_864": 0.5,
    "1732_114": 0.5
}   



input_dir = "data/"
TARGET_TOTAL_ROWS = 400

def create_synced_master(output_name):
    all_raw_data = {}
    
    # 1. โหลดข้อมูลทั้งหมดเข้ามาก่อน
    for pt in points:
        path_lt = os.path.join(input_dir, f"{pt}_lt_01.csv")
        path_rt = os.path.join(input_dir, f"{pt}_rt_01.csv")
        
        if os.path.exists(path_lt) and os.path.exists(path_rt):
            try:
                df_lt = pd.read_csv(path_lt, header=0)
                df_rt = pd.read_csv(path_rt, header=0)
                csi_col = 'CSI_VALUES'
                
                min_rows = min(len(df_lt), len(df_rt))
                df_lt, df_rt = df_lt.head(min_rows), df_rt.head(min_rows)
                
                mask_lt = df_lt[csi_col].astype(str).apply(lambda x: len(x.strip().split()) == 128)
                mask_rt = df_rt[csi_col].astype(str).apply(lambda x: len(x.strip().split()) == 128)
                valid_mask = mask_lt & mask_rt
                
                clean_lt, clean_rt = df_lt[valid_mask], df_rt[valid_mask]
                
                if len(clean_lt) > 0:
                    x_coord, y_coord = map(int, pt.split('_'))
                    synced_pt_df = pd.DataFrame({
                        'target_x': [x_coord / 1848.0] * len(clean_lt),
                        'target_y': [y_coord / 1046.0] * len(clean_lt),
                        'csi_lt': clean_lt[csi_col].values,
                        'csi_rt': clean_rt[csi_col].values
                    })
                    all_raw_data[pt] = synced_pt_df
                    print(f"📦 Loaded {pt}: {len(clean_lt)} rows")
            except Exception as e:
                print(f"❌ Error {pt}: {e}")

    if len(all_raw_data) < len(points):
        print("\n!!! WARNING: Data missing for some points. Proportions might be skewed.")

    # 2. ทำ Weighted Sampling
    final_dfs = []
    print(f"\n⚖️ Applying Weighted Sampling (Total Target: {TARGET_TOTAL_ROWS})...")
    
    for pt, weight in weights.items():
        if pt in all_raw_data:
            target_rows = int(TARGET_TOTAL_ROWS * weight)
            available_rows = len(all_raw_data[pt])
            
            # ถ้าข้อมูลมีไม่พอตาม % ที่ต้องการ ให้เอาเท่าที่มี
            actual_rows = min(target_rows, available_rows)
            
            sampled_df = all_raw_data[pt].sample(n=actual_rows, random_state=42)
            final_dfs.append(sampled_df)
            print(f"   -> {pt}: Requested {target_rows} rows | Sampled {actual_rows} rows ({weight*100}%)")

    # 3. รวมร่างและ Shuffle
    master_df = pd.concat(final_dfs, ignore_index=True)
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 4. บันทึกไฟล์
    output_path = os.path.join(input_dir, output_name)
    master_df.to_csv(output_path, header=False, index=False)
    
    print("-" * 40)
    print(f"🎉 SUCCESS: สร้าง Master File (Weighted) สมบูรณ์แล้ว")
    print(f"Output: {output_path}")
    print(f"Final Distribution:\n{master_df.groupby(['target_x','target_y']).size()}")

print("--- Starting Synced Master Generator (Weighted Mode) ---")
create_synced_master("master_synced_pairs.csv")
