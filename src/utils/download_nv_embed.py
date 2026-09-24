from huggingface_hub import snapshot_download

# Thay thế tên mô hình bạn muốn tải vào đây
repo_id = "nvidia/NV-Embed-v1" 

# Tải toàn bộ thư mục mô hình về thư mục 'my_local_model'
snapshot_download(
    repo_id=repo_id, 
    local_dir="./models/NV-Embed-v1", 
    local_dir_use_symlinks=False
)
print("Tải mô hình thành công!")