from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen2.5-7B-Instruct"
downloaded = snapshot_download(
    repo_id,
    cache_dir="/root/autodl-fs/data2/root/.cache/modelscope/hub/",
    endpoint="https://hf-mirror.com/",
)