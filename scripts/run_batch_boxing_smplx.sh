#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Ubuntu may append "1" when the same removable-drive mount-point name is
# already occupied.  Locate the real Elements volume instead of assuming its
# current device/mount name.
elements_root=""
for candidate in /media/ubuntu22/Elements /media/ubuntu22/Elements1; do
    if [[ -x "$candidate/sam-body4d/.venv/bin/python" && -d "$candidate/multi-hmr" ]]; then
        elements_root="$candidate"
        break
    fi
done
if [[ -z "$elements_root" ]]; then
    echo "错误：找不到 Elements 硬盘上的 sam-body4d Python 和 multi-hmr。" >&2
    echo "请确认移动硬盘已挂载到 /media/ubuntu22/Elements 或 Elements1。" >&2
    exit 1
fi

# Keep old commands valid if the desktop mounted the disk as Elements1.
args=()
for argument in "$@"; do
    if [[ "$elements_root" != "/media/ubuntu22/Elements" && "$argument" == /media/ubuntu22/Elements/* ]]; then
        argument="$elements_root/${argument#/media/ubuntu22/Elements/}"
    fi
    args+=("$argument")
done

echo "Using Elements root: $elements_root"
python_bin="${BOXING_PYTHON:-$elements_root/sam-body4d/.venv/bin/python}"
exec "$python_bin" "$script_dir/batch_boxing_smplx.py" "${args[@]}"
