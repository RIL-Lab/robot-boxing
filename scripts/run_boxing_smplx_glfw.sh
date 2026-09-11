#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
aitviewer_root="${AITVIEWER_ROOT:-/home/ubuntu22/aitviewer}"
source "$aitviewer_root/.venv/bin/activate"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export XDG_SESSION_TYPE=x11
unset QT_QPA_PLATFORM QT_OPENGL WAYLAND_DISPLAY
unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME

if [ "$#" -eq 0 ]; then
    set -- \
        /home/ubuntu22/boxing_smplx_teacher_demo/person_1_smplx.npz \
        /home/ubuntu22/boxing_smplx_teacher_demo/person_2_smplx.npz
fi

# Keep replay commands valid when Ubuntu auto-mounts the removable drive as
# Elements1 because the Elements mount-point name is already occupied.
elements_root=/media/ubuntu22/Elements
if [[ ! -d "$elements_root/smplx文件" && -d /media/ubuntu22/Elements1/smplx文件 ]]; then
    elements_root=/media/ubuntu22/Elements1
fi
args=()
for argument in "$@"; do
    if [[ "$elements_root" != "/media/ubuntu22/Elements" && "$argument" == /media/ubuntu22/Elements/* ]]; then
        argument="$elements_root/${argument#/media/ubuntu22/Elements/}"
    fi
    args+=("$argument")
done

python "$script_dir/play_boxing_smplx.py" "${args[@]}"
