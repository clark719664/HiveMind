# PyInstaller spec for the CPU-only standalone HiveMind HUD (one-folder build).
# Invoked by build_standalone.py from a dedicated virtual environment in which the
# project itself is installed as a wheel, so bundled modules match an installed package.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

hiddenimports = (
    collect_submodules("hivemind")
    + collect_submodules("uvicorn")
    + collect_submodules("transformers.models.gpt2")
    + collect_submodules("transformers.models.qwen2")
    + ["websockets.legacy", "websockets.asyncio.server"]
)

datas = collect_data_files("hivemind")
for dist in (
    "project-hivemind", "torch", "transformers", "tokenizers", "huggingface_hub", "safetensors",
    "peft", "accelerate", "numpy", "tqdm", "regex", "requests", "packaging", "filelock",
    "pyyaml", "fastapi", "starlette", "uvicorn", "pydantic", "aioquic", "websockets",
):
    datas += copy_metadata(dist)

excludes = [
    # GPU-only; the CPU build reports these features as unsupported instead.
    "bitsandbytes", "triton",
    "tkinter", "matplotlib", "IPython", "pytest", "torch.utils.tensorboard", "tensorboard",
]

a = Analysis(
    ["hivemind_hud_entry.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HiveMindHUD",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="HiveMindHUD")
