# Setup Guide

This project is named `rust_gpu`.

It targets a minimal Rust tensor runtime using `wgpu`, so the setup needs:

```text
Rust toolchain
wgpu-compatible GPU driver
basic build tools
optional benchmark tools
```

## Rust version

The current `Cargo.toml` uses:

```toml
edition = "2024"
```

Rust 2024 edition requires Rust `1.85` or newer. The recommended setup is the latest stable Rust through `rustup`.

Check your current version:

```bash
rustc --version
cargo --version
```

Install Rust with `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, restart the terminal or run:

```bash
source "$HOME/.cargo/env"
```

Use the stable toolchain:

```bash
rustup default stable
rustup update stable
```

Install useful Rust components:

```bash
rustup component add rustfmt clippy
```

Verify the project builds:

```bash
cargo check
```

## Linux system packages

On Ubuntu or Debian, install basic build tools:

```bash
sudo apt update
sudo apt install build-essential pkg-config
```

Install Vulkan utilities:

```bash
sudo apt install vulkan-tools
```

For Intel or AMD GPUs using Mesa:

```bash
sudo apt install mesa-vulkan-drivers
```

For NVIDIA GPUs on native Linux, install the appropriate proprietary NVIDIA driver from your distribution or from the official NVIDIA driver packages.

For NVIDIA GPUs inside WSL 2, do **not** install the Linux NVIDIA display driver inside WSL. Install the latest NVIDIA driver on Windows instead, then update WSL.

In PowerShell on Windows:

```powershell
wsl --update
wsl --shutdown
```

Then open WSL again and verify that the GPU is visible:

```bash
nvidia-smi
ls /dev/dxg
```

If `nvidia-smi` is not in `PATH`, try:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

For this project, CUDA is not required at first because `wgpu` uses graphics/compute backends such as Vulkan, DirectX 12, Metal, or OpenGL depending on the platform. CUDA is only needed if a later experiment compares against CUDA-specific tooling.

Verify Vulkan works:

```bash
vulkaninfo --summary
```

If `vulkaninfo` shows at least one GPU device, `wgpu` should have a usable backend.

## Rust dependencies

The first GPU prototype will likely need:

```bash
cargo add wgpu pollster bytemuck anyhow
```

For logging while debugging `wgpu` setup:

```bash
cargo add env_logger log
```

For benchmarks:

```bash
cargo add --dev criterion
```

If `cargo add` is not available, install `cargo-edit`:

```bash
cargo install cargo-edit
```

## Useful development commands

Format code:

```bash
cargo fmt
```

Run lints:

```bash
cargo clippy
```

Build:

```bash
cargo build
```

Run:

```bash
cargo run
```

Run tests:

```bash
cargo test
```

Run benchmarks after benchmark files exist:

```bash
cargo bench
```

## Optional tooling

Install `cargo-nextest` for faster test runs:

```bash
cargo install cargo-nextest
```

Run tests with:

```bash
cargo nextest run
```

Install `cargo-deny` if dependency/license checks become useful:

```bash
cargo install cargo-deny
```

## First setup checklist

```text
1. rustc --version shows 1.85 or newer
2. cargo check passes
3. vulkaninfo --summary shows a GPU
4. rustfmt and clippy are installed
5. wgpu dependencies are added once implementation starts
```

## Run nvidia-smi

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 2
```