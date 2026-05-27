# Rust / Winit / Egui / Wgpu Triangle (Minimal)

A stripped-down version of [wgpu-example](https://github.com/matthewjberger/wgpu-example) that renders a spinning triangle with [Rust](https://www.rust-lang.org/), [wgpu](https://wgpu.rs/), and [egui](https://github.com/emilk/egui). Native desktop and [WebGPU](https://www.w3.org/TR/webgpu/) only, with all the implementation in a single `src/lib.rs`.

> **Related Projects:**
> - [wgpu-example](https://github.com/matthewjberger/wgpu-example) - Full version with WebGL, Android, Steam Deck, and OpenXR
> - [wgpu-example-c](https://github.com/matthewjberger/wgpu-example-c) - C version
> - [wgpu-example-go](https://github.com/matthewjberger/wgpu-example-go) - Go version

## Prerequisites

- [just](https://github.com/casey/just) - Command runner
- A recent stable Rust toolchain (see `rust-toolchain`)
- [trunk](https://trunkrs.dev/) for the WebGPU build (`just init-wasm`)

## Quickstart

```bash
just run          # Native desktop, release build
just run-webgpu   # Serve the WebGPU build and open the browser
```

WebGPU runs in all Chromium-based browsers and in Firefox 141+.

## Commands

| Command | Description |
|---------|-------------|
| `just run` | Build and run the native desktop app |
| `just build` | Build the native app only |
| `just init-wasm` | Install the wasm target and trunk |
| `just run-webgpu` | Serve the WebGPU build at http://localhost:8080 |
| `just build-webgpu` | Build the WebGPU bundle into `dist/` |
| `just check` | Type check and verify formatting |
| `just lint` | Run clippy with warnings denied |
| `just test` | Run the test suite |

## Layout

`src/main.rs` is a thin desktop entry point. `src/lib.rs` holds the window handling, GPU setup, scene, and the wasm entry point.

## Controls

- **ESC** - Quit
- The window and canvas are resizable
