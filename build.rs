use std::path::PathBuf;

/// Writes `$OUT_DIR/ui.rs`: the built web UI (`ui/dist`, from `pnpm build` in `ui/`) as `(path, bytes)` pairs for
/// `h3 serve` to embed. Without a build the list is empty and `/` says how to make one.
fn embed_ui(out: &std::path::Path) {
    fn walk(dir: &std::path::Path, root: &std::path::Path, files: &mut Vec<(String, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, files);
            } else {
                files.push((p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p.canonicalize().unwrap()));
            }
        }
    }
    let root = PathBuf::from("ui/dist");
    println!("cargo:rerun-if-changed=ui/dist");
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    println!("cargo:rerun-if-env-changed=H3_REQUIRE_UI");
    if files.is_empty() && std::env::var_os("H3_REQUIRE_UI").is_some_and(|v| v != "0") {
        panic!("H3_REQUIRE_UI is set but ui/dist is empty: run `pnpm install && pnpm build` in ui/");
    }
    let mut src = String::from("&[\n");
    for (path, abs) in &files {
        println!("cargo:rerun-if-changed={}", abs.display());
        src += &format!("    ({path:?}, include_bytes!({:?}) as &[u8]),\n", abs.display().to_string());
    }
    src += "]\n";
    std::fs::write(out.join("ui.rs"), src).unwrap();
}

fn main() {
    embed_ui(&PathBuf::from(std::env::var("OUT_DIR").unwrap()));
}
