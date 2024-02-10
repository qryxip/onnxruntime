use anyhow::{anyhow, ensure, Context as _};
use cargo_metadata::{self as cm, MetadataCommand};
use itertools::Itertools as _;
use semver::Version;

fn main() -> anyhow::Result<()> {
    let md = &MetadataCommand::new().exec()?;
    check_cxxbridge_version(md)?;
    create_header_file(md)
}

fn check_cxxbridge_version(md: &cm::Metadata) -> anyhow::Result<()> {
    let cxx_version = &md
        .packages
        .iter()
        .filter(|cm::Package { name, .. }| name == "cxx")
        .exactly_one()
        .map_err(|e| match e.count() {
            0 => anyhow!("no `cxx` crate"),
            _ => anyhow!("multiple `cxx` crates"),
        })?
        .version;

    let cxxbridge_version = duct::cmd("cxxbridge", ["--version"])
        .read()?
        .strip_prefix("cxxbridge ")
        .and_then(|o| o.parse::<Version>().ok())
        .with_context(|| "could not parse the output of `cxxbridge --version`")?;

    ensure!(
        *cxx_version == cxxbridge_version,
        "cxx is v{cxx_version} but cxxbridge is v{cxxbridge_version}",
    );
    Ok(())
}

fn create_header_file(md: &cm::Metadata) -> anyhow::Result<()> {
    let pkg_id = md
        .workspace_members
        .iter()
        .exactly_one()
        .map_err(|_| anyhow!("multiple workspace members"))?;

    let cm::Target { src_path, .. } = md
        .packages
        .iter()
        .find(|cm::Package { id, .. }| id == pkg_id)
        .with_context(|| format!("`{pkg_id}` not found?"))?
        .targets
        .iter()
        .find(|cm::Target { crate_types, .. }| crate_types.contains(&"staticlib".to_owned()))
        .with_context(|| format!("no `staticlib` in `{pkg_id}`"))?;

    let content = duct::cmd("cxxbridge", [src_path.as_ref(), "--header"]).read()?;
    let dst = &md.target_directory.join("decrypt_vv_model").join("include");
    fs_err::create_dir_all(dst)?;
    fs_err::write(dst.join("decrypt_vv_model.h"), content)?;
    Ok(())
}
