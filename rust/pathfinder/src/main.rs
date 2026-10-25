fn main() -> Result<(), pathfinder::runner::RunnerError> {
    let args: Vec<_> = std::env::args_os().collect();

    if args.len() != 3 {
        let program = args
            .first()
            .map(|arg| arg.to_string_lossy())
            .unwrap_or_else(|| "pathfinder".into());

        println!("Usage: {program} <grid_file_path> <model_file_path>");
    } else {
        pathfinder::runner::call(&args[1], &args[2])?;
    }

    Ok(())
}
