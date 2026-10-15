// A small chat in the terminal, with the reply streamed as it is written.
//   cargo run --release --example chat -- ../models/Qwen3.5-0.8B-Q4_K_M.gguf
use kiln::{Channel, ChatOptions, ContextParams, Message, Model, ModelParams, Sampling};
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: chat <model.gguf>")?;
    let model = Model::load(&path, &ModelParams::default())?;
    let info = model.info();
    eprintln!("{} {} on {}", info.arch, info.quant, info.backend_name);
    let mut ctx = model.context(&ContextParams::default())?;

    let mut history = Vec::new();
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        history.push(Message::user(line));

        let mut answer = String::new();
        let stats = ctx.chat(&history, &ChatOptions::default(), &Sampling::default(), |p| {
            if p.channel == Channel::Answer {
                answer.push_str(p.text);
                print!("{}", p.text);
                let _ = io::stdout().flush();
            }
            true
        })?;
        println!();
        eprintln!(
            "[{} tok at {:.1} tok/s, {} of {} prompt tokens reused]",
            stats.generated_tokens, stats.gen_tps, stats.cached_tokens, stats.prompt_tokens
        );
        history.push(Message::assistant(answer));
    }
    Ok(())
}
