use std::io::stdin;

use chess_ai_core::uci;
use random_bot::RandomEngine;

fn main() {
    let mut msg = String::new();
    stdin().read_line(&mut msg).unwrap();

    if msg.trim() == "uci" {
        uci::start_uci("Rando", &RandomEngine);
    }
}
