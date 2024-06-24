use std::{
    fs::File,
    io::{stdin, stdout, Write},
    str::FromStr,
};

use crate::{
    game::{Board, Move},
    Engine,
};

pub fn start_uci(name: &str, engine: &impl Engine) {
    let mut log = File::create("C:\\Users\\Thomas\\Programming\\rust\\chess-ai\\log.txt").unwrap();

    respond(format!("id name {name}").as_str(), &mut log);
    respond("id author MediumDifficulty", &mut log);
    respond("uciok", &mut log);
    // respond("info HIIIII", &mut log);

    let mut msg = String::new();

    let mut board = Board::default();

    'main: while stdin().read_line(&mut msg).is_ok() {
        // for line in msg.lines() {
        log.write_all(format!("Received Message: {msg}").as_bytes())
            .unwrap();
        // writeln!(log, "{}", msg).unwrap();
        let mut segments = msg.split_whitespace();
        let command = segments.next().unwrap();

        match command {
            "isready" => respond("readyok", &mut log),
            "quit" => break 'main,
            "go" => respond(
                format!("bestmove {}", engine.best_move(&board)).as_str(),
                &mut log,
            ),
            "position" => {
                execute_position_command(segments.collect::<Vec<_>>().as_slice(), &mut board)
            }
            "d" => respond(&board.to_string(), &mut log),
            _ => (),
        }
        // }
        msg.clear();
    }

    log.write_all(b"Connection terminated.\n").unwrap();
    log.flush().unwrap();
}

fn execute_position_command(args: &[&str], board: &mut Board) {
    let mut args_i = args.iter();

    if args[0] == "fen" {
        args_i.next(); // skip "fen"
        *board = Board::from_fen(args_i.next().unwrap()).unwrap();
    } else {
        args_i.next(); // skip "startpos"
        *board = Board::default();
    }

    for &arg in args_i {
        if arg == "moves" {
            continue;
        }

        board.make_move(&Move::from_str(arg).unwrap())
    }
}

fn respond(msg: &str, log: &mut File) {
    println!("{msg}");
    stdout().flush().unwrap();
    log.write_all(format!("Responded with: {msg}\n").as_bytes())
        .unwrap();
    log.flush().unwrap();
}
