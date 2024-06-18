#[cfg(test)]
mod perft {
    use chess_ai_core::game::Board;
    use chess_ai_core::movegen::BoardCache;
    use chess_ai_core::Engine;
    use random_bot::RandomEngine;

    const GAMES: usize = 10_000;

    #[test]
    fn random_game() {
        let bot = RandomEngine;
        let mut total_moves = 0u64;
        for g in 0..GAMES {
            let mut moves = 0u64;
            let mut b = Board::default();
            loop {
                b.make_move(&bot.best_move(&b));
                moves += 1;
                total_moves += 1;
                if let Some(outcome) = b.get_game_outcome(&mut BoardCache::default()) {
                    println!("{b}");
                    println!("{outcome:?}");
                    break;
                }
            }
            // println!("{}", g);
            
            println!("Game {} had {} moves total", g, moves);
            println!("Total moves {}", total_moves);
        }
    }
}