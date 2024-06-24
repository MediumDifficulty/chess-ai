use chess_ai_core::{movegen::MoveGenDiagnostics, Engine};

#[derive(Clone)]
pub struct FirstEngine;

impl Engine for FirstEngine {
    fn best_move(&self, board: &chess_ai_core::game::Board) -> chess_ai_core::game::Move {
        let moves = board.generate_legal_moves(&mut MoveGenDiagnostics::default());
        moves[0].clone()
    }
}
