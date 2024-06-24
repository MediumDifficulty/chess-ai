use chess_ai_core::{
    game::{Board, Move},
    movegen::MoveGenDiagnostics,
    Engine,
};
use rand::{thread_rng, Rng};

pub struct RandomEngine;

impl Engine for RandomEngine {
    fn best_move(&self, board: &Board) -> Move {
        let moves = board.generate_legal_moves(&mut MoveGenDiagnostics::default());
        moves[thread_rng().gen_range(0..moves.len())].clone()
        // moves[0].clone()
    }
}
