use game::{Board, Move};

pub mod debug;
pub mod game;
pub mod movegen;
pub mod pgn;
pub mod renderer;
pub mod uci;
pub mod util;

pub trait Engine {
    fn best_move(&self, board: &Board) -> Move;
}
