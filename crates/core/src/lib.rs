use game::{Board, Move};

pub mod game;
pub mod movegen;
pub mod renderer;
pub mod uci;
pub mod util;
pub mod debug;

pub trait Engine {
    fn best_move(&self, board: &Board) -> Move;
}