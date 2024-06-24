use crate::{
    game::{Board, GameOutcome},
    movegen::{BoardCache, MoveGenDiagnostics},
};

pub fn gen_node_nums(
    board: &Board,
    cache: &mut BoardCache,
    depth: usize,
) -> (MoveGenDiagnostics, usize) {
    let mut diagnostics = MoveGenDiagnostics::default();
    let moves = board.generate_legal_moves(&mut diagnostics);
    let outcome = board.get_game_outcome(cache);
    if depth == 0 || outcome.is_some() {
        if let Some(outcome) = outcome {
            if matches!(outcome, GameOutcome::Checkmate) {
                diagnostics.checkmates += 1;
            }
        }

        return (diagnostics, moves.len());
    }

    let mut move_count = 0;

    for m in moves {
        let mut board_clone = board.clone();
        board_clone.make_move(&m);
        let (c, d) = gen_node_nums(&board_clone, cache, depth - 1);
        move_count += d;
        diagnostics += c;
    }

    (diagnostics, move_count)
}
