#[cfg(test)]
mod fen {
    use chess_ai_core::game::Board;

    #[test]
    fn default() {
        let fen = Board::DEFAULT_FEN;
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }
}
