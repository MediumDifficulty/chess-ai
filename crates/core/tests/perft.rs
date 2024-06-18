// https://www.chessprogramming.org/Perft_Results

#[cfg(test)]
mod perft {
    use chess_ai_core::{
        game::Board,
        movegen::BoardCache,
        debug
    };
    use paste::paste;

    macro_rules! perft_test {
        ($i:expr, $fen:ident, $name:ident) => {
            paste! {
                #[test]
                fn [<$name _ $i>]() {
                    let mut cache = BoardCache::default();
                    let (diagnostics, move_count) = debug::gen_node_nums(&Board::from_fen($fen).unwrap(), &mut cache, $i - 1);
                    println!("{diagnostics:?}");
                    assert_eq!(move_count, [<$fen _NODES>][$i - 1]);
                }
            }
        };
    }

    const DEFAULT_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const DEFAULT_FEN_NODES: [usize; 6] = [20, 400, 8902, 197_281, 4_865_609, 119_060_324];

    const POS2_FEN: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const POS2_FEN_NODES: [usize; 6] = [48, 2039, 97862, 4085603, 193_690_690, 8_031_647_685];

    const POS3_FEN: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    const POS3_FEN_NODES: [usize; 6] = [14, 191, 2812, 43238, 674624, 11030083];

    const POS4_FEN: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    const POS4_FEN_NODES: [usize; 6] = [6, 264, 9467, 422333, 15833292, 706045033];

    const POS5_FEN: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    const POS5_FEN_NODES: [usize; 5] = [44, 1486, 62379, 2103487, 89941194];

    const POS6_FEN: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
    const POS6_FEN_NODES: [usize; 6] = [46, 2079, 89890, 3894594, 164075551, 6923051137];

    perft_test!(1, DEFAULT_FEN, default);
    perft_test!(2, DEFAULT_FEN, default);
    perft_test!(3, DEFAULT_FEN, default);
    perft_test!(4, DEFAULT_FEN, default);
    // perft_test!(5, DEFAULT_FEN, default);
    // perft_test!(6, DEFAULT_FEN, default);

    perft_test!(1, POS2_FEN, pos2);
    perft_test!(2, POS2_FEN, pos2);
    perft_test!(3, POS2_FEN, pos2);
    perft_test!(4, POS2_FEN, pos2);

    perft_test!(1, POS3_FEN, pos3);
    perft_test!(2, POS3_FEN, pos3);
    perft_test!(3, POS3_FEN, pos3);
    perft_test!(4, POS3_FEN, pos3);
    perft_test!(5, POS3_FEN, pos3);
    // perft_test!(6, POS3_FEN, pos3);

    perft_test!(1, POS4_FEN, pos4);
    perft_test!(2, POS4_FEN, pos4);
    perft_test!(3, POS4_FEN, pos4);
    perft_test!(4, POS4_FEN, pos4);

    perft_test!(1, POS5_FEN, pos5);
    perft_test!(2, POS5_FEN, pos5);
    perft_test!(3, POS5_FEN, pos5);
    perft_test!(4, POS5_FEN, pos5);

    perft_test!(1, POS6_FEN, pos6);
    perft_test!(2, POS6_FEN, pos6);
    perft_test!(3, POS6_FEN, pos6);
    perft_test!(4, POS6_FEN, pos6);
}
