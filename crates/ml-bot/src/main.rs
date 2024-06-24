pub mod ppo;
pub mod supervised;

use std::{collections::HashSet, io::stdin, path::Path};

use chess_ai_core::{
    game::{Board, BoardPos, Move, Piece},
    uci,
};
use tch::{
    nn::{self},
    Device, Tensor,
};

/// Number of possible chess moves
const NUM_ACTIONS: i64 = 1968;

fn main() {
    println!("Starting...");
    let mut buf = String::new();
    stdin().read_line(&mut buf).unwrap();
    if buf.trim() == "uci" {
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);
        vs.load("C:/Users/Thomas/Programming/rust/chess-ai/crates/ml-bot/models/950.safetensors")
            .expect("Invalid model file");
        let model = ppo::model(&vs.root());
        let engine = ppo::MLEngine::new(model);
        uci::start_uci("Lamp", &engine)
    } else {
        // ppo::train();
        supervised::LichessDataset::load(Path::new(
            "D:/downloads/torrent/lichess_db_standard_rated_2024-05.pgn.zst",
        ))
    }
}

fn generate_output_map() -> [Move; NUM_ACTIONS as usize] {
    let mut moves = Vec::new();
    for rank in 0..8 {
        for file in 0..8 {
            let pos = BoardPos::new(file, rank);

            // Queen moves
            let offsets = [
                // Rook moves
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                // Bishop moves
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ];

            for offset in offsets {
                let mut current_pos = pos.add_offset(offset);

                while let Some(p) = current_pos {
                    moves.push(Move::new(pos, p));
                    current_pos = p.add_offset(offset);
                }
            }

            // Knight moves
            for offset in [
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ] {
                if let Some(p) = pos.add_offset(offset) {
                    moves.push(Move::new(pos, p));
                }
            }
            // King moves, castling and pawn moves are included in the queen's range of mobility
        }
    }

    // Promotions
    for file in 0..8 {
        // Iterate over ranks 1 and 7
        for (rank, offset) in [(6, 1), (1, -1)] {
            let pos = BoardPos::new(file, rank);
            // Iterate over pawn takes
            for offset in [(-1, offset), (0, offset), (1, offset)] {
                if let Some(p) = pos.add_offset(offset) {
                    moves.push(Move::new_promotion(pos, p, Piece::WBishop));
                    moves.push(Move::new_promotion(pos, p, Piece::WKnight));
                    moves.push(Move::new_promotion(pos, p, Piece::WQueen));
                    moves.push(Move::new_promotion(pos, p, Piece::WRook));
                }
            }
        }
    }

    moves.try_into().unwrap()
}

const BOARD_SIZE: i64 = 8;
const PIECE_PLANES: i64 = 13;
const ADDITIONAL_PLANES: i64 = 1;
const TOTAL_PLANES: i64 = PIECE_PLANES + ADDITIONAL_PLANES;
const BOARD_SQUARES: i64 = BOARD_SIZE * BOARD_SIZE;

const INT_INPUT_BITS: i64 = 6;
const NON_SPATIAL_DATA_LEN: i64 = INT_INPUT_BITS * 2;

const INPUT_SIZE: i64 = TOTAL_PLANES * BOARD_SQUARES + NON_SPATIAL_DATA_LEN;

fn num_to_binary_input(num: u64, size: i64) -> Tensor {
    Tensor::from_slice(
        &(0..size)
            .map(|i| ((num >> (size as u32 - i as u32)) & 1) as f32)
            .collect::<Vec<_>>(),
    )
}

fn get_inputs(board: &Board) -> Tensor {
    let mut layers = [[[0f32; BOARD_SIZE as usize]; BOARD_SIZE as usize]; TOTAL_PLANES as usize];

    // Add piece positions
    for file in 0..BOARD_SIZE as u8 {
        for rank in 0..BOARD_SIZE as u8 {
            let pos = BoardPos::new(file, rank);
            let piece = board[pos];
            layers[piece as usize][rank as usize][file as usize] = 1.;
        }
    }

    if let Some(en_passant) = board.en_passant() {
        layers[13][if board.white_to_move { 4 } else { 3 }][en_passant as usize] = 1.;
    }

    if board.castling().black_king {
        layers[13][7][6] = 1.;
    }

    if board.castling().black_queen {
        layers[13][7][2] = 1.;
    }

    if board.castling().white_king {
        layers[13][0][6] = 1.;
    }

    if board.castling().white_queen {
        layers[13][0][2] = 1.;
    }

    let spatial = Tensor::stack(&layers.map(|p| Tensor::from_slice2(&p)), 0);
    let non_spatial = Tensor::cat(
        &[
            num_to_binary_input(board.total_moves as u64, INT_INPUT_BITS),
            num_to_binary_input(board.halfmoves as u64, INT_INPUT_BITS),
        ],
        0,
    );

    let cat = Tensor::cat(&[spatial.view(-1), non_spatial], 0);
    debug_assert_eq!(cat.size(), vec![INPUT_SIZE]);
    cat
}

fn get_highest_ranked_legal_move(
    probs: &Tensor,
    legal_moves: &[Move],
    output_map: &[Move; NUM_ACTIONS as usize],
    board: &Board,
) -> usize {
    debug_assert_eq!(probs.size(), vec![NUM_ACTIONS]);

    let probs = Vec::<f32>::try_from(probs).expect("Incorrect Tensor type");
    let mut probs = probs.iter().copied().enumerate().collect::<Vec<_>>();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let legal_moves = legal_moves.iter().collect::<HashSet<_>>();

    for (i, _) in probs {
        let mv = &output_map[i];

        if legal_moves.contains(mv) && {
            let mut bc = board.clone();
            bc.make_move(mv);
            bc.repetitions() < 3
        } {
            return i;
        }
    }

    panic!("No legal moves found");
}
